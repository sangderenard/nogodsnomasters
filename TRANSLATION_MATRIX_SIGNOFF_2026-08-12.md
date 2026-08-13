# Language translation matrix — sign-off (code-grounded)

**Date:** 2026-08-12
**Method:** every verdict below was checked against the compiler source under
`turing/src/compiler` and `turing/src/common/tensors/accelerator_backends`, and
against test suites re-run in this session — not against the handoff prose. Where
a handoff doc and the code disagreed, the code wins and the disagreement is noted.
Toolchain probed on this box: `llvmlite 0.47.0`, `torch 2.5.1+cu124` (CUDA
available), `pycparser 3.00`, `node` present; **no `gfortran`, no `wat2wasm`**.

**Sign-off legend:** ✅ signed (verified) · 🟡 conditional (domain stated) ·
🔴 blocked (named blocker) · ⛔ quarantined (retained machine dialect, must not be
advertised as repository SSA) · ⬜ open obligation · — no obligation.

## The two spines, from the code

Everything routes through **repository SSA** — `transmogrifier/ssa.py`'s
`IRModule`/`Function`/`BasicBlock`/`Instr` over the `Handler` vocabulary
(`ssa_registry.py`). Two families of front/back ends hang off it:

```text
FRONT ENDS  (→ repository SSA / ProcessGraph)          BACK ENDS  (SSA / FusedProgram →)
Python AST/class   ast_process_graph, topological_reducer   AbstractTensor   numpy/torch/jax/pure/C
SymPy              symbolic_process_graph                    C (one boundary) c_primitive_program, c_jit_backend
GLSL source        glsl_source_ingestion (scalar slice)      LLVM IR (emit)   llvm_jit / llvm_optimizing_pipeline
JavaScript expr    javascript_process_graph (ESTree)         Fortran 2008     ssa_fortran_backend, fortran_jit
C / C++-shell      cpp_shell_desugar → pycparser → lifting   GLSL compute     fused_program_*_backend, ssa_webgl
LLVM IR (import)   accelerator_backends/llvm_repository_ssa   WebGL ES frag    ssa_webgl_backend
CPython native     cpython_compile_ssa                       WebGPU WGSL      ssa_webgpu_backend
                                                              SPIR-V asm       ssa_spirv_backend
                                                              WebAssembly      fused_program_wasm_backend, wasm_binary
                                                              Nodus IR/arena   nodus_graph_ir, nodus_backend
                                                              analog tape      tape_compiler → hardware/analog_spec

MACHINE SPINE (own dialect, quarantined from repository SSA)
x86-64 PE bytes ──X86ReferenceDecoder (scalar, authoritative)──┐
      ▲                                                        ├─→ machine_dialect_ssa ──legalize──> repository SSA
      └── write_reverse_selection (byte-exact encode) ─────────┘        ⛔ gate: repository_ssa_legalized()
x86-64 tensor lanes ── X86TensorReadHead / X86ReversibleReadHead (read/write/execute, journaled, forkable)
machine trace/state ── machine_trace_ssa, machine_stream_interposition (specialized replay, NOT decompilation)
```

The machine spine is the "bidirectional head." Note there are **two** x86 decode
surfaces and they are not the same object: the scalar `X86ReferenceDecoder`
(`machine_reference_vocabulary.py:3876`) is authoritative; the tensor
`X86TensorReadHead` (`x86_tensor_read_head.py:1036`) is an
accelerator/verifier that runs many lanes and journals reversible state. Both are
x86-specific in their *driver* even though the scalar one is *data-driven over an
`InstructionSpec` vocabulary*.

## The matrix

Rows = source, columns = target. Composed cells inherit the weakest leg. LLVM is
a full row and column because it is genuinely bidirectional here (import *and*
emit).

| from \ to | RepoSSA/PG | AbsTensor | C | LLVM IR | Fortran | GLSL/WebGL | WebGPU | SPIR-V | WASM | Nodus | JS | Page | x86 native | x86 bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Python** | ✅¹ | ✅ | ✅ | ✅⁵ | 🟡⁶ | ✅⁷ | 🟡⁸ | 🟡⁹ | 🔴¹⁰ | 🟡¹¹ | 🟡¹² | 🟡¹³ | ⬜ | ⬜ |
| **SymPy** | ✅² | ✅ | ✅ | ✅ | 🟡 | ✅ | 🟡 | 🟡 | 🔴 | 🟡 | 🟡¹² | 🟡 | — | — |
| **GLSL src** | 🟡³ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🔴 | 🟡 | — | 🟡 | — | — |
| **JavaScript** | 🟡¹² | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🔴 | 🟡 | 🟡¹² | 🟡 | — | — |
| **C / C++-shell** | 🟡⁴ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🔴 | 🟡 | — | 🟡 | — | ⬜ |
| **LLVM IR** | ✅⁵ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🔴 | 🟡 | — | 🟡 | — | ⬜ |
| **CPython native** | 🟡¹⁴⛔ | — | — | — | — | — | — | — | — | — | — | 🔴 | ✅ | ✅ |
| **x86-64 PE bytes** | 🟡¹⁵⛔ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | — | 🔴¹⁶ | ✅¹⁶ | ✅¹⁵ |
| **x86 machine trace** | ⛔¹⁷ | ⛔ | ⛔ | ⛔ | — | — | — | — | — | — | — | ⛔ | ⛔¹⁷ | ⛔¹⁷ |

## Front-end edges (into the hub)

**1. Python → ProcessGraph/SSA — ✅.** `ast_process_graph` + `topological_reducer`
ingest whole functions and whole `ClassDef`s (class table, methods as compiled
calls, shared field state). `if`→`select`; `try/except` reduces to on-demand
resolution; unsupported syntax retained as `opaque_python`, never executed.
**Code-confirmed correction to the older doc:** the front door is no longer
`compile_ast_aot` — that path now emits a `DeprecationWarning` at
`aot_compile.py:665` pointing to `fortran_c_shell.lower_ast_source_to_ssa`
(`fortran_c_shell.py:2979`): "ingests one complete authored program and lowers
control, arithmetic, tensors, calls, and memory directly to repository SSA…never
projects a numerical FusedProgram." Residue unchanged: semantic importer still
leaves bare `for`/`while` opaque in the linear SSA builder.

**2. SymPy → ProcessGraph — ✅.** `symbolic_process_graph.py` is the authoritative
math front end; not superseded by the Python importer. Open item: SymPy and
Python still build partially parallel vocabularies onto the same `Handler` set.

**3. GLSL source → SSA — 🟡 (scalar straight-line slice).**
`glsl_source_ingestion.py` evaporates recognized syntax straight into `Handler`
instructions; `mix`/`clamp`/`smoothstep` decompose to arithmetic+select; aliases
die at the source edge; texture/derivative ops are named shortfalls, not guessed.

**4. C / C++-shell → SSA — 🟡.** `cpp_shell_desugar.py` rewrites a *narrow*
C++-like shell (single-inheritance struct embedding, `obj.method`→`Class__method`,
no virtuals/templates/overloading — all fail-closed as `CppShellUnsupported`) into
real C, handed unchanged to the `pycparser`-based `machine_code_lifting` route.
`pycparser 3.00` present. `test_cpp_shell_desugar` + `test_dream_cpp_pipeline`
pass this session.

**5. LLVM IR ↔ repository SSA — ✅ bidirectional (this is the row I missed).**
- **Import (LLVM → repo SSA):** `accelerator_backends/llvm_repository_ssa.py`
  translates LLVM instructions into the existing `Handler` vocabulary via a direct
  table (`add/fadd→Add`, `icmp` predicates→`Eq/Lt/ULt/…`, `getelementptr→GetElementPtr`,
  `phi→Phi`, `select`, memory ops) and **expands `switch` terminators into
  `Eq`/`CondBr` chains** — it creates no new opcodes. `test_llvm_repository_ssa.py`
  (20 tests) passes.
- **Emit (repo SSA → LLVM):** two deliberately separate paths. `llvm_jit_backend.py`
  is the unoptimized MCJIT reference (`opt=0`, generic scalar — kept as a
  differential oracle). `llvm_optimizing_pipeline.py` is the vectorizing
  counterpart: it documents the exact three things LLVM needs and default-misses —
  a `PassBuilder` speed level, a named host target, and `noalias` facts — and adds
  them, with fast-math **off** by default because the repo verifies backends
  numerically against each other. `test_llvm_signal_math` / `test_ssa_fortran_and_optimizing_llvm`
  pass.
- **Environment caveat:** the LLVM *JIT-execution torture* suite has 6 failures
  this session (`test_llvm_jit_backend.py::…torture_baseline…[where|cumsum|advanced_tensor_topology]`
  and the advanced-layout case) — these exercise the profiled-C-shell execution
  boundary, not the IR translation; they are toolchain/execution-path failures,
  and the translation-layer suites above are green.

## Back-end edges (out of the hub)

**6. → Fortran 2008 — 🟡 (emit ✅ / assemble env-gated).** `ssa_fortran_backend.py`
targets Fortran precisely for its no-alias guarantee (vectorization C/LLVM can't
get without `restrict`/`noalias`); `iso_c_binding`/`bind(C)` so it drops into the
same shell ABI; `Phi` eliminated by predecessor assignment. Emission needs no
compiler; **assembly needs gfortran, which is absent here**, so
`fortran_jit_backend`/mandelbrot-IO execution paths can't complete on this box.

**7. → GLSL compute / WebGL — ✅ / 🟡.** `fused_program_*_backend` + `ssa_webgl_backend`.
56 canonical GLSL primitives; multi-output shaders share intermediates; the
Mandelbrot region proof (153 ops → one 4-output dispatch, count field exact vs
NumPy, Y/Cb/Cr within ~1.6e-5) is the standing evidence. WebGL fragment path:
texture/derivative ops explicitly unlowered. `test_ssa_webgl_source_roundtrip`
passes.

**8. → WebGPU WGSL — 🟡, with 2 live failures.** `ssa_webgpu_backend.py` emits WGSL
compute directly from SSA with a real binary/unary op table. This session:
`test_webgpu_ssa_backend.py::test_ast_generated_float32_program_emits_wgsl_compute`
and `::test_float64_is_a_named_webgpu_core_shortfall` **fail** (assertion-level, in
emission — not toolchain). Real regression to trace, not an environment gap.

**9. → SPIR-V — 🟡 (scalar + array-shaped elementwise, single-block).**
`ssa_spirv_backend.py`: module-scoped id/type/constant namespace via
`_ModuleBuilder`; array elementwise ops **unrolled** (shapes static); structural
ops/reductions/multi-block control flow reported as honest shortfalls; SSBO/
descriptor binding ABI deliberately out of scope. `test_ssa_spirv_backend.py`
(11 tests) passes.

**10. → WebAssembly — 🔴 blocked, and the vocabulary is drifting under its tests.**
`wasm_binary.py` is a real from-scratch binary assembler (LEB128, IEEE-754,
per-value-type opcode maps for f32/f64/i32/i64) so no `wat2wasm` is needed;
`fused_program_wasm_backend.py` is the lowering. This session it has the most
churn of any target:
- `test_wasm_fidelity.py`: **23 failures**, all one root — the differential
  verifier at `wasm_fidelity.py:268` does `actual[finite]` where `actual` has size
  1 but the mask has size 6 (`IndexError: boolean index did not match`). The
  *verifier harness* is broken, which masks whatever the emitter now does.
- `test_machine_targets_wasm.py`: 2 failures because the capability catalogue is
  **stale in the safe direction** — the test asserts `{tan, pow, mod, sign} ⊆
  unsupported`, but `mod`/`pow`/`sign` are now *supported*, so the WASM target
  grew ops the test still lists as missing. Same story in
  `test_wasm_binary.py::test_the_catalogue_decides_what_is_reachable`
  (`{isnan, mod, tan}` no longer all unsupported).
- `test_read_head_wasm_state_machine.py::test_it_computes_in_integer_instructions_not_floating_point`
  fails (1).
These are not the flagship `explore_forking_paths` state-feedback ABI blocker
(`site_bundle.py:2297`) from the handoff — they are emitter/verifier/catalogue
drift found by re-running, and they sit directly under the head-fluency work.

**11. → Nodus — 🟡.** `nodus_graph_ir.py` exports tool graphs; `nodus_backend.py`
routes elementwise dispatch + matmul through the arena with no silent NumPy
fallback (each unimplemented op named at the dispatch). `test_nodus_backend` /
`test_nodus_graph_ir` pass. Un-met milestone: executing a multi-region
ProcessGraph-derived program inside Nodus.

**11b. Nodus KernelIR → SPIR-V — ✅ (new, 2026-08-13).** Previously the only
route was `SpirvTranslator` (KernelIR → GLSL → external `glslangValidator`),
and its `GlslEmitter` was a stub emitting `void main() { }` for every kernel —
zero instructions ever lowered. `src/kernels/kernel_spirv.cpp` now assembles
SPIR-V 1.3 binaries **directly** (no external tool; same rationale as turing's
`wasm_binary.py`): module-scoped pooled types/constants, Block-decorated SSBOs
from `buffer_value_ids`, a `gl_GlobalInvocationID.x` bounds guard from
`element_count` (published to ADDR under the `kGlobalIndexValue` sentinel), and
lowering for AND/OR/XOR/NOT, BINARY/UNARY/CMP/CAST via `CanonicalOp` sub_ops,
SELECT, ADDR/LOAD/STORE — everything else refused by name. It is also the
**first real `TranslationMatrix` registration** ("spirv"). Verified: the
Tier-1 gray-code recipe (`bitops_lowering.h`, unmodified) assembles to a
197-word module that passes `spirv-val --target-env vulkan1.1`, disassembles
to the exact intended semantics, and the ATOMIC refusal fires;
`tests/test_spirv_assembler.cpp` + CMake wiring (`nodus_tensor_core`,
`nodus_runtime`, `spirv_assembler_test`) added. The GLSL-mediated path is
retained as the future differential reference. Full nodus CMake/vcpkg build
not re-run this session; the new test target compiles and passes standalone
with the target's own flag set.

**11c. Repository SSA → KernelIR → SPIR-V — ✅ end-to-end (2026-08-13), no
FusedProgram on the path.** `turing/src/compiler/kernel_ir_lowering.py` is the
emission-side mirror of `repository_ssa_legalized`: an SSA instruction crosses
into KernelIR only when its op spelling resolves through
`nodus/ops/canonical_ops.json` (canonical name or Handler member) to an
append-only canonical ID, which becomes `sub_op`; anything else is a named
shortfall. One SSA `Function` → one named kernel (structure preserved,
multi-function programs stay multiple kernels). The elementwise frame maps
args to read-only SSBOs and the output to a writable one, indexed by the
assembler's `kGlobalIndexValue` sentinel. Interchange is KIRTEXT v1
(`serialize_kernel_ir` ↔ `nodus/src/kernels/kernel_ir_text.cpp`, a tiny owned
line format — no JSON dependency), with `kir_to_spirv` as the CLI crossing.
Verified: a u32 gray-code chain and an f32 mul/add/tanh chain, both hand-built
in real repository SSA with mixed op spellings, lower → serialize → parse →
assemble → **pass `spirv-val --target-env vulkan1.1`**, with exact semantics
confirmed in disassembly. Tests: `turing/tests/test_kernel_ir_lowering.py`
(6 green) and the extended `nodus/tests/test_spirv_assembler.cpp`, which
parses verbatim Turing KIRTEXT. SPIR-V is thereby the final device form
*below* KernelIR, not an inter-repo intermediary. Remaining for GPU
execution: the Vulkan dispatch host (`kernel_vulkan.cpp` is still a stub);
scope frontiers (multi-block control via KernelIR `IF`, f64, reductions) are
named shortfalls, not gaps.

**12. JavaScript ↔ ProcessGraph — 🟡 bidirectional.** `javascript_process_graph.py`
ingests one ESTree expression (from `vendor/js_ast_parse.js`) into the *identical*
canonical node schema SymPy/GLSL use — verified against the real AOT/SSA pipeline,
not an approximation (`test_javascript_process_graph.py`, 6 tests, pass) — and
renders nodes back to JS. Operators with no `Handler` equivalent are shortfalls.

**13. → HTML page bundle — 🟡, class-entry gap confirmed in code today.**
`build_program_bundle` lands immutable bundles under `site/programs/…`, but
`discover_source_contract` (`site_bundle.py:895–898`) still filters `module.body`
to `ast.FunctionDef` and raises `"source defines no public top-level function"` —
**no `ClassDef` entrypoint branch**, even though whole-class ingestion exists a
layer down and two unwired whole-class-to-page mechanisms exist
(`parametric_card_program.py`, `wasm_class_coordinator.py`).

## Machine spine (the bidirectional head)

**14. CPython native → repository SSA — 🟡 ⛔.** `cpython_compile_ssa.py` lifts
CPython's own `compile` through the PE parser + AMD64 vocabulary + CFG lifter +
`machine_dialect_ssa`, never invoking CPython's compiler and leaving no runtime
native call. It consumes the `repository_ssa_legalized` gate. Long recursive
extraction remains user-authorized only.

**15. x86-64 PE bytes ↔ dialect ↔ repository SSA — ✅ decode/encode, 🟡 legalize, ⛔ boundary.**
- Decode: `X86ReferenceDecoder` is data-driven over `X86_64_REFERENCE_VOCABULARY`,
  a tuple of `InstructionSpec` records each carrying opcode/mask/ModRM-extension/
  REX/legacy-prefix rules **and** `reversible_*` layout metadata; the constructor
  builds a first-byte dispatch table and rejects duplicate/ambiguous encodings.
  Reference decode is authoritative; tensor lanes cannot narrow it.
- Encode (write-back): `_REVERSIBLE_LAYOUTS` + `_attach_reversible_layout` infer a
  byte-exact reversible layout per token (families → `REG_RM`/`RM_REG`/`RM`/
  `RELATIVE`/`IMMEDIATE`/…), and `plan_reverse_selection`/`write_reverse_selection`
  (`binary_ingestion.py:1150/1225`) drive the write head. Byte-exact roundtrip of
  *decoded* tokens is real (`test_machine_code_lifting_roundtrip`,
  `test_x86_reversible_read_head`, `test_pe_recompilation` all pass this session).
- Legalize to repository SSA: `machine_dialect_ssa.py`; the two named residuals
  (`SCASB`, `LOCK_ADD_RM8_R8`) still lack CFG SSA lowering.

**⛔ The quarantine boundary — unchanged and load-bearing.** `repository_ssa_legalized(function)`
exists (`machine_dialect_ssa.py:49`), is tested (`test_machine_dialect_ssa`, green),
and is consumed by `cpython_compile_ssa`. A function is repository SSA **only after
every machine op legalizes** to ordinary control/arithmetic/memory. Retained
machine dialect may reuse `IRModule`/`Function` containers; that reuse must never
be advertised as repository SSA. Universal emitter-side rejection is still partial
(restart step 3 of the continuation doc): predicate ✅, blanket enforcement ⬜.

**16. x86 native execution / → page — ✅ native / 🔴 page.** `binary_machine_program.py`
+ `machine_execution.py` run the reversible AMD64 machine natively, forward and
reverse, with `system_tape` and path-forest forking. Never compiled into a page:
all six published bundles still carry `prebuilt-program-interior`.

**17. x86 machine trace → SSA → modified SSA → re-slipped execution — ⛔ (in flight, external).**
`machine_trace_ssa.py` explicitly says it "specializes the observed run…does not
claim to be a whole-program symbolic decompilation." `machine_stream_interposition.py`
inserts a separately-laid-out, read-head-framed, byte-decoded instruction stream at
a trigger address — live guest registers/flags/memory, only instruction provenance
changes (`test_machine_stream_interposition`, 13 tests, pass). This is the codex
agent's read-ahead → lift → modify → re-encode → interpose track; the removed
demos were removed for labeling retained dialect as repository SSA. **Nothing on
this row may be claimed until the returning work lands behind the
`repository_ssa_legalized` gate.**

## Test evidence gathered this session

Green (translation layer, no toolchain needed): `test_llvm_repository_ssa` (20),
`test_ssa_spirv_backend` (11), `test_ssa_webgl_source_roundtrip`,
`test_javascript_process_graph` (6), `test_cpp_shell_desugar`,
`test_dream_cpp_pipeline`, `test_machine_dialect_ssa`, `test_llvm_signal_math` —
**103 passed together**; plus machine-spine `test_machine_stream_interposition`
(13), `test_machine_trace_ssa`, `test_pe_recompilation` (7),
`test_machine_code_lifting_roundtrip`, `test_x86_reversible_read_head` (12),
`test_glsl_source_ingestion`, `test_dream_document`, `test_nodus_backend`,
`test_nodus_graph_ir`, `test_wasm_class_coordinator` — **178 passed** in that batch.

Red clusters (all characterized above): tape→LLVM/C direct lowering with an empty
`forward_capture` tape (5, `test_c_backend_llvm_ssa`), LLVM JIT torture execution
(6, toolchain), WebGPU emission (2), WASM fidelity verifier (23, one root),
WASM catalogue drift (3), read-head-WASM integer test (1).

## Shortest cell-flips

1. **🔴→✅ WASM fidelity:** fix the size-1-vs-6 boolean-mask bug at
   `wasm_fidelity.py:268` — it is masking the whole emitter's real status.
2. **stale→✅ WASM catalogue:** update `test_machine_targets_wasm` /
   `test_wasm_binary` to the grown op set (`mod`/`pow`/`sign`/`isnan` now
   supported) — the capability moved, the assertion didn't.
3. **🟡→✅ machine→repo SSA:** legalize `SCASB` + `LOCK_ADD_RM8_R8`, then finish
   emitter-side enforcement of `repository_ssa_legalized`.
4. **🟡→✅ Python(class)→page:** add the `ClassDef` branch to
   `discover_source_contract`, converging the two existing whole-class-to-page
   mechanisms rather than adding a third.
5. **⛔→trace-row sign-off:** land the returning interposition work behind the
   quarantine gate with an independent differential witness.

## Standing rules re-affirmed (from the code, not the prose)

1. Never advertise retained machine dialect as repository SSA
   (`machine_dialect_ssa.repository_ssa_legalized` is the enforcement point).
2. No numerical projection / runtime fallback / silent op deletion on the machine
   path (`nodus_backend` and `machine_reference_vocabulary` both fail-closed by
   construction).
3. A source-edit that hides one program's symptom is not a compiler fix.
4. An empty matrix cell is a failed proof obligation, not "unsupported."
5. No unrequested long recursive compiles; no final fused reduction for
   multi-function programs.
