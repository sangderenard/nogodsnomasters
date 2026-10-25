# Sangderenard compiler and mathematical systems research

## Technical value and the October 2026 convergence problem

Prepared for Albert Stichka by dot

Research date 21 October 2026

## 1 The reason to continue and the reason to change the test of progress

There is a technically defensible reason to continue the core of this work. The repositories contain more than a proposed compiler, a collection of APIs or a future mathematical framework. They contain numerical representations that preserve information ordinary floating point loses, authored tests that take graph-derived derivatives into native execution, recorded native numerical results checked against independent references, and physical models assembled from named mathematical laws. Some of those paths have produced useful executed outcomes. The work has substance even though the current compiler bootstrap has not converged.

The reason to continue is specific. You are trying to make complicated mathematical machinery cheap to author, change, differentiate and deploy: custom numerical semantics and precision, calculations expressed as graphs, native or ahead-of-time output, and eventually constrained WebAssembly rather than a permanently expensive Python interpreter underneath. You also want the representations to help people understand the sets, graphs and systems they are manipulating. That goal is broader than writing a faster float64 matrix multiply and narrower than supporting every possible Python program. The strongest evidence should be judged against that goal.

The evidence does not justify continuing every current workstream unchanged or assuming the eventual payoff is inevitable. The central gap is trustworthy composition. Individual representations, algorithms, runtime paths and backends exist, but their conjunction is not yet dependable. A source program can lower and link while its sort has disappeared, its returned record is missing from the physical interface, or a consumer holds a stale dtype. More features and more admitted units do not, by themselves, close that gap. [R01] [R02] [L01]

October has produced both engineering progress and bounded executed successes. The October 3 compiled 7 by 19 Jacobian matched SymPy; N2 numerical parity survived later structural repairs; a 27-piece craft command-change test passed physical assertions. It would be wrong to erase those results. It would also be wrong to substitute them for the outcome you are waiting for. The newest local records still contain no natively verified provenance-bootstrap units and no completed semantic execution of the Tire workload. The fresh October 11 N4 result was a source-to-native build and structural audit, not numerical execution of that system. [R03] [R04] [R05] [L01]

My assessment is therefore conditional: preserve the mathematical and compiler thesis, but make its next justification a reproducible named result rather than implementation volume or a changing fail-fast numerator. Native provenance-first bootstrap is a legitimate route to reducing compiler iteration cost. A bounded scientific program is a more direct route to proving user value. Neither route automatically proves the other. The choice should be explicit, with a stable acceptance contract and evidence that the selected work is reducing the particular uncertainty that matters.

This report separates current source inspection, historical execution receipts and recommendations. The investigation was read-only. No compiler build, test suite, benchmark, bootstrap or generated program was run for this report. Historical results certify their recorded source states, not the heavily modified local tree observed on 21 October.

## 2 What the repositories are trying to make possible

### A common mathematical and structural purpose

The projects share a coherent concern: a mathematical or structural definition should remain recognizable while it becomes executable. In Turing, that means tensor operations, ProcessGraph, repository SSA, target code and identity or provenance records. In the physical-law work, a named equation is transformed into a piece used by a managed timestep system. In Nodus, graph structure and declared ports connect registered computation to runtime surfaces. In SpeakToMe, a causal search graph preserves alternatives and relationships that would disappear if only a selected answer were retained. [R06] [R07] [R08]

These are related ideas, but relatedness is not proof that the whole collection forms one finished system. Several projects already have independent execution paths. Numerical laws can use hosted Turing compilation without a self-hosted compiler. Nodus can execute registered native tools. Pluck has direct C++ and GPU rendering. SpeakToMe uses established model and tensor backends. A failure in the compiler bootstrap does not make all of that code inert, and success in one of those paths does not establish the others. [R08] [R09] [R10]

### Repository roles and current source states

The local root at C:\dev\Powershell is primarily a coordination and demonstration repository, but it is not literally root-only. Its Git tree tracks engine_toy and site, together with top-level assets and research material. Turing, Nodus, SpeakToMe and spectral-analyzer remain independent repositories. A root merge therefore does not transfer a sibling compiler change. The current ownership matters when reproducing a cross-repository result. [L02]

| Repository | Role in the overall work | Local state observed on 21 October |
| --- | --- | --- |
| nogodsnomasters | Coordination, physical-law demonstrations, engine_toy and site artifacts | Branch nogodsnomasters at a67e031, ahead by three; engine_toy artifacts untracked |
| Turing | Numerical representations, graphs, differentiation, SSA, native and target compilation, provenance | Main at 65b4fc0657e2, plus 47 modified tracked files and untracked convergence reports 23 through 32 |
| Nodus | Graph and runtime interfaces, ports, registered operations and execution | Main checkout branch nogodsnomasters at 4500930, clean; a separate Python graph worktree at a3a3986 contains October work |
| SpeakToMe | Branching language-model search, causal graph and experimental analysis | Branch codex/direct-x86-vocabulary-report at 43de955, ahead by seven, with modified and untracked work |
| spectral-analyzer or Pluck | Optical and related physical computation, render/runtime integration | Local 1fa5752, ahead by one, with uncommitted electrical-network source and tests |
| geometry and other siblings | Supporting research or separate projects | Geometry inspected as explicitly partly scaffolded; AMP and FFTFree oriented; PCB, Pigeon, Socratic and Meta-NN chiefly inventoried |

The published sources examined do not always equal the local states above. Remote Turing was pinned at 65b4fc0657e242663aec5812c7604ad433e06261; root at 49d2de12ecb0efe2d133e9c5eeb8335aa3a6f768; Nodus at 45009309781daa074a5ba69e253427a6702a1326; spectral-analyzer at ffe5150eedbd35326b5f92f2be967730f00f9b22; and SpeakToMe at 95df7ffa81bd8ed8a98ed7f3b46a310e61256152. Public references below use those inspected snapshots. New local findings are identified separately rather than falsely linked to unchanged public files. [L02]

### The complete promise is a conjunction

The distinctive target combines several demands: author a mathematical computation naturally, preserve a nonstandard number representation, differentiate it correctly, produce an actual target artifact, fit a deployment budget, and obtain enough speed or development leverage to justify compilation. Each demand is meaningful. Their combination is where most of the remaining risk lies.

For example, an eight-limb arithmetic test, an ordinary-precision native training test and a WASM matmul test do not collectively prove eight-limb training in constrained WASM. The missing evidence is the program that crosses those boundaries with its semantics intact. Conversely, that conjunction should not become the acceptance target for every next patch. It is better approached through explicit stages than by making the first useful result encompass the entire ambition.

## 3 Technical achievements that justify taking the work seriously

### Custom number semantics are implemented

Turing's extended_precision.py represents floating expansions as ordinary tensor channels. It also implements rational, complex and combined representations rather than silently treating every value as a scalar float. Precision remains explicit until a deliberate collapse. That design is relevant to your aim because the representation can be visible to graph construction and lowering instead of being an opaque host-language object. [R06]

The machinery includes error-free transforms, renormalization, width-preserving arithmetic and reductions, and distinct treatment of complex components and expansion limbs. Binary64 and binary32 limbs use different splitting constants. Strict floating-point behavior is part of the contract because reassociation can destroy the correction terms that give an expansion its accuracy. Multiplication forms limb products and renormalizes them; extra precision therefore has real computational and storage cost. It is not eight cheap floats magically becoming an arbitrary-precision scalar. [R06]

The authored acceptance tests are unusually relevant to the claim. The precision surface checks that more limbs buy information a double cannot retain and compares constants such as pi, e and logarithms against Fraction or mpmath references. Complex tests include information ordinary complex128 loses and a source-to-LLVM execution path. Rational tests cover structural quotients, promotion independent of operand order and combinations of numeric features. The composite rational compiler tests are chiefly structural lowering evidence, not an established broad native-execution result. None of those tests was rerun here. [R11] [R12] [R13] [L03]

Eight binary64 limbs should not be described as a universal 512-bit arithmetic guarantee. Each well-behaved limb contributes approximately the significand information of its underlying floating format, subject to the operation, conditioning and normalization. Nor does structural rational arithmetic automatically provide unlimited exact integer arithmetic; the numerator and denominator tensor representations still determine its limits. The tests and implementation support a concrete numerical design, not unrestricted arbitrary precision.

A historical eager timing receipt also exposes the need for compilation. For 256 elements and six divisions, it records about 0.563 ms for ordinary real values, 161.131 ms for Precision[2], 416.972 ms for RationalPrecision and 5420 ms for ComplexRationalPrecision, with different plain real or complex baselines appropriate to each representation. These costs cannot be summarized as the price of one identical operation, because the semantics differ. They show that rich representations are expensive in the current eager machinery. They do not show that native lowering has already removed that expense. [L03]

### Graph differentiation reaches real native computations

The ProcessGraph autograd test bodies go beyond a tape or a backward-callback API. They encode native execution for linear forward and loss calculations, rectangular convolution adjoints, an AbstractNN XOR model, and an 80-step SGD loop. The XOR test explicitly checks that Python backward callbacks are absent, derives a combined graph, lowers it to repository SSA, emits LLVM, compares gradients with explicit NumPy expressions, and checks training behavior. This is authored executable coverage, not a current run receipt. [R14]

There is also recorded execution evidence. On October 3, a symbolic orbital-law path was differentiated, lowered and compiled. Its 7 by 19 Jacobian had 55 nonzero entries and matched an independent SymPy reference with maximum relative error 4.138e-16. Forward values matched within 2.39e-15. That result required actual compiler repairs: graph-native backward calls had been silently dropped when the planner expected an AST Call, copied SSA references collided, specialization lost tensor facts, and mixed integer/floating where branches produced incorrect derivatives. Correcting those failures and obtaining an independent numerical match is a substantive result. [R03]

The limits are equally concrete. Lowering took roughly 235 seconds. Broader scalar-loss and adjoint paths still had a shape disagreement and a 420-second timeout. A separate October 3 reverse-precision experiment was explicitly parked and unverified, with numeric-class or limb facts lost across selected return producers. That experiment is not the successful ordinary-precision Jacobian result, and the earlier October 2 feasibility blocker should not be presented as though the October 3 success never happened. Later focused reverse tests also did not recertify the full Jacobian on today's source. [R03] [R15] [L04]

The implemented ProcessGraph differentiation lane should also be distinguished from the separate autoautograd whiteboard pipeline, parts of which remain stubbed. The name of an unfinished neighboring subsystem is not evidence that the implemented native differentiation path is imaginary. The reverse is also true: one implemented lane does not finish the other. [L04]

### WASM is a real target with an explicit precision constraint

The WASM fidelity tests execute emitted binaries through Node and compare operations such as batched matmul and reductions against references. They include rejecting a binary that runs but computes the wrong numerical answer. The precision pipeline has a split-product alternative to an FMA-dependent implementation, with tests comparing the resulting limbs and showing how that alternative removes a missing-FMA capability barrier for WASM. These are meaningful target-specific mechanisms, not a target name in a backend list. [R16] [R17]

Historical WASM handoffs record actual browser execution for their examples. They do not demonstrate the entire custom-precision, differentiated-training and constrained-deployment combination. A useful future receipt needs to specify what constrained means: memory, imports, runtime environment, capabilities and forbidden host fallbacks. Without that, running one emitted binary cannot certify the intended deployment envelope. [L05]

### Named equations become managed physical computation

The reaction-wheel path gives a strong cross-repository example of authoring leverage. orbital_actuation.py derives axial rotor rate from a named Newton equation, and derives current, back-EMF and copper loss from named Faraday equations. It solves and substitutes symbolic forms and checks unexpected structures or unbound symbols instead of merely copying a final formula into an unrelated implementation. [R07]

The integration test source compiles equations into native pieces, assembles RK4 stages and commit logic through declared bindings, and places those pieces in the actual timestep-controller machinery. Assertions cover angular momentum, rotation energy, orthogonality, speed limits, electrical and copper-loss accounting, and the identity of the underlying owned spans. An oversized timestep is intentionally rejected, and wheel momentum must be exactly restored before retry. These assertions test composition, ownership and rollback, not just a scalar numerical answer. [R18]

October receipts establish bounded execution beyond the source definitions. The 27-piece craft bank built under its memory ceiling, with a sampled peak of about 3.042 GB. A five-phase command-change test simulated ten seconds and passed unchanged physical assertions for command changes, momentum, energy and mass accounting. It did so slowly: phase execution took 186.260 seconds, with 27,556 attempts and 23,969 error-limit failures. The larger test record includes roughly 1,478.85 seconds for the complete command-change gate under the 6 GB ceiling. Those timings cover different scopes and should not be merged into a single runtime number. [R04] [L06]

The result is not a finished flight simulation. Long-duration angular momentum, desaturation, near-empty cutoff and broader flight acceptance remained open. A later supply test returned 0.9999999999999999 where exactly one was required. Production collocation and frame-partition acceptance were also unresolved. The useful conclusion is that native physical composition and rollback have concrete evidence, while controller behavior and numerical acceptance can still dominate finished-program cost. [R19] [L06]

### Provenance can be preserved without retaining the whole compiler at runtime

One of October's clearest practical improvements separated the audit archive from the runtime execution payload. Nineteen historical production archives occupied about 714 MB on disk but retained an estimated 6.0 GB of reachable Python objects, roughly 5.9 GB of that in identity books. Retaining execution payloads instead reduced the corresponding serialized runtime pieces to 7.9 MB. A sequential-load measurement peaked at about 767 MB RSS and ended near 436 MB, while the full archives remained intact. [R20]

That matters beyond an impressive ratio. It shows a way to keep the provenance you value without making every deployed calculation carry all compiler analysis state. The retention test source also checks that the full archive hash is unchanged, native columns reload, and repeated execution writes into the caller's actual output buffer. This is a real architecture-to-deployment connection, with a more defensible claim than simply saying that provenance must be expensive. [R21]

### Nodus has real execution routes with explicit limits

Nodus documents a ProcessGraph to GraphIR to registered ToolIR route, with 28 elementwise F32/F64 operations and append-only canonical identity. Its generated canonical vocabulary contains more operation names than that, but an operation catalogue is not a count of implemented tools. The distinction between canonical IDs and Turing dispatch ordinals is part of the interoperability contract. [R08]

The headless test drives actual canvas-step submission to ThreadManager, declared ports, quiescence, transport buffers, topology save/load and overflow behavior. The graph-execution harness runs registered operators against expected outputs and refuses unsupported dimensional reductions rather than quietly applying a different full reduction. These are real runtime assertions. Current headless tests include 16-byte ports carrying float[4], so Nodus transport should not be described as universally limited to one four-byte scalar. [R22] [R23]

The separate October Python graph worktree demonstrates scalar examples and configurable fixed-size byte payloads, including 16-byte four-float transport. This does not establish general tensor shape and ownership; its native bridge explicitly refuses mixed scalar and owned AbstractTensor-pointer execution. Topology persistence does not prove complete restoration of live runtime state, and the InMemoryCalculator route runs inline rather than providing the future free-running KPN scheduler. Empty TranslationMatrix and placeholder GLSL-emitter files are gaps in those named routes, not evidence that every existing integration is a stub. [R24] [L07]

### Optical and language graph work provide additional evidence and separate risks

Pluck's thick-lens acceptance record contains actual historical native renders and narrow CPU/GPU parity measurements. Reported diffuse relative L2 is about 3.87e-8 and glass about 1.56e-7; a 64 by 64 VCM render had thousands of lit pixels and recognizable structure. The same record calls the image noisy, excludes GPU-only VCM from its parity claim and does not claim calibrated sensor joules. A reduced Helmholtz/vector-angular-spectrum direction is not full three-dimensional Maxwell simulation. [R09] [R25]

Its transaction-coverage record is especially informative. Some wave and queue participants have rollback coverage, while sensor accumulators, transport state and generation-publication machinery remain uncovered. A working image therefore does not certify that the entire scientific pipeline can reject a timestep and restore every mutation. The integration history also documents removal of a misleading arrangement in which private FIFOs and manual ticking made pieces appear connected without actual Nodus-owned shared edges doing the work. That is an observed integration failure mode, not a hypothetical criticism of the project. [R26] [R27]

Fresh uncommitted electrical-network source in spectral-analyzer includes complex nodal solving, multiport and RLC elements, batteries, thermal and safety behavior, and managed-dt snapshots. Its tests assert solver routing, conservation and rollback. These are substantial current source observations, but no execution was performed and they are not a released or accepted electrical system. [L08]

SpeakToMe's FluxGraph preserves causal topology, supports bidirectional growth and re-rooting, and carries bounded expansion and flow-related structures. Its geometry plan carefully separates exact topology from semantic coordinates and distinguishes branching or stratified structure from a smooth manifold. It proposes equal-compute comparisons of top-k, uniform-frontier, flux-weighted and geometry-aware sampling. That is a concrete research question aligned with your interest in understanding information systems. It is not yet evidence that geometry improves truthfulness, coverage or model quality. Compiling a controller also does not automatically accelerate model kernels already running in an established tensor backend. [R10] [R28]

## 4 What October actually changed

### Early October established useful execution and exposed hidden defects

The month did not begin from a stable compiler. October 1 work included source-to-emission ownership and return-site handling alongside a newly recorded record-return miscompile. October 3 then produced the Jacobian parity result and the archive/runtime retention improvement described above. These were genuine results with explicit remaining failures, not evidence of a completed compiler followed by an unexplained stall. [R03] [R20]

Between October 4 and 8, native solve progressed from eager-only success and emission failure into C and LLVM execution, including nested loop and timestep shapes. Later repairs corrected float32 storage and per-specialized-copy shape proofs. N2 achieved bit-exact parity across 69 columns, including state, timestep and telemetry, with twelve state columns changing. Yet a whole-module dominance probe subsequently found ten violations: one undefined-on-path LNot and nine stale Phi initial identities. After repair, the audit reported 458 functions, zero dominance violations and zero clang warnings while preserving N2 parity. [R05]

The important lesson is not that the earlier numerical success was worthless. It is that a bounded numerical path and structural validity answer different questions. Both had to be tested. That pattern recurs throughout the month.

### Compile scaling improved without settling the broad workload

Signature-only linking reduced retained graphs. In a reported N2 comparison, the graph fell from 21,704 to 13,787 nodes; closure decreased from roughly 9.1 to 4 seconds; arrival at the same checkpoint improved from about 984 to 280 seconds. N8 specialization moved from not finishing after 45 minutes to about 329 seconds. N8 still had an approximately 37 GB private-memory or swap wall, and the small-copy optimization had not been accepted on the whole case. The frontier audit also found more failing baseline files than a repeatedly cited seven-failure baseline suggested. [R05]

October 11 delivered an important but specific milestone. The unchanged public source compiler freshly rebuilt four stale law pieces without saved-IR input, diagnostic transplantation or post-lowering repair. The source-to-native run took 458.310 seconds including imports, profiling and leaf rebuilds, constructed 996 functions and reached an observed OS peak around 6.67 GiB. Its structural audit found zero findings. Numerical dt execution and parity were not run; eight or all 35 rules were not tested on that source state; 60,383 unsourced facts and 468 unposted identities remained open. Calling this a fresh N4 source-to-native build is justified. Calling it verified end-to-end N4 physics is not. [R29] [L09]

### The first bootstrap target became much more demanding

October 12 through 14 added substantial compilation infrastructure: staged products, checkpointing, resumption, memory regulation and more complete closure accounting. Reported partial-stage improvements included source closure from 133.78 to 105.91 seconds, tensor specialization from 581.12 to 185.07 seconds and arrival at a catalogue-control checkpoint from 1,058.66 to 591.62 seconds. These are stage comparisons, not whole-bootstrap speedups. Runs still crossed 10 GiB or were stopped after about half an hour. [R30]

The provenance-first contract established on October 14 was a meaningful expansion of the first success criterion. The compiler must compile the complete provenance/book implementation with native storage and semantic behavior before outer compiler work proceeds. That includes posting, indexing, queries, sorting, formatting and compressed logs. A native accessor around a Python book is not sufficient. At one checkpoint the core already contained about 58,522 nodes, 63,437 edges and 493 functions. A 1,200-second run timed out during adoption without a native module. [R31]

This target is coherent with your requirement that the compiler preserve its own evidence. It also exercises heterogeneous objects, mappings, generators, tuples and dynamic-looking Python behavior that a numerical kernel does not settle. The October 14 completion contract requires more than a raw bound library: native storage, the required CPython initializer and whole-module semantic acceptance. A four-stage growth example resumed and produced native results of minus three, zero and 21, but that bounded proof is not the complete provenance bootstrap. [R31] [R32]

### Midmonth census showed movement but no near complete bootstrap

| Recorded census | Planned or recorded units | Compiled unverified | Partial | Failed | Native verified |
| --- | --- | --- | --- | --- | --- |
| October 16 | 322 | 35 | 49 | 238 | 0 |
| October 17 saved census | 323 | 37 | 49 | 237 | 0 |
| October 18 | 327 | 41 | 66 | 220 | 0 |

These are broader censuses, not the later fail-fast admitted prefixes. The populations and sources changed, so the table is not a controlled percentage-complete series. It nevertheless shows some units crossing earlier blockers and failures moving into partial lowering. It also shows the scale still unresolved. The October 18 accounting contained 104 independent non-clean roots, including 38 failed and 66 partial roots; dependency cascades made the total failed population larger. [R33]

An actual orchestration defect also distorted apparent progress: default jobs=12 silently limited total stage admission to twelve, not merely concurrent work. Separating admission from concurrency repaired that behavior. A small fail-fast numerator could therefore reflect scheduling and the first encountered failure as well as compiler capability. [R33]

### Late October included general repairs and identifiable rework

Some repeated work was intrinsic discovery: fixing one semantic boundary exposed the next. Some was avoidable churn, and the receipts identify it precisely. One convergence loop had two writers fighting: one restored a resident over its conversion Cast, and another recreated the Cast. This was a real circular repair, not a metaphor for every long compiler investigation. [R01]

Another shared failure was an unconsumed contract. The reducer wrote function_output_shortfall for incompatible return arities, but no consumer read it. Three apparent blockers descended from that omission. Publishing and reading the return-arm contract then exposed sequence-return merging, optional aggregate ABI and cross-unit binding gaps. [R01]

Unit 042 supplies a particularly important warning. It compiled while sorted(generator) silently lost its sort. Generator materialization and sorting shared one effect identity, so mutation deduplication discarded the sort. The lowering repair was real, but the receipt withheld native runtime confirmation of the replacement operation. Compiled-unverified was the correct label, not excessive caution. [R01]

Three downstream repairs were later reverted and rewritten at their generators because they repaired generated content rather than the authoritative producer. A tuple-valued mapping declaration was also landed without its corresponding consumers, reducing admission from 31 to five in that sequence before being reverted. Those are concrete cases of work repeated or undone. The required mapping feature spans schema, physical ABI, lookup/storage and authoritative leaf identity; landing one declaration is not completing that feature. [R02]

## 5 Why progress has not composed into the awaited outcome

### Different gates have been treated as if they were one measure

The work moves through source acceptance, graph construction, repository SSA, structural validity, provenance closure, emission, linking, binding, numerical execution and performance. Each can improve while the next remains blocked. A build receipt is valuable, but it does not answer the same question as a native numerical comparison. A lowered bootstrap unit is valuable, but it does not prove a native replacement for its Python behavior.

The local implementation makes that distinction explicitly. project_compilation_product.py classifies structurally complete units as compiled-unverified. Its execution verification applies only to a narrow eligible resident-table/integral case in the inspected code. Most structurally complete units are not executed by that route. Zero verified bootstrap units therefore does not mean there are no native examples anywhere; it means this bootstrap gate has not produced the native semantic receipts it is designed to require. [L10]

### Authoritative facts and their consumers do not yet form a closed contract

Many failures are representational rather than missing arithmetic. They occur at record versus primitive ABI, aggregate projection, return-arm layout, descriptor rereads, aliases, wrapper versus payload identity, predicate dtype and compiler-owned mapping or iterator traversal. A producer revises a value, but every downstream consumer does not necessarily revise the facts it holds about that value.

Provenance and concordance address a real problem here. The new structural-concordance source treats source, SSA and storage structures as derived views rather than competing identities, and refuses incomplete boundary correspondence. The detector finding stale float64 consumers after an aggregate revision is concrete evidence for the need. It does not establish that the enforcement system is itself fully integrated. The latest record still calls for recipes through projection and locking, wrapper/payload separation, witness retention and a genuine settlement gate. [L01] [L11]

The engineering implication is stronger than “add more tests.” A representation feature needs an end-to-end contract describing who produces the fact, how it is revised, which consumers read it and what artifact behavior proves it survived. Otherwise a locally plausible repair can leave the same inconsistency alive at another boundary.

### Broad compiler workloads enlarge the semantic surface

The provenance bootstrap deliberately includes the machinery used to represent the compiler's own analysis. That is much more demanding than compiling one loop over tensors. Tire supplies a different broad workload, with authored records and physical return adaptation exposing failures the bootstrap does not necessarily reach first. Using both can reveal shared defects, but running both at full scale before a focused invariant is demonstrated makes each attempt expensive.

This does not prove that self-hosting is the wrong goal. Lowering the execution cost of the compiler could materially improve its future feedback cycle. It does mean that compiler-private Python semantics and the intended scientific-program semantics should not be treated as identical evidence of value. A repair can be important to the former while leaving the latter's decisive uncertainty untouched.

### The cost of discovering each next failure is high

Minutes-long compilation, multi-GiB retention, source fingerprints and invalidated saved plans reduce the number of complete experiments that can be performed. Staging and resume are genuine engineering improvements because they preserve expensive work. They have not yet made the broad runs cheap enough or reliable enough to deliver the awaited outcome. [R30] [L12]

The local source is also concentrated in very large modules. Approximate observed sizes included 57,252 lines in fortran_c_shell.py, 29,744 in glsl_deployment_strategy.py and 26,521 in precompile_to_ssa.py. Those counts do not prove the architecture is wrong. They support a narrower concern: it is expensive to establish that a semantic revision reaches every relevant consumer when ownership is distributed through so much code and several representations. Extraction alone is not an acceptance result; a stable invariant and its consumers are the useful unit to track. [L12]

### Documentation drift consumes evidence quality

Some documents still describe rational composition as unimplemented despite current classes and tests. The compiled linear-algebra test file retains old opening prose about a strict-xfail defect although its later tests say the sequential-store issue was fixed and are not marked xfail. An earlier differentiation feasibility record predates a successful Jacobian execution. Historical Nodus integration complaints also need reconciliation with current headless tests. [R12] [R34] [L03] [L04]

This makes an apparent answer easy to get wrong in either direction. Optimistic handoffs can overstate a complete route; stale blocker prose can erase a real repair. A report or continuation should give the date, exact source state, scope and acceptance result rather than carrying forward a broad “works” or “blocked” label.

## 6 What the performance evidence does and does not buy

### There are real large gains over dispatch heavy execution

The August 19 function-to-deployment handoff records a 115-node eigendecomposition taking 521.5 seconds through AbstractTensor Jacobi, 0.257 seconds for a plain-array version through Turing, and 0.009 seconds with NumPy/LAPACK. The first comparison is approximately 2,030 times faster. It supports the idea that a readable algorithm can shed enormous interpreter and dispatch overhead. The same evidence shows the compiled algorithm about 29 times slower than the optimized NumPy/LAPACK baseline, so it cannot support a claim of beating established native linear algebra. [R35]

That record also contains an authoring result: changing an absolute rotation cutoff to a relative criterion repaired a graded-matrix accuracy failure by changing readable Python and recompiling. The value is the ability to alter numerical meaning without separately rewriting a backend kernel. It is not solely the hot-loop ratio. Current compiled-linalg test source checks eigenvalues, orthogonality and residuals, but the historical timings were not remeasured. [R34] [R35]

| Evidence | Recorded comparison | Meaning and limit |
| --- | --- | --- |
| 115-node Jacobi handoff | 521.5 s eager; 0.257 s compiled; 0.009 s NumPy/LAPACK | Large dispatch reduction; not native-library superiority |
| Tiny Jacobi profile | Roughly 88 to 432 ms eager versus 0.009 to 0.024 ms warm native; about 2.328 s compile | Bounded historical cases; tiny timings and compile cost need context |
| 4096-node dependency levels | 138.34 ms eager kernel versus 0.8374 ms native; 179.90 versus 12.53 ms whole wall | Kernel ratio differs from application ratio; two-thread 13.52 ms did not improve wall time |
| 64 cubed GEMM bakeoff | LLVM 0.390 ms versus 107.374 ms through that Fortran route | One shape and two emitted routes; not a CPython or optimized BLAS comparison |

The latter three rows come from recorded compiler-analysis or backend experiments, not complete current applications. The dependency-level and combined native-analysis examples show opportunities to reduce compiler analysis cost, but do not demonstrate a completed faster compiler. [R36] [L13]

### Runtime speed compilation cost and authoring leverage are different returns

Runtime speed asks how long an accepted artifact takes to perform equivalent work. Compilation cost asks what must be paid before that execution: cold lowering, backend build, binding, memory and cached restart. Authoring leverage asks whether a meaningful mathematical change can be made in one source definition while derivation and deployment continue to work without handwritten target repairs.

The evidence is encouraging on individual instances of all three, but it is uneven. The Jacobi record demonstrates a mathematical correction and large dispatch reduction. The October 13 timings demonstrate cheaper compiler stages. The orbital Jacobian demonstrates a derived calculation checked against SymPy. None alone establishes low-investment development of arbitrary mathematical machinery, inexpensive dynamic AOT or broadly faster finished applications.

For repeated execution, compile cost must eventually be repaid by the per-run saving, or justified by another benefit such as access to a required target. A benchmark should therefore report cold compilation, cached startup, first execution, warm execution, expected repetition and memory separately. A multi-minute compilation can be sensible for a long-running workload and unacceptable for a one-shot interactive action. There is no universal acceptable number without the intended use.

The baseline must also preserve semantics. Ordinary NumPy float64 is informative for an ordinary float64 algorithm. It may not express the chosen custom-number computation, in which case equivalent eager execution and an independent high-precision oracle answer more relevant questions. Faster execution that silently loses a limb, changes a branch, drops a sort or falls back to a host implementation is not the requested acceleration.

The craft result illustrates a second limit: removing interpreter overhead does not repair an algorithm or controller that needs tens of thousands of rejected attempts. Its runtime cannot be attributed to compiler throughput without a matched baseline, but it warns that native pieces alone do not make the assembled program fast. [R04]

## 7 The newest local frontier

### The uncommitted source matters more than the published tip for current blockers

The local Turing checkout was at committed HEAD 65b4fc0657e2 with approximately 7,274 inserted and 450 deleted tracked lines across 47 modified files, plus untracked structural-concordance source, tests and reports 23 through 32. Those receipts describe successive changing working-tree states. They are not ten committed milestones, and a benchmark from an earlier commit cannot certify that tree. [L02]

Reports 23 through 29 record formal-consumer, vehicle and constructor work. Examples include a constructor becoming linkable, visual_snapshot becoming linkable, deterministic planning of 125 vehicle units including 30 class-related units, and an aggregate result reduced to an explicit first-aggregate-row refusal. One focused suite recorded 245 passes and six failures; another 259 passes and eight failures. These are targeted receipts with known limits, not proof of a native whole-vehicle build. [L14]

The dual-oracle work then moved broader failures downstream. Report 31 records 23 of 330 bootstrap units admitted: 22 compiled-unverified, one partial, zero failed and zero verified. Tire lowered all 396 planned regions and linked 369 functions with zero backend shortfalls, then failed at the adaptation between the authored Metrics record and the physical return surface. That is substantial construction depth. It is not correct semantic execution merely because no backend shortfall remained. [L15]

### The latest receipt detects a new boundary without completing either oracle

Report 32, structural_provenance_core dated 21 October, labels itself not a completion receipt. Its initial focused verification recorded 154 tests passed. Its paired full run admitted four of 337 bootstrap units: three compiled-unverified, one partial, zero failed in the admitted prefix and zero verified. It explicitly described that state as not an advance over its earlier 21-of-331 comparison. [L01]

The structural detector found two consumers retaining float64 facts after an underlying resident had been revised to ssa.aggregate. Tire again reached 396 lowered regions and 369 linked functions with zero shortfalls, but remained at failure-log or finalization after physical-call adaptation. The outstanding work includes propagating recipes through projection and locking, separating wrapper and payload, retaining witnesses and making structural settlement an actual gate. [L01] [L11]

The drop from 23/330 to 4/337 must not be converted into a percentage of lost functionality. Source populations, ordering, fail-fast stopping points and structural rules changed. It is an earlier stop under a new gate, and the receipt itself says it is not an advance. The stale-consumer finding may represent better detection of a pre-existing inconsistency rather than deletion of previously correct functionality. A genuine wrong-result regression requires a comparable previously accepted program and changed result; these admission counts alone do not supply that evidence.

The defensible present conclusion is precise: the newest recorded broad bootstrap and Tire outcomes remain incomplete, despite deeper construction and more exact detection. The specific current repair seam is authoritative representation transport into physical call and return boundaries, not an absence of any native code-generation capability.

## 8 A continuation strategy that can answer the value question

### Choose the uncertainty the next work is meant to reduce

There are two legitimate near-term priorities. One is to complete a native provenance-first slice because Python retention and compiler execution cost are limiting iteration. The other is to establish a current scientific-program receipt because the broader payoff has not been demonstrated recently enough or completely enough. The evidence cannot choose between them without your assessment of which cost is most binding. It can prevent treating one as automatic proof of the other.

If provenance-first remains the priority, its success contract should stay faithful to the existing requirement: native storage and behavior, the required binding and initializer, and semantic comparison with the canonical Python implementation. An accessor shim or a linked-but-unexecuted unit should not be relabeled success. Track completed named units, independent failure roots and semantic verification against a frozen plan where possible, rather than using the admitted prefix as a progress percentage. [R31] [L10]

If proving scientific-program value is the priority, start from an already authored accepted case. The ordinary-precision Jacobian, a compact custom-precision kernel, a graph-derived training computation or a bounded reaction-wheel case each provides a different existing starting point. Selecting one is preferable to introducing another model, representation system or backend at the same time. This report recommends evaluation choices; it does not propose that new implementation has already been undertaken.

### Build the evidence in bounded stages

First establish a fresh baseline for one existing computation through the sanctioned source entry, with no edited generated IR or hidden substitution. Record source identity, numerical output, current provenance/concordance outcome, artifact identity, compile cost and runtime. This answers whether the current source still realizes one known useful result.

Next add one defining requirement that the baseline does not yet establish. For example, make custom precision survive the selected forward calculation; then establish the corresponding derivative against an independent reference; then execute the same accepted semantics on the intended constrained target. An already integrated path can combine those stages, but the acceptance target should not silently grow after each repair. A small chosen conjunction is evidence. A new requirement added every time the previous one nearly works is not a stable test.

Finally demonstrate authoring leverage directly. Change one meaningful mathematical choice in the source definition, such as a numerical criterion, precision width or model equation, and show what must change elsewhere. The desired result is that the sanctioned derivation and lowering pipeline handles it, not a separately hand-maintained backend implementation. Record any specialist intervention honestly. This would test the “low investment” part of the goal rather than inferring it from the existence of an abstraction.

### Keep a complete receipt for the chosen claim

| Question | Evidence required |
| --- | --- |
| What program was evaluated | Exact repository revisions, dirty-diff identity where relevant, source entry and capability contract |
| Did semantics survive | Independent value and derivative checks, meaningful cancellation or conditioning cases, and required ownership or rollback assertions |
| What actually executed | Target artifact, runtime and binding identity; no undisclosed host fallback or generated-content repair |
| Was compilation affordable | Cold source-to-artifact time, peak memory, cached startup and any stage reuse |
| Was execution useful | First and warm execution, equivalent baseline, workload size and expected reuse |
| Was authoring easier | A meaningful source-level change, the work required to propagate it and any manual interventions |
| Can the result be repeated | Saved receipt tied to the actual source and artifact, with exclusions and unresolved failures stated |

The receipt should not require an arbitrary speed threshold. A modest speedup can be valuable if it enables unique semantics, constrained deployment or much cheaper authoring. A spectacular hot-loop ratio can be unhelpful if compilation, memory or integration defeats the actual use. The relevant test is the named use case and its total cost.

### Use the current failures to test a shared invariant

The latest bootstrap and Tire failures point toward a focused invariant: when an authoritative value changes structure, dtype or identity, every projected or physical call/return consumer must either obtain the revised correspondence or refuse before claiming completeness. A short focused case that demonstrates this behavior is more informative than immediately repeating an expensive census with only a downstream patch.

Once that invariant is demonstrated, the broad oracles can test whether the repair generalizes. A downstream failure is not automatically wasted effort: it may expose a new unsupported case. The record should distinguish that from recurrence of the same stale-fact mechanism, an invariant violation or an actual wrong-result regression. This preserves the value of rigorous refusal without letting an indefinitely moving frontier count as convergence.

### What would support continuing or changing direction

Continue investing in the core when fresh receipts show that the same representations preserve the intended mathematics, that general repairs close several previously independent failures, and that compile or execution cost is moving into a useful range for the selected application. A stable source-level change that reaches native or WASM output without a new backend implementation is particularly relevant evidence of authoring leverage.

Narrow the supported envelope if each new numerical class, Python construct or backend creates another unbounded set of failures before a useful integrated program remains reproducible. That could mean a more explicit numerical subset, a smaller target set or a bounded supported composition, while keeping the same mathematical vision and sanctioned architecture. Narrowing the promise is different from abandoning the work.

Reconsider the performance route if equivalent custom-number programs cannot obtain useful total-cost gains, if compilation cannot be amortized for the intended usage, or if constrained deployment requires a host fallback for the defining semantics. Reconsider the bootstrap priority if its remaining work is mainly broad compiler-private Python coverage while it does not reduce the iteration bottleneck or produce native verified replacements. These are reasons to reassess resource allocation, not an evidence-free instruction to stop after a fixed number of runs.

Failure to beat LAPACK at ordinary float64 algebra does not by itself invalidate the project. Failure to preserve and usefully execute the mathematical semantics that make the project distinctive would be much more consequential.

## 9 Overall assessment

The case for continuing rests on demonstrated mechanisms and outcomes: explicit custom numerical representations, graph-derived native differentiation, independently checked Jacobians, native physical pieces with conservation and rollback, large reductions of dispatch overhead, and a successful separation of audit provenance from runtime retention. Those are concrete reasons to believe the central idea can produce useful machinery. They are not a guarantee of novelty, generality or eventual success.

The month feels stalled because the outcome currently being pursued has not arrived, and because several different kinds of progress have been reported in its place. Some of that work strengthened correctness by exposing silent errors. Some reduced substantial resource costs. Some repeated earlier work because producer/consumer contracts were incomplete or fixes were applied at the wrong layer. The evidence supports all of those statements, not a single reassuring or dismissive explanation.

The next convincing answer will be a current, reproducible result whose claim stays fixed: either a genuinely native verified provenance slice that reduces the compiler's burden, or a useful mathematical program that preserves its defining semantics and demonstrates its actual development and deployment cost. The work already contains credible starting points for both. Its strongest future depends less on adding another layer than on making one chosen chain reliable enough that you can change the mathematics and trust what executes.

## Appendix A Coverage and confidence

| Area | Inspection depth | Evidence available | Important boundary |
| --- | --- | --- | --- |
| Turing October convergence | Deep source, history and receipt inspection | Public receipts through October 20; local uncommitted reports 23 through 32; classification and entry-point code | No rerun or current whole-program acceptance |
| Custom numbers and differentiation | Deep targeted source and test inspection | Representations, precision/rational tests, native autograd test bodies, dated parity receipts | Composite custom-precision training not demonstrated as a whole |
| Physical-law and craft path | Deep targeted derivation, integration tests and receipts | Named-law derivation, native pieces, conservation/rollback and retained-memory evidence | Long-duration and full production acceptance remain separate |
| Nodus | Targeted source, headless tests and interop documents | Registered graph routes, transport, topology and isolated Python graph worktree | No universal shape/ownership or complete free-running scheduler claim |
| Pluck and electrical work | Targeted render, transaction and integration evidence; local electrical source | Historical output and narrow parity; explicit coverage gaps; authored new tests | No fresh run of current electrical or complete optical transaction stack |
| SpeakToMe | Graph implementation, controller and experimental design | Causal structure and a bounded comparative research plan | Geometric superiority remains unvalidated |
| Geometry and other siblings | Moderate orientation or inventory only | Source state and stated scope | Not an exhaustive code audit or evidence of mature capability |

Confidence is highest in the distinction between observed source, dated receipts and current uncommitted state. Historical timings and numerical results are reported as recorded, not independently replicated. Code review cannot establish current suite health or the absence of other failures. Commit counts, line counts, catalogue sizes and test counts describe scope or activity, not correctness.

The research deliberately avoided secrets, unrelated private conversations, broad generated-artifact scans and changing repository state. It did not audit every file in every repository. The coverage table identifies the center of the investigation rather than implying comprehensive certification of the entire account.

## Appendix B Published evidence

The following links point to the inspected public snapshots. Citation IDs are used throughout the text. Where a link is a commit receipt, the relevant report and source changes are in that commit. A later local working tree may differ.

- R01 October 19 provenance bootstrap and sweep continuations. [Bootstrap continuation](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/compiler_convergence/20_provenance_bootstrap_continuation_2026_10_19.md) and [sweep continuation](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/compiler_convergence/21_provenance_sweep_continuation_2026_10_19.md). Restore/Cast cycle, ignored return contract, sorted-generator effect identity and withheld native verification.
- R02 [October 20 consumer continuation](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/compiler_convergence/22_provenance_consumers_continuation_2026_10_20.md). Generator-side rework, tuple-mapping rollback and latest committed bootstrap status.
- R03 [October 3 Jacobian commit and receipt](https://github.com/sangderenard/Turing/commit/095a3c0c). See also local source map L04 for the exact continuation location and subsequent related repairs.
- R04 [Craft bank and physical command-change receipt](https://github.com/sangderenard/Turing/commit/ca0f1b20). Build memory and runtime/physical acceptance with exclusions.
- R05 [October 7 frontier audit with October 8 updates](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/FRONTIER_AUDIT_2026-10-07.md). N2 parity, dominance repairs, graph and specialization scaling, N8 resource limits.
- R06 [Extended precision implementation](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/src/common/tensors/extended_precision.py). Representation contract, error-free transforms, renormalization, multiplication, division and composed classes.
- R07 [Orbital law derivation lines 1010 to 1057](https://github.com/sangderenard/nogodsnomasters/blob/49d2de12ecb0efe2d133e9c5eeb8335aa3a6f768/engine_toy/orbital_actuation.py#L1010-L1057). Symbolic use of named equations.
- R08 [Nodus ProcessGraph interoperability](https://github.com/sangderenard/nodus/blob/45009309781daa074a5ba69e253427a6702a1326/docs/process_graph_interop.md). Registered elementwise route, IDs and limits.
- R09 [Thick lens render acceptance](https://github.com/sangderenard/spectral-analyzer/blob/ffe5150eedbd35326b5f92f2be967730f00f9b22/THICK_LENS_RENDER_ACCEPTANCE.md). Native render and bounded CPU/GPU parity.
- R10 [FluxGraph source](https://github.com/sangderenard/speaktome/blob/95df7ffa81bd8ed8a98ed7f3b46a310e61256152/speaktome/core/flux_graph.py) and [lookahead controller](https://github.com/sangderenard/speaktome/blob/95df7ffa81bd8ed8a98ed7f3b46a310e61256152/speaktome/core/lookahead_controller.py). Inspected implementation and scope, not a comparative quality result.
- R11 [Precision surface tests](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/tests/test_precision_surface.py). Width-dependent accuracy and constant tests.
- R12 [Complex precision tests](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/tests/test_complex_precision.py). Information retention and authored source-to-native path.
- R13 [Rational precision tests](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/tests/test_rational_precision.py). Structural quotients, promotion and composition.
- R14 [ProcessGraph native autograd tests](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/tests/test_process_graph_autograd.py). Linear, convolution, XOR and training-loop assertions.
- R15 [Parked October 3 reverse precision experiment](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/concordance_census/precision_experiment_2026-10-03/README.md). Explicitly unverified and distinct from the successful Jacobian lane.
- R16 [Precision pipeline tests](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/tests/test_precision_pipeline.py). FMA and split-product capability paths and precision semantics.
- R17 [WASM fidelity tests](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/tests/test_wasm_fidelity.py). Actual emitted-binary execution test definitions and numerical rejection.
- R18 [Orbital native integration tests](https://github.com/sangderenard/nogodsnomasters/blob/49d2de12ecb0efe2d133e9c5eeb8335aa3a6f768/engine_toy/tests/test_orbital_library_integration.py). Native rotor/body/attitude composition, conservation and rollback assertions.
- R19 [Remaining supply and numerical gate](https://github.com/sangderenard/Turing/commit/41b26759). Exact-one mismatch and open acceptance boundaries.
- R20 [Initial retention blocker](https://github.com/sangderenard/Turing/commit/efbd1636) and [archive/runtime split](https://github.com/sangderenard/Turing/commit/d2240a96). Disk, reachable-object and runtime-load measurements.
- R21 [Equation piece retention test](https://github.com/sangderenard/nogodsnomasters/blob/49d2de12ecb0efe2d133e9c5eeb8335aa3a6f768/engine_toy/tests/test_equation_piece_retention.py). Audit preservation, runtime reload and output-buffer ownership.
- R22 [Nodus headless runtime test](https://github.com/sangderenard/nodus/blob/45009309781daa074a5ba69e253427a6702a1326/tests/nodus_headless_runtime_test.cpp). Real submission, ports, transport and topology; lines 54 to 55, 92 to 93 and 106 to 108 include 16-byte float-array transport.
- R23 [Registered tensor graph execution test](https://github.com/sangderenard/nodus/blob/45009309781daa074a5ba69e253427a6702a1326/tests/tensor_graph_execute_test.cpp). Execution and explicit unsupported-reduction refusal.
- R24 [Tensor calculator bridge](https://github.com/sangderenard/nodus/blob/45009309781daa074a5ba69e253427a6702a1326/docs/tensor_calculator_bridge.md). Inline calculator scope and future scheduling boundary.
- R25 [Wave model replacement decision](https://github.com/sangderenard/spectral-analyzer/blob/ffe5150eedbd35326b5f92f2be967730f00f9b22/WAVE_T4_REPLACEMENT_DECISION.md). Reduced wave model and release gates.
- R26 [Native optical transaction coverage](https://github.com/sangderenard/spectral-analyzer/blob/ffe5150eedbd35326b5f92f2be967730f00f9b22/NATIVE_OPTICAL_TRANSACTION_COVERAGE.md). Covered and uncovered mutation/rollback participants.
- R27 [Nodus integration brief](https://github.com/sangderenard/spectral-analyzer/blob/ffe5150eedbd35326b5f92f2be967730f00f9b22/NODUS_TOOL_BRIEF.md). Misleading private-FIFO/manual-tick integration and its correction; older missing-infrastructure statements require current-source reconciliation.
- R28 [Branching search geometry plan](https://github.com/sangderenard/nogodsnomasters/blob/49d2de12ecb0efe2d133e9c5eeb8335aa3a6f768/SPEAKTOME_BRANCHING_SEARCH_GEOMETRY.md). Proposed equal-compute experiment and explicit conceptual limits.
- R29 [October 11 fresh N4 build receipt](https://github.com/sangderenard/Turing/commit/8eb40eae0d201486cd72a9af7245738841be1981). Source-to-native construction and audit, excluding numerical dt parity.
- R30 [Bootstrap performance report](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/compiler_convergence/11_bootstrap_performance.md). Partial-stage timing improvements and continuing resource limits.
- R31 [Provenance first bootstrap contract](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/CONTINUATION_PROVENANCE_FIRST_BOOTSTRAP_2026-10-14.md). Native core requirements, scope and incomplete acceptance.
- R32 [Native compilation growth](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/compiler_convergence/13_native_compilation_growth.md). Bounded staged native execution and remaining whole-provenance binding requirement.
- R33 [October 16 census and repair rounds](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/compiler_convergence/19_census_2026_10_16.md). October 16 to 18 populations, roots, dependency failures and admission/concurrency repair.
- R34 [Compiled linear algebra tests](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/tests/test_compiled_linalg.py). Sequential stores, eigenvalues, orthogonality and residual assertions; old opening defect prose is stale.
- R35 [Function to deployment handoff](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/FUNCTION_TO_DEPLOYMENT_HANDOFF.md). August 19 timing and mathematical correction; historical open-defect sections require reconciliation.
- R36 [Backend bakeoff](https://github.com/sangderenard/Turing/blob/65b4fc0657e242663aec5812c7604ad433e06261/docs/compiler_convergence/04_backend_bakeoff.md). One-shape emitted-backend timing, not a general benchmark.

## Appendix C Local evidence map

These references were inspected in the local repositories on 21 October 2026. Paths below are relative to C:\dev\Powershell unless stated otherwise. Line numbers describe that observed working tree and can shift. They are locators, not public links. In particular, the untracked convergence reports and structural-concordance source are not represented by the published Turing HEAD.

- L01 turing/docs/compiler_convergence/32_structural_provenance_core_2026_10_21.md. Latest untracked receipt; focused 154-pass result, 4/337 admitted paired run, stale consumers, incomplete Tire finalization and outstanding settlement work.
- L02 Local Git status, branch, HEAD, worktree and tracked-tree inspection for the named repositories. Root git ls-tree confirms tracked engine_toy and site; Turing diff summary records the observed modified tree. No fetch, checkout or repository change was performed.
- L03 turing/src/common/tensors/extended_precision.py, around lines 515, 1563, 1597, 1690, 1707 and 1871, for Precision, Rational and composed classes and promotion; turing/docs/RATIONAL_FEATURE_TIMING.md; turing/tests/test_rational_compiler.py; turing/tests/test_complex_precision.py around line 105.
- L04 turing/docs/concordance_census/CONTINUATION_SESSION_2026-10-03.md lines 21 to 25 records the executed Jacobian result. Associated commits include 095a3c0c, 0519095d, ccba325c, 51b4cebe and ce587b82. DIFFERENTIATION_FEASIBILITY_2026-10-02.md lines 140 to 146 predates that success. speaktome/AGENTS/experience_reports/1791561850_DOC_Turing_Bounded_Graph_Work.md lines 131 to 135 says later focused reverse tests did not rerun the full 7 by 19 result.
- L05 turing/docs/WASM_SHELL_HANDOFF.md, WASM_CLASS_GRAPH_HANDOFF.md and WASM_CLASS_COORDINATOR.md. Historical browser and WASM examples, not the integrated custom-precision training claim.
- L06 turing/docs/concordance_census/ORBITAL_DT_METRICS_2026-10-03.md, sections located by lines 3 to 8, 273 to 288 and 291 to 296 in the inspected receipt. Bounded craft execution, rollback, physical torque assertions and 27-piece command-change gate with broader flight exclusions.
- L07 nodus/include/translation_matrix.h lines 2 to 5; nodus/include/kernel_ir_glsl_emitter.h; nodus/include/canonical_ops.h; separate .nodus-python-graph worktree at a3a3986 on codex/pythonic-graph-interface-2026-10-04. In that worktree, nodus_graph/model.py lines 50 to 76 configures token_bytes; bridge lines 20 to 23 and headless tests lines 103 to 114 demonstrate 16-byte four-float transport; native.py lines 497 to 525 implements byte push/pull, while lines 478 to 489 refuses mixed scalar and owned AbstractTensor-pointer execution. Named stub routes do not negate the registered routes cited in R08 and R22 to R24.
- L08 spectral-analyzer untracked electrical_dt_engine.py, electrical_tensor_network.py, machine_electrical_graph.py and tests/test_electrical_tensor_network.py. Source and authored assertions inspected; no execution or acceptance claim.
- L09 speaktome/AGENTS/experience_reports/1791561850_DOC_Turing_Bounded_Graph_Work.md lines 198 to 220. October 11 fresh N4 source-to-native build and audit; lines 218 to 219 explicitly exclude numerical dt execution/parity.
- L10 turing/src/compiler/project_compilation_product.py lines 7371 to 7375 and 7661 to 7714. Classification as compiled-unverified and narrow execution-verification eligibility.
- L11 turing/src/compiler/structural_concordance.py lines 1 to 7 and around 226 and 233; turing/src/compiler/identity_concordance.py around line 5700. Derived structural views, incomplete-boundary refusal and authoritative reads.
- L12 turing/src/compiler/fortran_c_shell.py around line 56242 and lines 56273 to 56277; compile_project_catalogue.py lines 323 to 324; bootstrap_compiler_concordance.py around line 181. Sanctioned public entry, holistic versus explicit-source modes, stage persistence and resume. Module sizes are from the same local inspection.
- L13 turing/docs/compiler_convergence/05_eigh_profile.md and evidence/05_eigh_profile.json; speaktome/AGENTS/experience_reports/1791561850_DOC_Turing_Bounded_Graph_Work.md around lines 1879 to 1886 and 2206 onward. Historical tiny eigensolver and native dependency-analysis timings, not whole-compiler execution.
- L14 Untracked turing/docs/compiler_convergence/23_formal_consumer_provenance_continuation_2026_10_20.md; 24_vehicle_validator_creep_2026_10_20.md; 25_vehicle_bootstrap_cross_trials_2026_10_20.md; 26_vehicle_effect_and_iterator_repairs_2026_10_20.md; 27_vehicle_class_fragment_and_constructor_formals_2026_10_20.md; 28_vehicle_mapping_and_return_join_2026_10_20.md; 29_ssa_record_abi_and_aggregate_query_frontier_2026_10_20.md. Successive diagnostic and isolated-construction receipts, not whole native execution.
- L15 Untracked turing/docs/compiler_convergence/30_bootstrap_tire_dual_oracle_2026_10_20.md and 31_bootstrap_tire_abi_predicate_frontiers_2026_10_21.md. Broad paired construction depth, bootstrap status and authored Metrics versus physical-return boundary.
