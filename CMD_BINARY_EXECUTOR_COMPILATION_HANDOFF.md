# Runnable cmd.exe binary executor compilation handoff

**Date:** 2026-08-05  
**Primary repository:** `C:\dev\Powershell\turing`  
**Publication repository:** `C:\dev\Powershell` (`nogodsnomasters`)  
**Goal status:** active and incomplete

## Goal

Compile the existing reversible Windows AMD64 machine, including its executor
coordinator and capability-gated system behavior, from the Dream Document
through Turing's ordinary compiler and immutable site publisher. The resulting
static bundle must run a user-supplied real `cmd.exe` in the browser, accept
terminal input, publish exact machine snapshots, step forward and backward,
and retain deterministic tape behavior at boundaries that cannot execute
resident in the browser.

The project-authored PE fixture remains the distributable default subject. Do
not publish Microsoft's `cmd.exe` bytes. The real-binary acceptance test loads
the user's local `cmd.exe` through the existing `subject-binary` file port.

## Definition of done

The goal is complete only when all of the following are true:

1. `examples/reversible_chip_simulator.dream` is the program source that owns
   machine setup, execution, display, and browser system routing.
2. Every executable Dream section has a concrete emitted artifact. A graph,
   metadata record, source string, or `executable=True` flag is not an artifact.
3. The machine coordinator used by the page is compiled from the Dream section
   IR. It is not a handwritten JavaScript executor and not a Python loopback
   server.
4. The existing AMD64 loader, instruction semantics, machine-block recompiler,
   authenticated journals, tape state, snapshot ABI, and system-port policies
   remain the semantic owners. No second machine implementation is introduced.
5. The static page can load a real PE32+ AMD64 `cmd.exe`, reach a receptive
   prompt, accept a command, and display the resulting terminal bytes without
   a local Python or Node process.
6. Forward, reverse, pause, speed, and both single-step controls operate on the
   same exact execution history. Reversal crosses instruction and admitted
   capability edges.
7. Unsupported resident behavior uses provenance-bound exact tape replay or
   fails closed. Replay is not represented as newly executed guest code.
8. The bundle is produced by the existing normal compiler/publisher path under
   `site_bundle.py`. Its manifest must not report
   `prebuilt-program-interior`.
9. The immutable output lands under
   `C:\dev\Powershell\site\programs\reversible-binary-machine\versions\...`,
   is discovered by the existing root gallery, and passes browser validation
   from a static origin.

## Non-negotiable architecture constraints

- Do not add a publisher. Extend the existing compiler dispatch and
  `build_program_bundle` ownership where needed.
- Do not add another HTML shell. Continue through `emit_dream_html_shell()` and
  the shared shell IO and snapshot liaisons.
- Do not substantially rewrite the shell. The program interior owns machine
  behavior and display; the shell supplies declared resources and ports.
- Do not implement another AMD64 interpreter in JavaScript or Wasm.
- Do not bypass `BinaryMachineProgram`, `MachineExecutionOrchestrator`, or the
  authenticated per-instruction journal contract.
- Do not use ambient host execution, Win32 passthrough, or browser-native
  behavior as an unrecorded substitute for a guest capability.
- Do not claim execution when the page is only walking retained snapshots.
- Do not weaken witness, digest, address, memory-effect, or continuation checks
  to increase apparent coverage.
- Preserve the input subject's bytes and all existing integer/register widths.
- Preserve unrelated dirty worktree changes.

## Verified baseline

The current root publication is:

`site/programs/reversible-binary-machine/versions/v1-800973946ec42e0b/`

It proves the following useful pieces:

- a project-authored PE32+ AMD64 fixture is included;
- three entry instructions execute from a provenance-bound Wasm block;
- the Wasm journal replaces only the replay prefix it owns;
- exact retained replay covers the outer `RET` boundary;
- five complete `TMSNAP01` frames support static forward and reverse controls;
- the existing Dream-owned GLSL display, terminal surface, register HUD,
  memory pages, controls, file parameter, and root publisher all work.

It does **not** satisfy the goal. Its manifest says
`compiler.backend = prebuilt-program-interior`; `machine_web_publication.py`
compiles one block after shell generation; the page has no resident machine
coordinator, PE loader, dynamic dispatcher, tape owner, or capability owner;
and a real `cmd.exe` still requires `reversible_machine_web_host.py`.

The native machine itself is substantially further ahead than the static page.
It has executed real `cmd.exe`, including the `echo data | hello-card alpha`
pipeline, exact system state, VFS, registry, pipes, virtual child processes,
threads, linked modules, external completions, segmented tape, and reversible
snapshots. Preserve that implementation instead of recreating a demo subset.

## Current ownership map

| Concern | Existing owner | Required change |
| --- | --- | --- |
| Dream parsing and section routing | `src/compiler/dream_document.py` | Make section compilation return emitted artifacts and ABI requirements, not graph descriptions alone. |
| Dream SSA deployment catalogue | `DreamDocument.lower_to_ssa()` | Bind executable artifacts to section and deployment identities. |
| AMD64 machine program | `src/compiler/binary_machine_program.py` | Define the compilable resident coordinator surface without changing guest semantics. |
| PE execution and exact state | `machine_execution.py`, `amd64_machine_semantics.py` | Reuse from the resident coordinator; do not fork semantics. |
| Wasm safe-prefix lowering | `machine_block_recompiler.py` | Generalize artifact selection/dispatch, preserving per-instruction witnesses. |
| Browser block packaging | `machine_block_web_bundle.py` | Package multiple runtime-selected blocks and shared ABI data, not only one entry block. |
| Capability policy | `machine_system_ports.py` | Expose a browser-resident adapter over the same typed requests/completions. |
| Shell IO/VFS/devices | `shell_io.py`, `wasm_html_shell.py` | Bind existing file, terminal, IndexedDB/OPFS, and device ports to the compiled coordinator. |
| Snapshots and controls | `machine_state_buffer.py`, Dream shell liaison | Keep `TMSNAP01`; point controls at the resident owner. |
| Static fallback | `embed_machine_snapshot_replay()` | Retain exact tape spans only for unavailable resident operations. |
| Current assembly detour | `machine_web_publication.py` | Reduce to compiler projection support or remove from the final route once normal compilation owns these artifacts. |
| Immutable publication | `site_bundle.py` | Extend existing `build_program_bundle` source dispatch for Dream artifacts. Do not add a publisher. |
| Current example orchestration | `examples/reversible_machine_web_host.py` | Leave as native diagnostic host; it must not be the static runtime or final publisher entry. |

## Required work, in dependency order

### 1. Make Dream section compilation produce artifacts

`DreamDocument.compile_sections()` currently produces
`DreamSectionCompilation.graph` plus visible shortfalls. Python sections are
explicitly non-executable, and JavaScript/GLSL sections are called executable
even though the record does not own emitted bytes.

- Introduce an artifact-bearing result within the existing Dream compilation
  abstraction. It must carry artifact kind, bytes or source, entry symbol,
  imports/exports, memory requirements, content digest, and shortfalls.
- Compile `chip-setup` and `head-step` through the existing ProcessGraph/AOT/SSA
  route. The emitted artifact must be derived from the lowered graph, not from
  copied handwritten browser code.
- Emit GLSL/WGSL artifacts through their existing lowering/backend paths and
  bind them to the corresponding deployment regions.
- Treat browser JavaScript only as an existing source-language deployment. It
  may drive presentation liaison behavior, but it may not become the machine
  executor.
- Make unresolved executable sections fail publication. A metadata-only
  Python section must no longer produce a runnable bundle.

Acceptance tests:

- `test_every_dream_section_has_a_graph_and_explicit_language_dispatch` is
  replaced or extended to assert concrete artifacts for every executable
  section.
- Tampering with one section artifact or digest fails before instantiation.
- The Dream deployment table and artifact manifest have a one-to-one identity
  mapping.

### 2. Define the compiled machine coordinator ABI

Do not attempt to compile the entire Python object graph as one opaque function.
First define the narrow state-machine ABI the existing owners already imply.

The resident coordinator must own:

- subject byte admission and PE initialization;
- exact core state and bounded guest page memory;
- dispatch-plan lookup and code-page version checks;
- Wasm block invocation and authenticated journal commit;
- fallback-tape selection at an unsupported boundary;
- pending external request/completion transitions;
- direction, pause, speed, and one-transition stepping;
- `TMSNAP01` generation;
- persistent tape checkpoint/delta flush and hydration.

Represent these operations in repository SSA as explicit control/effect calls.
Do not hide them behind a Python callback reference in the graph. The ABI must
make mutable resources and effect domains visible so the existing compiler can
reject unsupported lowering rather than silently omitting behavior.

Start with the project-authored fixture, but design the subject and guest-memory
resources as dynamic byte spans so the same artifact accepts `cmd.exe`.

Acceptance tests:

- The coordinator initializes the authored PE and reaches its exact halted
  state using only emitted artifacts.
- Every committed instruction still corresponds to one validated witness and
  one reversible state edge.
- The coordinator rejects stale code bytes, stale page versions, invalid
  continuations, malformed journals, and out-of-bounds guest memory.

### 3. Move from one prebuilt block to resident dynamic dispatch

`build_machine_web_publication()` currently recompiles only the initial block.
A real command processor needs blocks selected as RIP and executable memory
change.

- Reuse `MachineExecutionOrchestrator.recompile_block_wasm()` and the existing
  artifact cache key inputs to define a browser-consumable block catalogue.
- Support ahead-of-publication blocks where the subject graph is statically
  known and runtime block compilation/lowering where admitted executable pages
  are created or changed.
- If runtime lowering cannot yet be resident, publish a bounded authenticated
  catalogue for a named real `cmd.exe` proof and fail closed outside it. Do not
  call such a catalogue arbitrary-binary support.
- Generalize `machine_block_web_bundle.py` from the fixed
  `machine/recompiled-entry` asset to content-addressed block artifacts and a
  dispatch index.
- Preserve the interpreter/tape boundary for unsupported instructions,
  lifecycle sentinels, external calls, and effect shapes until their resident
  implementation exists.
- Add memory/device journal commit before allowing such journals to replace
  retained frames. Register-only prefix replacement is not enough for
  `cmd.exe`.

Acceptance tests:

- At least two blocks connected by real control flow execute and reverse in a
  browser without snapshot replay.
- A memory-writing block updates guest memory, snapshot page occupancy, and
  reverse history exactly.
- Self-modifying or newly executable pages invalidate stale blocks.
- A replay frame is used only when the manifest identifies the exact unsupported
  boundary and its predecessor state digest.

### 4. Compile the PE loader and runtime state construction

The static page currently receives a packed state for the authored fixture.
That is not loading a selected binary.

- Lower the existing bounded PE parser/link-plan/state-construction path needed
  by `BinaryMachineProgram.load_pe()` into the resident coordinator.
- Bind the existing `subject-binary` file port to this loader.
- Preserve PE headers/sections, relocations, imports, TLS, startup queue,
  PEB/TEB, stack, virtual memory, and external-reference identities.
- Keep dependency acquisition capability-gated. Browser bundle references and
  explicit user-provided files are allowed; ambient host DLL lookup is not.
- Publish loader limits in the runtime contract: subject bytes, module count,
  aggregate dependency bytes, page count, core count, stack, and tape budget.

Acceptance tests:

- File selection changes the actual loaded image and initial RIP, not only a
  label or preview asset.
- The authored fixture loads and executes from bytes after page startup.
- A locally selected real `cmd.exe` reaches its first guest instruction with
  the same initial-state digest as the native owner for the same configuration.
- Malformed, oversized, overlapping, unresolved, or unsupported PE inputs fail
  closed with a visible machine trap.

### 5. Bind browser system ports to exact machine completions

The current static bundle declares ports, but only the loopback controller
services live machine requests.

- Compile or instantiate a browser adapter for
  `CapabilityGatedExternalPort`; keep the existing handler identities and
  request/completion schemas.
- Route terminal input, terminal output, VFS, environment, registry, clock,
  heap/virtual memory, pipes, process records, thread records, and typed devices
  through exact state transitions.
- Reuse `shell_io.py` IndexedDB/OPFS hydration and flush barriers for persistent
  VFS/tape bytes. Do not create a parallel browser filesystem.
- Map browser-safe operations only. Host paths, host process launch, ambient
  libraries, and undeclared devices remain unavailable.
- For deterministic system-owned results, either execute the existing pure
  handler resident in the page or consume a provenance-bound tape completion.
- For unavailable environmental results, require an explicit user capability
  or stop pending. Never fabricate success to keep the prompt moving.

Acceptance tests:

- Terminal input becomes `console.input` state and reverses.
- Terminal output comes from guest execution/capability completion and
  reverses.
- VFS, pipe, registry, process, and thread state survive page reload through
  the existing persistent stores.
- Unknown function identities and unsupported argument shapes remain pending
  or trapped with their exact identity visible.

### 6. Make exact tape fallback a first-class compiled deployment

Finite replay is currently embedded as a whole array. The final runtime needs
addressable exact fallback spans without confusing replay with computation.

- Define a tape-span artifact in the Dream/SSA deployment model with subject,
  initial-state, predecessor, sequence, state, and content digests.
- Let the coordinator request a span only when its current exact state matches
  the span's declared predecessor and boundary identity.
- Commit each replayed edge through the same history/snapshot path used by
  resident execution.
- Preserve forward and reverse traversal and branch identity.
- Store larger spans as content-addressed bundle assets or existing segmented
  tape objects; do not embed multi-megabyte base64 arrays in HTML.
- Mark replay provenance in the UI/runtime status so a reviewer can distinguish
  resident Wasm, exact tape, and blocked execution.

Acceptance tests:

- Wrong subject, state, sequence, effect, or digest refuses the span.
- A resident prefix, tape boundary, and next resident block form one reversible
  timeline.
- Reload restores the same branch tip and can reverse across the deployment
  boundary.

### 7. Route the result through the normal compiler and publisher

The final route must not call `publish_prebuilt_program_bundle()` with an
already assembled machine page.

- Extend the existing `build_program_bundle()` source-kind dispatch to accept a
  Dream Document and invoke Dream section compilation, artifact assembly,
  standard shell emission, runtime manifest construction, and immutable write.
- Keep one `turing-program-bundle-v1` layout, one content-version calculation,
  one artifact inventory, and one gallery refresh path.
- Add every compiler implementation file capable of changing Dream/machine
  artifacts to `_BUNDLE_COMPILER_IMPLEMENTATION_FILES` so immutable versions
  change when emitted semantics change.
- Record compiler route, Dream section digests, coordinator ABI, machine-block
  index, tape spans, limits, system capabilities, and snapshot ABI in
  `bundle.json`.
- Convert `examples/reversible_machine_web_host.py --publish-bundle` into a
  call to the normal compilation entry or retire that option. Keep the host as
  a native runtime diagnostic.
- Ensure final output defaults to the parent root and never to `turing/site`.

Acceptance tests:

- One normal build command consumes the `.dream` source and produces the page,
  coordinator Wasm, shader assets, machine blocks/index, authored subject,
  tape spans, and manifest.
- The manifest compiler backend identifies the Dream/machine compiler route
  and contains no `prebuilt-program-interior` marker.
- Rebuilding identical source and compiler inputs reuses the immutable version;
  changing any executable artifact owner changes it.
- The existing Python-source publisher and all other bundle types remain
  unchanged.

### 8. Prove a static real-cmd vertical slice

Use a user-selected local `C:\Windows\System32\cmd.exe`; do not add it to Git
or the published bundle.

Minimum proof sequence:

1. Serve the root repository as a static origin.
2. Open the newly compiled immutable bundle with no Python/Node machine host.
3. Select `cmd.exe` through `subject-binary`.
4. Verify the page reports the selected SHA-256 and PE entry RIP.
5. Advance to a receptive command input state.
6. Send `echo hello` followed by CRLF through the existing terminal port.
7. Observe exact `hello\r\n` terminal output and a new prompt.
8. Pause; single-step backward across output and at least one capability edge;
   single-step forward and recover byte-identical state/output.
9. Reload; hydrate the exact persisted tip; reverse and forward again.
10. Verify runtime status identifies which edges were resident Wasm and which,
    if any, were exact tape.

This proof may initially use a named, digest-bound block/tape catalogue for the
tested system `cmd.exe`. The limitation must be explicit. General arbitrary
PE support is not complete until runtime block generation and all required
capability families are resident.

## Validation matrix

Run focused tests after each slice, then the broader machine selection. Keep
all tests targeted to `C:\dev\Powershell\turing` with
`PYTHONPATH=C:\dev\Powershell\turing`.

Core focused suites:

```powershell
python -m pytest tests/test_dream_document.py tests/test_site_bundle.py -q
python -m pytest tests/test_machine_block_recompiler.py tests/test_machine_wasm_runtime.py -q
python -m pytest tests/test_reversible_demo_subject.py tests/test_machine_state_buffer.py -q
python -m pytest tests/test_machine_system_ports.py tests/test_shell_io.py tests/test_wasm_html_shell.py -q
python -m pytest tests/test_reversible_machine_execution.py tests/test_virtual_filesystem_and_system_tape.py -q
```

Before publication, rerun the broad reversible-machine selection recorded in
`speaktome/AGENTS/experience_reports/1785935145_DOC_Reversible_Binary_Machine_Continuation_Handoff.md`
and run `git diff --check`.

Browser validation must use Playwright at desktop and mobile widths and assert,
not merely screenshot:

- no fatal shell banner or console exception;
- nonblank GLSL display;
- selected subject digest and RIP;
- resident/tape/blocked execution status;
- terminal command/output bytes;
- forward and reverse state digests;
- reload hydration;
- no request to a Python/Node loopback endpoint.

Validate the root repository with:

```powershell
Set-Location C:\dev\Powershell
go test ./...
go run .
```

The generated version must appear in the root gallery and under the root
`site/`, not under `turing/site/` or `turing/build/`.

## Stop conditions and honest failure reporting

Do not declare completion when any of these remain true:

- Python Dream sections still say they have no executable artifact.
- The page needs `reversible_machine_web_host.py` for subject execution.
- Only the authored fixture runs.
- `cmd.exe` behavior is retained replay without resident instruction execution.
- Memory/device journals cannot commit in the browser.
- System effects mutate browser state without exact machine completion edges.
- File replacement does not rebuild the actual PE state.
- Reverse controls only change a snapshot index.
- The manifest says `prebuilt-program-interior`.
- The bundle exists only in a local Turing build directory.

When blocked, record the exact unsupported section, SSA operation, AMD64
semantic, capability identity/shape, artifact digest, or browser API. Never
replace a named blocker with a broad statement such as "Wasm cannot do this."

## First implementation move

Start at `DreamDocument.compile_sections()` and make the Python `chip-setup`
section return one small executable artifact for a bounded, pure coordinator
operation already expressible in the current AOT IR. Thread that artifact into
`emit_dream_html_shell()` metadata and the existing normal bundle build, then
add a focused test proving the emitted bytes are source/IR-derived and digest
bound. This is the cheapest check of the controlling hypothesis: if that small
section cannot pass through the ordinary compiler, attempting to package the
full resident machine first will only recreate the current prebuilt detour.

## Required reading before continuation

- `turing/AGENTS.md`
- `turing/docs/REVERSIBLE_MACHINE_CHIP.md`
- `turing/docs/REVERSIBLE_MACHINE_COMPLETION_AUDIT.md`
- `turing/docs/SHELL_SYSTEM_PORTS.md`
- `TURING_REVERSIBLE_MACHINE_DEMO_HANDOFF.md`
- `PUBLISHING_BUNDLES_TO_ROOT.md`
- `speaktome/AGENTS/experience_reports/1785935145_DOC_Reversible_Binary_Machine_Continuation_Handoff.md`
- `speaktome/AGENTS/experience_reports/1785939034_DOC_Reversible_Machine_Demo_And_Graph_Crop_Update.md`

This document is a continuation contract, not an assurance that the remaining
work has already been implemented.
