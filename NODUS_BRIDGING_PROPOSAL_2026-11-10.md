# Nodus bridging proposal, 2026-11-10

Premise from Albert: a whole compiled program is ONE row. The system is tick
scheduled. A row may be a blocking complex program once entities let several
messengers run concurrently. A table becomes a program only by behaving inside
the tick table architecture.

Everything below is read from source by seven read-only investigators today.
Citations are file:line in the named repo. Nothing was edited, built or run.
The table runtime is not changed by this proposal; the two places where a
change would touch runtime-adjacent code are flagged as decisions.

## 1. What the tick table architecture requires of a program-row

Facts (nodus, src/thread_manager.cpp unless noted):

- A module's rows are a strict top-to-bottom sequence on ONE per-module
  `RawStackFrame`: loop at :841, frame from `ensure_module_tool_stack` :263-278,
  reset every tick at :757. Row N's pushes are row N+1's pops.
- Plugin rows run `ITool::execute_stack` (:985). `succeeded=false` breaks the
  row loop and skips the module's remaining rows including Output rows (:986-992).
- Stack capacity is fixed 4096 bytes (include/thread_manager.h:385); an
  overflowing push fails silently (value_types.h:~91, return ignored at push_value).
- Input rows consume one token from their edge onto the stack (:842-946); an
  Input row with no token pushes 0.0 (legacy path). Output rows pop one value
  per attachment and publish (:1384-1477); with no edge the value is discarded.
- Modules run in parallel only when the module graph is acyclic; threads are
  created and joined per tick (:1579-1592). The tick ends when every module has
  joined, and `submit_tick(wait)` waits on it (:351-362). A long row therefore
  blocks the whole canvas tick today. The Scheduled/FreeSpinning flag is
  reserved and currently identical (:394-397). Per-module `exec_skip_count` and
  sleep delay exist (:683-692, :800-821) but run inside the shared tick.

Consequence: a program-row is a step function. Per tick it pops its declared
argument handles, advances once, pushes its declared return handles. Nothing
survives on the stack between ticks. Its cross-tick state must be tensors it
owns in the arena (ToolPortKind::Internal exists for declaring that count,
tool_api.h). This is the dt-system step shape; Turing products whose control
entry is a per-tick advance fit without adaptation. Products whose control
entry runs a whole loop to completion are one blocking row, acceptable only
under section 5.

## 2. The ABI boundary is already handle-based and copy-free at the stack

- `AbstractTensorHandle` is `{uint64 id}` (abstract_tensor_handle.h:7-9); the
  POD descriptor is `NodusTensorDesc`, rank <= 8, element strides
  (tensor_abi.h:65-77). Only the 8-byte handle crosses a row (value_types.h:452-455).
- `nodus_tensor_map` is a pinned pointer borrow, no copy (in_memory_backend.cpp:2251-2268).
- The Buffer ABI wrapper (artifact_importer.cpp:841-1077) maps inputs and
  passes the pointers straight into `buffers[]` (:1022-1024): inputs are not
  copied. Outputs and workspaces are created fresh EVERY call (:996, :1010),
  workspaces zeroed (:1017), workspaces destroyed after (:1054). Shape and
  dtype mismatch, and runtime validation code != 0, set `succeeded=false`,
  unmap, destroy, and restore the inputs to the stack (:954-960, :1049-1052).
- The YAML (Fortran) wrapper (:629-840) is scalar-only: it reads element 0 of
  each input by copy and never sets `succeeded`, so it cannot refuse. It is
  not the row route for tensors; the Buffer ABI V2 route is.
- FIFO edge: fast path one backend-to-backend transfer (edge_tensor_fifo.h:1036);
  fallback is ~3 copies through a host temp (:1053-1090). Full ring with
  `top_k==0` refuses the push, no drop (:815-833). 16 slots default (:182).
- `ValueTypeRegistry::global()` is NOT shared across a DLL boundary
  (tool_api.h:313-325): rows in separate DLLs cannot share struct types. Use
  `VT_ABSTRACT_TENSOR` / `VT_VOID_PTR`, or compile rows into the host image
  (the composite path, module_library_actualizer.cpp:372-389, exists but repo
  package ingest does not use it: it builds one DLL per TOOL, plugin_manager.cpp:156).

Memory cost of a program-row per tick, from the code: output allocation plus
workspace allocation and zeroing. That is the one per-tick allocation the
wrapper does; inputs are borrowed. Whether that is acceptable depends on the
program; the wrapper could not reuse outputs without an owner for them, which
is a design decision (section 7), not a patch.

## 3. The importer already does one program = one row

`gp_artifact_import` selects one entry (control, else first numerical,
:594-605), writes one wrapper, one `TOOL` line, one package (:1134-1206).
`turing_product_adopter.cpp:437` requires `wrappers_written == 1`. The
manifest format accepts many TOOL lines (repo_package.cpp:190-215) but that is
not needed under the one-program-one-row premise. No multi-entry work.

## 4. The automation gap is on the Turing side: programs do not all describe themselves

- Turing writes `turing-compiled-program-api-v1` beside Fortran artifacts
  (ssa_fortran_backend.py:6220-6226), standalone shell products
  (fortran_c_shell.py:10721-10723) and per native link of a project product
  (project_compilation_product.py:2335-2355). The Buffer ABI V2 descriptor is
  written by nodus/tools/turing_row_tool_compiler.py per request.
- Every Meowdoku product writes a JSON ABI of its own family instead:
  `turing.fountain-products.v3` (fountain_build.py:50,203-218),
  `turing.graph-buffer-abi.v1` (graph_wasm.py:41), `turing.mobile-wasm-abi.v1`
  (entity_motion.py:1172, policy_wasm.py:47), `turing.split-js-wasm.v1`
  (split_lowering.py:119). None carries the API YAML. There is no
  program-level manifest listing all products of one program.
- The Meowdoku server process consumes no DLL at all (no ctypes/CDLL in
  server.py or session.py); every compiled product is shipped to the browser.

So "a program that can describe the API of its binaries" is true for the
native lanes and false for the web lanes. The bridging step is: every product
family writes the same self-description the importer reads, from the same
Function objects (compiled_program_api.py already exists for that), beside the
JSON it writes today. No new schema; the existing one, at the existing writer
sites listed above.

## 5. Messenger rows on the canvas and the window

- A `PluginTool(plugin_id)` row in `nodus.graph-definition.v1` installs via
  `tool_registry_global().create(plugin_id)` with no DLL, source or load step
  (nodus_graph/native.py:413-428 -> nodus_headless_abi.cpp:428 ->
  canvas_abi_rope_map.inl:~5708, ~782). The negotiator rows register exactly
  that way (`register_negotiator_tools`, contract_negotiator_tool.h:30-46) and
  must be registered by C++ in the same process before `set_rows`.
- Gate: when no tool is registered under the id, adoption prints
  `no registered tool for plugin_id=...` to stderr and leaves a null instance
  (rope_map.inl:~795); the call does not fail. The row then falls through to
  the builtin switch at execution (thread_manager.cpp:995). A messenger
  canvas adopted in a process that forgot to register would run silently
  wrong. This is DECISION A below; it is canvas adoption code, not the tick loop.
- The negotiator can bind its result transport to a graph edge instead of its
  standalone FIFO (contract_negotiator.h:55-61). Payloads are engine-owned ids
  (64-byte `Result`), never bytes (:3-5). `NegotiatorTool::execute_stack` runs
  one market or worker turn and touches no stack (contract_negotiator_tool.h).
- Window: headless and GUI cannot share a process (`nodus_headless_create`
  refuses when a canvas singleton exists, nodus_headless_abi.cpp:244-247).
  Hand-off is file based: `Graph.save` writes CANVAS V4 (native.py:1191,
  save_load.inl:62); `frontend_shell --workspace=<path>` loads the same format
  (frontend_shell.cpp:1204, 2190). CANVAS V4 does not persist FIFO specs
  (nodus_headless_abi.h:126-129). The GUI needs the same plugin ids registered
  in its own process or rows come up as names only (frontend_shell.cpp:1396-1402).
  `frontend_shell` builds only with `NODUS_ENABLE_TORCH` plus SDL3 and OpenGL
  (CMakeLists.txt:646-660). No Python-side spawn exists.
- Therefore "the server spawns a window to watch and control the messengers"
  means: the GUI process runs the messenger canvas, and the Python server is a
  participant through the existing `contract_socket.h` loopback protocol
  (72-byte frames), not the owner of the graph. That is already how the
  negotiator's own test joins the server as "another Host callback".

### 5a. Albert's refinement (2026-11-10): portable canvas + socket-typed top-level port

The prototype connection to a ThreadManager instance (the route
`tests/contract_socket_graph_test.cpp` proves: registered tools, loopback TCP,
native FIFO, producer commit) is to become a portable canvas file, and the
canvas gets a top-level port whose type is a socket, so a running Nodus and
the Python server connect directly through that port.

What exists for it:
- Named top-level ports already exist in graph-definition v1 and the headless
  ABI (`PortDefinition(name, connector, fifo, start_at_head)`, model.py:258;
  `nodus_headless_publish_edge_token` / `consume_edge_token` added 2026-11-09).
  Today a port is exchanged by an in-process caller.
- `contract_socket.h` is the socket adapter, but as a ROW that takes an
  already-connected socket and speaks the 72-byte v1 frames. It is not a port
  type.
- CANVAS V4 persists rows and the plugin id (save_load.inl:356-361) but NOT
  FIFO specs or named host ports (nodus_headless_abi.h:126-129). A portable
  messenger canvas therefore cannot yet be one file: the port and FIFO
  configuration would have to travel beside it (graph-definition v1 carries
  them; CANVAS V4 does not).

What it would take, named without designing it:
- a port transport kind on the named-port declaration (socket) so that a
  consumer/publisher of that port is the socket adapter rather than an
  in-process caller; the adapter logic is already contract_socket.h;
- the saved canvas carrying its ports and FIFO specs, either by extending what
  CANVAS V4 serializes (canvas save/load code) or by shipping the
  graph-definition v1 mapping beside the V4 file as the portable unit.
Both touch canvas serialization or port declaration, not the tick loop. This
is DECISION D.

## 6. Why the messenger loop is slow today, by receipt

spectral-analyzer/build/dot-first-hit-loan-20261109/production-pump-receipt.json
(256 rays, 95,217 triangles): total 27.3 s. Components overlap; do not sum.

| component | seconds | where it comes from |
| --- | --- | --- |
| capture_setup | 19.80 | per-strip compute program compile and buffer allocation, studio_first_hit_loan.py:99-105 |
| mailbox (all HTTP) | 4.43 | every relay call is JSON over loopback TLS; scene JSON is 35 MB (http_request_bytes 35,096,948) |
| exchange_wall | 4.18 | 0.1 s sleeps in ready-wait, publish retry and take loop (studio_cooperative_sensor.py:94, :42, :147) |
| native_step | 2.25 | `host.step()` wall time including those waits |
| scene_registration | 2.90 | JSON scene transport |
| client compute | 0.052 | the actual Wasm first-hit work |
| verification | 0.005 | GPU verifier |
| server-only baseline | 0.066 | sensor and readback without sharing |

Ray data crosses: GL readback copy, `rays.copy()`, `tolist()`, `json.dumps`,
three `deepcopy` in the relay (client_ray_relay.py:90, :171, :199), `fetch`
JSON, `Float32Array(flat())`, Wasm memory copy, `Array.from`, JSON back,
`json.loads`, `np.asarray`, `glBufferSubData`. Ray data never crosses the Nodus
FIFO; the FIFO carries 64-byte handles over a socketpair from a Python-hosted
DLL (nodus_mail_host.py:22, contract_service_peer.py:20-21). A new
`NativeMailHost` is created per strip (cooperative:149). The relay's
`submit/publish` hard-code ray and hit semantics, so the generic
offer/accept/commit layer is not yet factored out of the ray layer.

The compute is 0.05 s; everything else is boundaries that exist because each
piece is a standalone product with its own transport. On the canvas those
boundaries become FIFO edges on one tick: delay per hop is one tick period
(an Input row consumes the previous tick's publish), and the 35 MB scene JSON
becomes a handle to a resident tensor. That is what makes the delay analytic.

## 7. Proposed order of work (existing mechanisms only)

1. Turing self-description for the web lanes: write the existing
   `compiled_program_api` YAML beside each Meowdoku product family's JSON, at
   the four writer sites in section 4. Receipt: every product directory the
   server builds carries an `.api.yaml` whose entry_points agree with the JSON
   ABI's inputs/outputs. No importer change.
2. Nodus messenger package: register the negotiator and socket rows in the
   host image (same binary as the headless lib and as frontend_shell), and
   adopt a two-module messenger canvas through graph-definition v1 with the
   negotiator's result transport bound to the graph edge. Receipt: the existing
   `contract_socket_graph_test` route expressed as a saved CANVAS V4 workspace
   that `frontend_shell --workspace` opens with live rows.
3. Replace the Python `NativeMailHost` + socketpair with the Spectral engine
   joining that canvas's socket as a Host callback; scene becomes a resident
   tensor handle, rays stay engine-owned by id. Receipt: same 256-ray bitwise
   parity, mailbox and exchange seconds measured on the canvas.
4. Program-rows: compile one Meowdoku per-tick law (the `damped_axis` product
   already lowered with zero concordance disagreements) through the row
   compiler and place it on the same canvas as a row. Receipt: same tick
   values as the Python law.
5. Concurrency: entities are a documented contract with no code
   (GRAPH_ENTITY_AND_SPRITE_CONTRACT.md:3; no EntityRecord anywhere). Until
   E0-E5 exist, several messengers are several modules in one acyclic canvas
   (parallel within the tick but joined) or several canvases (frontend_shell
   steps them serially on one thread, :1104-1140). A blocking program-row
   stalls its canvas's tick either way.

Prerequisite gates already named by the 2026-11-07 audit and still open:
absent Input token fabricates 0.0; full lossless Output ignores a failed
publish; the cycle fix in thread_manager.cpp is uncommitted and its build never
ran; the Python-row wrapper reports success on interpreter failure.

## Decisions that are Albert's, not mine

A. Adoption with an unregistered plugin id: leave the stderr message, or make
   `set_module_rows` refuse (canvas adoption code, rope_map.inl:~795).
B. Cross-tick state of a program-row: tensors the row owns (Internal ports,
   allocated once in `initialize`), or state tensors allocated by a
   `TensorAllocator` builtin row and routed through inout columns. The Buffer
   ABI wrapper today allocates outputs fresh each tick; reusing them needs an
   owner.
C. Whether frontend_shell's Torch requirement stays, since the server machine
   must build it to spawn the window.
D. Portable messenger canvas: extend CANVAS V4 to carry named ports and FIFO
   specs, or make the portable unit the graph-definition v1 mapping beside the
   V4 file; and whether the socket becomes a port transport kind or stays a row.

## Status 2026-11-10 (end of day)

Per commit messages; measured items are those the messages name. Detail: `nodus/docs/NODUS_BRIDGING_CONTRACT_PORTS_2026-11-10.md`.

1. Turing web-lane self-description: DONE for the wasm/WGSL writers. Turing worktree C:/Temp/wtb-api, branch bridging/web-lane-api, 0d267080 and cc1b307e. Measured: test_meowdoku_product_api_yaml.py 2 passed, test_wasm_core_product.py 4 passed; policy_wasm and fountain3d write lines not exercised. Graph-forward lane (graph_wasm.py) NOT done.
2. Nodus messenger package (negotiator registry, socket rows, canvas adoption): IN PROGRESS. Done in nodus d16e395 (registry, engine socket, listener, refusal rule), 01f7d65 (headless contract ports, persisted trailer), ddd78c5 (test). Not done: GUI frontend_shell --workspace hosting contract ports (in progress); autosave does not rewrite the trailer.
3. Spectral engine joins the canvas socket instead of NativeMailHost: NOT STARTED (the Python peer tools/contract_service_peer.py exists, d16e395).
4. Program-rows (damped_axis through the row compiler): NOT STARTED.
5. Concurrency / entities: NOT STARTED.

Decisions: A resolved (programmatic adoption refuses unregistered ids, file load keeps the name fallback; d16e395). D partly addressed: the portable unit is CANVAS V4 plus appended trailer records (01f7d65), and the socket is a contract port declared on a result edge. B and C not addressed.

## Status, later 2026-11-10

- GUI hosting proven (nodus docs/receipts/GUI_CONTRACT_PORT_PROOF_2026-11-10.md). nodus pushed at `nogodsnomasters`.
- Turing branch `bridging/web-lane-api` (worktree C:/Temp/wtb-api, pushed): web-lane API contracts; lowering library (tables, scan, blobs, versions, selection, leases); authz admin/member/guest; Apache front vhost + proxy_trust. All server.py / access.py integrations are specs pending in the live tree.
- Apache: C:/Apache24 exists (2.4.58, dormant, no auth); nothing applied. Enabling the proxy before the forwarded-origin change would make every visitor lan/edit.
- Live servers (8766, 8767) untouched; nothing deployed; no restart needed yet.
