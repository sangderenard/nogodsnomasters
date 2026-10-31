# Token Orchard `String` namespace compiler handoff

## Objective

Teach Turing to recognize exact calls to the non-instantiable
`token_orchard.api.String` namespace and link Token Orchard's SSA product at
the maximum semantic depth actually reached by those calls.

This is not Python `str` support and must not create a string object model.
Runtime values are integer token IDs. Byte spans exist only at ingress. The
compiled product owns one always-present, self-growing vocabulary, its bounded
request queue, writer state, and any optional workers selected by reachability.

## Existing authored contract

Read these files before changing Turing:

- `C:/dev/Powershell/string_core/token_orchard/api.py`
- `C:/dev/Powershell/string_core/token_orchard/string_shim_contract.py`
- `C:/dev/Powershell/string_core/token_orchard/semantic.py`
- `C:/dev/Powershell/string_core/token_orchard/semantic_tensor.py`
- `C:/dev/Powershell/string_core/token_orchard/ssa_contract.py`

The exact namespace identity is `token_orchard.api.String`. Version zero
implements `bytes`, `bytes_batch`, `decode_receipt`, and the checked Python
veneer `decode`. `STRING_METHOD_CONTRACTS` is the method/opcode/depth and
physical argument/result authority. Do not infer string handling from method
names or from arbitrary classes named `String`.

## Required compiler behavior

1. Resolve both `from token_orchard import String` and
   `from token_orchard.api import String` to the same exact source identities.
2. At extraction/graph ingestion, classify only the contracted methods as the
   Token Orchard component boundary. Do not pursue `api.py`'s locks, Python
   queue, daemon thread, exception veneer, or singleton implementation.
3. Preserve a receipt on every recognized call containing namespace identity,
   method identity, shim ABI version, opcode, and required semantic depth.
4. Compute the product's depth as the maximum over reachable contracted calls.
   A program using only `String.bytes` or `String.decode_receipt` links depth
   zero: no embedding positions, network, Adam state, publication, or query.
5. Link calls to the same Token Orchard source lowered through
   `src.compiler.fortran_c_shell.lower_ast_source_to_ssa`. Do not create a
   second native implementation, host callback, CPython dependency, or
   hand-written string runtime.
6. Give the compiled product one vocabulary/writer service state. All mutating
   calls enter its bounded FIFO queue and share deterministic mint order.
   Threads and queue ownership are product lifecycle state, not values stored
   in token handles.
7. Preserve the integer/status ABI. `decode_receipt` must retain the byte token
   on malformed input and return text token zero. The checked `decode` veneer
   must either reproduce its failure behavior or be refused by name until the
   target has an explicit checked-status/trap convention.
8. Reject `String()` construction and any attempt to carry a `String` instance
   through the graph. There is no instance layout and no per-string object.
9. Keep token identity scoped to the compiled product's vocabulary/snapshot
   receipt. Never treat token numeric order as text or semantic distance.
10. Join existing exact Python string facts through
    `python_capture_contract.py`: source glyph recipes, StringTable glyph
    projections, explicit UTF-8 encode aliases, and byte-join transforms. A
    reachable ready capture automatically requires this same component and
    vocabulary. Do not add a second recognizer or match a method spelling.

Use Turing's existing exact source-identity, extraction receipt, FunctionTable,
ProgramABI, class declaration, and source-linked call mechanisms. Do not add a
name-based rewrite or bypass identity concordance. If the established
intrinsic route is used, give it a dedicated `token_orchard` lowering namespace
and lower it to a receipted source-linked component call rather than a backend
opcode with an independent implementation.

## Acceptance

- A live Python program importing `String` returns stable integer tokens and
  uses Token Orchard's real writer queue.
- An unrelated `class String` with a `bytes` method is ordinary user code.
- Graph inspection shows exact Token Orchard identity and opcode receipts.
- A bytes-only compiled program contains no semantic/NN state.
- Two reachable ingress calls share one compiled vocabulary and deterministic
  writer order.
- `decode_receipt(b"\xff", UTF8)` publishes a nonzero byte token, zero text
  token, and `STATUS_INVALID_ENCODING` in Python and every compiled backend.
- C, LLVM, and Fortran agree with Python on a batch containing empty, repeated,
  valid UTF-8, malformed UTF-8, and arbitrary binary payloads.
- Identity concordance and the relevant extraction, class/import, linked-call,
  ProgramABI, and backend gates retain their baseline failure names.

Do not weaken FunctionTable source-cell requirements to make this pass. The
separate generated-VJP provenance regression still gates semantic training
depth four; ingress depth zero must not depend on that repair.

## Measured current behavior

### Integrated consumer compile, 2026-10-31

The tracked `acceptance/native_string_program.py` is one program using all
four public `String` methods. It requests byte and text tokens through the
same vocabulary and returns their integer receipts. Its compile driver uses
Turing's public `lower_ast_source_to_ssa` entry, then requests a native C
artifact. The driver refuses any execution contract that admits a Python host,
Python-resident unlowered work, or Python callbacks.

From `C:/dev/Powershell/string_core`, with the current Turing worktree:

```powershell
python -u tools/compile_native_string_program.py --turing-root C:\Users\alber\.codex\worktrees\exception-admission\turing --output C:\dev\Powershell\string_core\build\native_string_program_20261031
```

The 2026-10-31 run selected `host_runtime: native`, `whole_program`,
`unlowered_behavior: reject`, `require_full_native: true`, and
`python_callbacks: reject`. Source closure nevertheless pursued the Python
`String.bytes`, `bytes_batch`, `decode_receipt`, and `decode` bodies, then
`_one_token`, `_byte_values`, `_global_runtime`, and `_GlobalTokenRuntime`.
Reduction refused before SSA emission: `SOURCE_VALUE_CLASS` for
`('token_orchard.api', '_byte_values')` value 3 was recorded as
`builtins.bytes` from the builtin aggregate result, while a propagated
`String.bytes` call result proposed `builtins.int`. No native artifact was
produced. The exact `String` component boundary remains the cause to fix;
changing `_byte_values`'s type would not implement that boundary.

Run from `C:/dev/Powershell/string_core`:

```powershell
python tools/audit_turing_string_shim.py
```

The audit uses Turing's existing AST source-pursuit machinery with the exact
`token_orchard.api` source declared. Current Turing pursues
`String.bytes -> _one_token -> _global_runtime -> _GlobalTokenRuntime` and
then classifies `token_orchard.service.TokenService` as
`default:unknown`, action `reject`, reason `provenance_not_declared`. This is
the intended pre-repair reproduction: the compiler must stop earlier at the
exact `token_orchard.api.String.bytes` boundary and emit the component receipt,
not admit `service.py` and compile Python locks or queues.
