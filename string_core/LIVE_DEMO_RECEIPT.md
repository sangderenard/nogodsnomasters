# Live core workbench implementation receipt

Authored entry points and UI live in `demos/`, not disposable build products.
`python -u demos/live_core.py` opens the window; **Compile whole core** builds
the actual reachable interning manager through Turing's canonical source entry.
C and LLVM consume the same admitted module and their existing artifact runtimes.
No native algorithm, handwritten FFI, SSA mutation, or Python fallback is present.

The interning and extra query/composition products are independent. A query
refusal does not replace the native method with Python or prevent interning.
Per-frame persistence is explicit: one manager lifetime per invocation, with
growing histories replayed in fresh contexts between frames. Timings are
medians of three repetitions, with native workspace preparation shown separately.

Measured Python tests: all 79 String Core tests pass, including 7 live-workbench
law/protocol tests. The window layout was inspected with the computer-use skill.

Integration findings, not masked:

- Main09e36ef7 refuses Vocabulary.expand's materialized list returns. Existing
  commit1eff49c5 addresses that and passes its3 tests in an isolated checkout.
- The demo exposed an independent authored `.slice(...)` call being folded as
  `slice(None,None,None)`. A bounded Box.slice repro traced the original fold
  writer and exact method_ref0 before editing. Its6 C/LLVM tests pass7.56s
  after the correction, plus5 neighboring slice tests19.78s.
  Draft Turing PR1111 includes that correction and the existing list fix:
  https://github.com/sangderenard/Turing/pull/1111 . Main was not edited.
- The query product still refuses an Indexed value with no source class or
  planned scalar dtype. Its exact refusal is published; it is not working.
- A first interning integration included token_domain queries and reached
  `return_arms_dtype_disagree ('int64','uint8')` in token_domain. Those calls
  belong to the independent query product; the interning product itself calls
  only intern_batch/intern and reads its actual state for telemetry.

Native verification completed on main09e36ef7: the unchanged tracked
`acceptance/manager_workload.py` compiled through the canonical entry and
concordance audit into C and LLVM libraries. Both matched Python and the
independent supplied golden law at 0, 1 and 3 rounds (6 native frames).
Receipt: `build/live_core/20261029T010739Z-254976b0/run.json`, status verified.
This is the default manager product. Its storage chart shows invariants checked
inside the authored acceptance, not separately exported telemetry. It does not
verify fingerprint columns or prove the additional methods work.

The extended interning telemetry product remains refused by concordance:
`withdrawn sequence descriptor: (('live_string_core__live_intern_workload', 1), 155)`.
Its failed receipt is `build/live_core/20261029T010008Z-62bbd7f1/run.json`.
The query product also remains refused. No fallback is used.

The substantial five-size native check uses:

```powershell
python -u demos/live_core.py --verify --methods 0 --collision-modes 0 --verify-rounds 0,1,17,300,3000 --backends c,llvm
```

The all-method command omits `--methods 0 --collision-modes 0`; a blocked query product must
return a partial/failure result, never an all-method success. Each build records
the source/core hashes, compiler commit and existing compiler fingerprint in
`build/live_core/<unique-run>/run.json`. Only generated products live there.
