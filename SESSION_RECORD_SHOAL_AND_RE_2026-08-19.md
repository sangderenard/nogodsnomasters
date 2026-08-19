# Session record 2026-08-19: Shoal verified, re.compile at the rank seam

Full handoff: `turing/HANDOFF_SHOAL_AND_RE_TARGETS.md` (commit `7762c91`
on `codex/recursive-reduction-bridge`; five commits this session).

Headlines, measured:

* The fluid flagship is named **Shoal**. `differential_translation.py`
  reports AGREED on every observable for the first time — the last
  divergence was the native runtime binder silently reading unobservable
  channel accumulators as 0.0; channels now bind to the public `last_*`
  write-back slots and refuse when unobservable.
* The Shoal C-shell executable rebuilt with today's compiler and runs;
  against a Python-controller + verified-native-advance oracle the
  conserved sums agree to ~5e-13, while the fully-compiled dt controller
  takes a different substep trajectory (dt_next 0.01667 vs 0.00652) —
  bounded, recorded, next instrument is a frame-level differential.
* The work contract now governs the direct lanes: the shared identity
  pass runs in `compile_sympy_equations`, the C/WASM private sqrt
  spellings are contract-gated (WASM refuses honestly under exact-only),
  the 4-lane verifications pin `deploy`, and the SSA pickle carries a
  contract sidecar.
* `re._compile`: the lowering crash (`_NamedIntConstant.__reduce__ =
  None`), the two raise-boundary unresolved calls, and the dead
  materialized comprehension `range` all fell. The closure is
  python-host-free (31 functions after the new dead-pure-region sweep,
  catalogue §2.2). Emission stands at exactly ONE honest refusal: the
  carried-inout rank seam (one id declared dynamic-extent array by the
  caller and scalar by its region). Behind it, gfortran names the known
  baselined Phi-rank family. The bootstrap gate is blocked by rank
  bookkeeping, not by any Python dependency.
