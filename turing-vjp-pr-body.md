## Outcome

Generated AbstractNN VJP now keeps source-cell provenance through the canonical graph and SSA route. A two-row tanh derivative retains its settled 2×2 extent, and native Adam matches its Python reference.

## Change

- Declare the generated motion's function, formal bindings, and semantic outputs from their existing graph identity cells before lowering.
- Send the motion through the ordinary resolved-process-graph deployment path.
- Keep unresolved region tensor operations semantic until exact call-feed shape or scalar storage is known. Preserve dynamic span extent metadata while doing so.
- Carry settled shape and view identity through the existing specialization and tensor SSA passes.
- Add `c622_tanh_vjp_two_rows.py`, a focused native LLVM regression, and one hazards entry.

## Verification

- Windows focused VJP, tanh VJP, Adam, and dynamic-span tests: 4 passed on d2bd9038.
- Linux focused set on d2bd9038: running.
- Full typed corpus and FUZZ_EXT=4 seed on d2bd9038: running. The preceding bdea8fa9 revision returned 16 PASS, 4 named LOUD, 0 WRONG on the same fuzz seed.
- Linux gate on d2bd9038: running. On bdea8fa9, all four parts ended in pytest summaries and exactly matched a fresh untouched main f8c74d8d gate: aa 77 failures, ab 7, ac 26, rec 20. Both trees show the same scatter test absent from the committed baseline, so this is **not** a strict baseline-green claim.

## Separate observed issue

`test_perforated_network_compiles_and_executes_forward_and_vjp` reaches numerical comparison but fails on `base_weight[0,0]`: native compensated matmul returns 0.0, NumPy dot returns -1.3322676295501878e-17. This is a separate precision-policy question; no tolerance or arithmetic-policy change is included here.

Albert merges. This PR is draft until final gate and corpus receipts are attached.
