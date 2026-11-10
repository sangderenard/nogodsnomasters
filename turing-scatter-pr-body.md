## Outcome

Authored `AbstractTensor.scatter` now has one C/LLVM translation identity: `scatter_sum_double`. The overwriting `index_assign_double` kernel is mapped to the existing SSA `IndexedStore` operation.

## Observed failure

On clean main f8c74d8d, `test_numerical_lowering_routes_scatter_through_real_llvm_algorithm` expects `index_assign_double` but the lowerer emits `scatter_sum_double`. The source method, backward registry, and repeated-destination tests all define scatter as accumulation. The translation table advertised both kernels for `scatter`, so choosing by table order could silently switch to overwrite. The validator test had the same stale scatter expectation, plus a stale sign mapping expectation.

## Verification

- Focused translation, precompile, C/LLVM source corpus, and named-policy refusal tests: 6 passed on Windows.
- New `c623_repeated_scatter_sum.py` enters through `lower_ast_source_to_ssa`; C and LLVM both return `[3, 5, 0]` exactly for repeated indices.
- Full typed corpus, FUZZ_EXT seed, and authoritative Linux gate: running. One hazards entry records the failure family and pending receipts.

No baseline entry is added or failing test is skipped. Albert merges.
