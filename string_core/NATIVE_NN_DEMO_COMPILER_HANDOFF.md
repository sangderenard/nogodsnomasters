# Token Orchard native NN demo compiler handoff

## Current measured blocker

Measured 2026-10-24 against canonical Turing `main` at `daeebb36`, after PR
#923's signed-integer LLVM repair was merged:

```powershell
cd C:/dev/Powershell/turing
python -m pytest -q tests/test_perforated_network_llvm.py::test_compiled_adam_can_mask_fixed_batch_padding
```

The test fails before LLVM emission:

```text
function_address: no source cells for
'engine_dendrite_learner__forward_vjp' -- the caller named no source definition
```

PR #923 independently made Token Orchard's runtime-sized signed-integer query
and retention gates green in LLVM. It did not alter this generated-VJP source
provenance failure.

Token Orchard's eager demonstration is
`C:/dev/Powershell/string_core/demos/profiled_text_learning.py`. It exercises
the real token writer queue, graph-delta learner queue, ordered composition,
typed transform observation, AbstractNN forward/backward, and Adam updates. It
reports `native_training: False` rather than confusing eager tensor execution
with a native product.

## Required compiler result

Repair the existing generated-function provenance path in
`lower_training_motion_to_repository_ssa`. The generated forward VJP's
FunctionTable declaration must derive its source cells from the exact authored
training-motion/forward graph cells already present in the identity book. Do
not weaken `function_address`, fabricate a name-based identity, mark the
function external, or add a Token Orchard exception.

Acceptance:

1. The focused `test_compiled_adam_can_mask_fixed_batch_padding` test passes.
2. The plain parametric VJP and scalar-loss adjoint cases retain their exact
   source provenance and pass their existing gates.
3. The identity concordance audit has no new findings.
4. Token Orchard can link semantic depth four and compile one product whose
   native execution performs position, ordered/context target preparation,
   forward, reverse, and Adam update.
5. The native product publishes optimizer step, loss, and embedding epoch so
   the profiled demo can compare eager and native receipts on identical token
   and observation batches.

This repair belongs in the Turing repository and must be coordinated with its
active PR/merge sequence. No Turing source is changed by Token Orchard.
