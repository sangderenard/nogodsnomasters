# Precision int adaptor plan

Oct 28, 2026 · @Albert Stichka

## Goal

An `int` whose range the interval solver cannot prove is computed as int64 with an overflow check, and a result that leaves int64 continues exactly as a Precision integer (64-bit limbs) instead of wrapping. This is your option 1 (13:27).

- **Today:** an unproven `int` is int64. The C backend computes `+ - *` through unsigned casts, so it wraps silently. LLVM uses plain `add`/`mul`, which also wraps. Python never wraps, so the native result is wrong without an error.
- **Proven ranges stay as they are:** int64, uint64 or int128 picked by `python_int_range_dtype`, with no check.
- **The limb arithmetic is compiled, not hand-written C:** it comes from Python routines lowered through Turing itself, under the standing rule.

## Stages

Each stage is one small PR from main, with corpus programs in c902-c909. Stage 1 alone already ends the silent wrap.

1. **Checked int64.** Signed int64 `+ - * neg <<` and `**` on a value with no proven int64 range get an overflow test. LLVM uses `llvm.s*.with.overflow`; C uses the compiler's overflow builtins in emitted code. Until stage 2 lands, an overflow raises `OverflowError` through `turing_validation_error`: loud, never wrong.
2. **Precision integer type.** An exact-type union arm `int: int64 | PrecisionInt[n]`, with `n` 64-bit limbs and a sign. It is built on the existing union ProgramABI arms, so no new class machinery is needed. The overflow branch builds the limbs from the exact wide result (the int128 pair the check already computed).
3. **Limb kernels.** Add, subtract, multiply, compare, floor-divide and mod, shifts, and `&`, `|`, `^` on limbs. They are written as Python over lists of ints and compiled by Turing into SSA helpers, called from the slow arm.
4. **Boundaries.** Returning to Python rebuilds the `int`. Storing into a typed column (for example the uint64 key columns) narrows with a range check that faults if the value doesn't fit. Hashing and dict keys hash the exact value.
5. **The string core.** Re-run the whole-core audit, then corpus programs for the `fingerprint` and `fnv1a` products whose ranges the solver still can't prove.

## Risks

The main risk is stage 1 on its own: today some unproven products wrap and then get reduced `% 2**64`, which gives Python's answer by luck. With a check and no promotion yet, those would fault instead.

- **What I'll measure first:** count the unproven int64 `+ - *` in the string core and in the corpus, and find which feed a `% 2**k` or `& mask`. If any do, stages 1 and 2 ship together.
- **Cost:** the check is one compare and branch per unproven operation. Proven operations get no check.
- **Other backends:** GLSL, JS and WASM don't run the string core. A checked operation those backends can't express is refused by name, never lowered unchecked.
