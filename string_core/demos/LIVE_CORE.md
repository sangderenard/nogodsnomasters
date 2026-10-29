# Live native String Core workbench

From `string_core`, run:

```powershell
python -u demos/live_core.py
```

Choose **Compile whole core**. This submits the tracked
`demos/live_intern_workload.py` to Turing's canonical `lower_ast_source_to_ssa`,
following the real `token_orchard.core`, `kernel`, and `contract` source.
The complete reachable workload includes the real interning, validation,
fingerprinting, exact candidate comparison, cover construction, and the
interning paths. The Python service queues, codecs and
learning adapter are outside this integer core workload.

The demo builds C and LLVM libraries from that same admitted module. The
first query/composition request independently compiles `demos/live_workload.py`
through the same entry and source contract. A compiler gap in those extra
methods is reported as a method refusal and leaves interning usable. It
does not hand-translate the algorithm, patch SSA, or fall back to Python.
Compiler progress and errors appear in both the window and the console.
The window stays responsive while a separate native worker compiles or runs.
Every build gets a unique directory, avoiding overwriting a loaded Windows DLL.

After READY, choose a method and **Run live**, or **Try all methods**:

| Method | Real core calls exercised |
| --- | --- |
| Batch interning | `intern_batch` |
| Single-item interning | `intern` |
| Expand / length / cover | Batch interning plus `token_domain`, `expand`, `length`, `cover` |
| Slice / concat / atom_at | All query calls plus `slice`, `concat`, `atom_at` |

Each round submits a distinct Unicode pair and byte triple, then repeats
the Unicode request. Checks verify stable token reuse, domain separation,
exact expansion and reconstruction, and rejection of a Unicode surrogate
without changing the epoch. **Force hash collisions** sets the actual
Vocabulary fingerprint mode to constant: exact candidate comparison must
still distinguish the different sequences.

The chart and table show actual native output: aggregate rows, atom count,
epoch, checks, last token, cover entries and a digest of the stored fingerprint
columns. All eleven output integers must match the same authored Python
workload exactly. Independent laws additionally require zero failures,
epoch/rows `2*n`, atoms `5*n`, and checksum `2*n*n+n`. A discrepancy is a
failure, even if the Python/native results happen to agree on it.

Frames run at sizes 0, 1, 2, 4, ... and the exact selected target. **Each
frame creates a fresh context and replays its own history.** Persistence is
within a frame, not between GUI frames. The demo does not claim a persistent
manager handle across host calls. Stop after frame drops queued frames;
closing the window terminates its worker.

Times are medians of three repetitions after a native warmup. The display
separates the native runtime call from native preparation plus the call.
Python timing includes constructing and executing the reference workload.
Compilation is excluded. This is a measured scenario comparison, not a
general throughput claim. Forced collisions intentionally increase work.

Headless native acceptance uses the same runner and all method/collision
combinations at 0, 1 and 3 rounds:

```powershell
python -u demos/live_core.py --verify --backends c,llvm
```

An explicit checkout can be selected with `--compiler-root C:\path\to\turing`.
To verify only the independently buildable interning product, add `--methods 0,1`.
That is explicitly partial method coverage; it never claims the query product passed.
The actual compiler commit and the core/workload source hashes are recorded
in `build/live_core/<unique-run>/run.json`. Products are disposable; all
authored workload, UI and tests live outside `build/`. No old artifact is
silently reused when sources change.

To watch the actual receipts of an already running headless verification in
the live window, without a second compilation, use
`--observe-run build/live_core/<that-run-directory>`. Its method buttons stay
disabled because that worker owns the verification sequence; **Compile whole
core** starts a separate interactive worker if desired. Observed results are
the real native run's published frames, not generated sample data.

At main `09e36ef7`, the new query/composition paths expose compiler gaps.
[Turing PR #1111](https://github.com/sangderenard/Turing/pull/1111) contains
the existing materialized-list-return fix and the observed correction that
keeps resolved `slice` method calls out of structural slice folding. It is a
draft dependency; main was not changed by this demo. The explicit local
review checkout used for integration is `C:\dev\tg-child-review`.

The workbench is also an integration probe: a refusal while compiling a
product remains visible. An interning refusal prevents READY; a query-product
refusal blocks its methods, with no fallback. `Vocabulary.expand` and `cover`
need the compiler's materialized-list-return support; the demo does not
replace them with private field reads to work around that requirement.
