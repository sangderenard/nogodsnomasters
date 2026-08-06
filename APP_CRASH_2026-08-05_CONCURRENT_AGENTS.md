# App crash during two concurrent agent sessions — 2026-08-05

## What happened

Around **2026-08-05T15:39–15:42 UTC**, the Claude Code host process exited
(crashed or was killed) while **two sessions were running concurrently** in
this repo:

- Session `5d9332bf-1307-4c9d-8fe2-134f5390248a` — had a background shell
  command in flight (`toolu_01Q8Pz3iUytS9ju5KNtYsnph`, task id `bfyorw6jx`).
  Its own transcript shows a silent gap from `15:39:37Z` to `15:42:45Z`,
  after which it received a `<task-notification status="stopped">` reading:
  *"No completion record was found for this background shell command from
  the previous session... it may have been running when the previous Claude
  Code process exited."*
- Session `6186d513-b45c-44d0-8929-cdeac1d20ddf` — had a background
  `Explore` subagent in flight (agent id `a7a5d4695778fb93f`, researching
  the speaktome universal executor / reversible-machine console code). Its
  subagent transcript shows the same kind of gap, last activity at
  `15:40:27Z`, then a resume message at `15:42:09Z` telling it: *"You were
  stopped mid-task (the host app dropped and reconnected, not something I
  did)."* Its metadata file was written with `"stoppedByUser": true`.

Both sessions were cut off in the same ~2–3 minute window and both came
back to an identical "stopped, no completion record, possibly the whole
process exited" notification. That's strong evidence this was a single
**app-level crash/restart affecting the whole host process**, not something
scoped to one session or one tool call.

## What it was *not* (as far as the logs show)

The user reported it as caused by one of the agents "using grep instead of
looking at recent files." I checked for that specifically:

- Every `Grep`/`grep` invocation in both session transcripts around the
  crash window was scoped to a single subdirectory
  (`turing/src/compiler`), not a repo-wide or recursive scan — nothing that
  should have been heavy enough to OOM or hang a shell.
- No `ENOMEM`, `OOM`, heap-exhaustion, or stack-overflow errors appear in
  either transcript. The only in-band errors near that time are unrelated
  `ECONNRESET` API hiccups ~14:04–14:06Z, well before the crash window.
- The assistant in session `6186d513` pushed back on the "your grep crashed
  it" claim in real time, attributing the interruption instead to the host
  app dropping/reconnecting — which matches what both transcripts
  independently show.

**Conclusion:** the logs support "the whole app crashed once, in the middle
of two concurrent agent sessions" happening on 2026-08-05 around 15:40 UTC.
They do not show a specific tool call, grep, or resource blowup as the
cause — the crash left no diagnostic trace inside either transcript, only
the aftermath (`stopped` notifications on both sides). If this recurs, the
next place to look is host-process-level logs (outside these per-session
`.jsonl` transcripts), since neither agent's own log captured why the
process went down.

## Note

This is the **second time in a row** this kind of app-wide crash/restart
has occurred during a run with two agents active concurrently. Worth
watching for a pattern if it happens a third time.
