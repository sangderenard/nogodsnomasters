"""WHAT THE COMPILER DID, IN FULL, EVERY TIME.

    from compile_probe import CompileProbe

    with CompileProbe() as probe:
        module, outputs, exports = probe.lower(source, "EngineCycleSim.step")
    print(probe.report())

WHY THIS EXISTS. Every compile in this project has so far been read
through a one-off monkeypatch written to answer one question, thrown
away, and rewritten differently the next time -- which is how the same
facts got re-derived three times in a night and how two of my own probes
measured the wrong thing (one instrumented a function that is not on the
path; one wrapped a function whose SOURCE feeds the cache key, so the
instrument changed the key it was measuring).

A compile that takes half an hour deserves to answer every question at
once.

NOTHING HERE IS TRUNCATED. The refusals this compiler raises carry their
whole payload -- every blocker, every opaque effect, with the source line
that caused it -- and the habit of printing `str(error)[:200]` throws
away precisely the part that names the fix. Lists are printed entire. If
that is long, the compile was long.

WHAT IT RECORDS

  phases      every `progress` message with the seconds since start and
              the seconds that phase took, so a 29-minute lowering can
              be read as a profile rather than a wait.
  effects     every `LoopStateEffect` the compiler builds, with the mode
              it was assigned. `opaque` is the DEFAULT, not a detection:
              `topological_reducer.py` classifies a loop-body mutation
              only when the state is a built-in container AND the method
              is one of eight names (add/append/clear/extend/pop,
              update/pop/setdefault). Everything else -- every `.step()`
              ever written -- falls through. So the opaque inventory is
              the list of state transitions this compiler has no model
              for, and that list is the specification for what a record
              has to cover.
  abi         every `records_for_function` / `values_for_function` answer.
              These two paths are NOT equivalent: a `values` declaration
              has a live consumer on the function-lowering path, a
              `records` declaration is attached and (there) read by
              nothing. Seeing both answers side by side is how that was
              found.
  cache       every checkpoint load, hit or miss, with the reason and the
              seconds. A cache HIT can be the whole cost of a startup --
              measured, 115 s of cloudpickle for two laws -- and looks
              exactly like a slow compile unless you can see it.
  warnings    kept, not swallowed. `_report_unmaterialised_record_parameters`
              reports rather than raises, and an emission that looks
              suspiciously empty is usually explained there.
"""
from __future__ import annotations

import collections
import re
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any

import graph_physics  # puts turing on sys.path  # noqa: F401

_DIGITS = re.compile(r"\d+")
_TAGGED = re.compile(r"^\[([a-zA-Z][\w-]*)\s*#\d+\]")


@dataclass
class Phase:
    message: str
    at_s: float
    took_s: float = 0.0
    repeats: int = 0


@dataclass
class AbiAnswer:
    kind: str                    # "records" | "values"
    function: str
    parameters: tuple
    answered: tuple
    method_owner: Any = None


@dataclass
class CacheEvent:
    phase: str
    identity: str
    status: str
    seconds: float


@dataclass
class CompileProbe:
    """Install every reader, run the lowering, keep everything."""

    phases: list = field(default_factory=list)
    effects: list = field(default_factory=list)
    abi: list = field(default_factory=list)
    cache: list = field(default_factory=list)
    caught: list = field(default_factory=list)
    refusal: BaseException | None = None
    refusal_traceback: str = ""
    started: float = 0.0
    finished: float = 0.0
    unit_plans: list = field(default_factory=list)
    _restore: list = field(default_factory=list)

    # -- installation --------------------------------------------------
    def __enter__(self) -> "CompileProbe":
        self.started = time.perf_counter()
        self._hook_effects()
        self._hook_abi()
        self._hook_cache()
        self._warnings = warnings.catch_warnings(record=True)
        self.caught = self._warnings.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc) -> bool:
        self.finished = time.perf_counter()
        try:
            self._warnings.__exit__(*exc)
        except Exception:
            pass
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()
        return False

    def _patch(self, owner, name, replacement) -> None:
        self._restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def _hook_effects(self) -> None:
        """Record every loop state effect and the mode it was given.

        Patched on `loop_composer`, not on `loop_ir`: the construction
        sites resolve the name from the composer's own globals at call
        time, so that is where a replacement is seen.
        """
        import src.compiler.loop_composer as lc
        from src.compiler.loop_ir import LoopStateEffect as Real

        record = self.effects

        def build(**kw):
            effect = Real(**kw)
            record.append(effect)
            return effect

        self._patch(lc, "LoopStateEffect", build)

    def _hook_abi(self) -> None:
        from src.compiler.extraction_contract import ProgramABIContract

        records_real = ProgramABIContract.records_for_function
        values_real = ProgramABIContract.values_for_function
        answers = self.abi

        def records_for_function(inner, function_name, *, method_owner=None,
                                 parameters=()):
            out = records_real(inner, function_name, method_owner=method_owner,
                               parameters=parameters)
            answers.append(AbiAnswer("records", str(function_name),
                                     tuple(parameters), tuple(out),
                                     method_owner))
            return out

        def values_for_function(inner, function_name):
            out = values_real(inner, function_name)
            answers.append(AbiAnswer("values", str(function_name), (),
                                     tuple(out)))
            return out

        self._patch(ProgramABIContract, "records_for_function",
                    records_for_function)
        self._patch(ProgramABIContract, "values_for_function",
                    values_for_function)

    def _hook_cache(self) -> None:
        """A cache HIT can be the entire cost of a startup. Time them."""
        from src.common.tensors.accelerator_backends import aot_checkpoint as ac

        real = ac.AOTCheckpointStore.load
        events = self.cache

        def load(inner, phase, implementation):
            began = time.perf_counter()
            value = real(inner, phase, implementation)
            events.append(CacheEvent(str(phase), inner.identity[:12],
                                     str(inner.last_load_status),
                                     time.perf_counter() - began))
            return value

        self._patch(ac.AOTCheckpointStore, "load", load)

    # -- the run -------------------------------------------------------
    #: Two messages are "the same phase repeating" when they match once
    #: every number is blanked. The graph builder emits one message per
    #: AST node -- 22 for a three-line function, tens of thousands for the
    #: engine -- and they differ only by an index and a node id, so a
    #: plain prefix match does not fold them and the handful of messages
    #: that say where half an hour went get buried under them.
    FOLD_WIDTH = 48

    @staticmethod
    def _fold_key(message: str) -> str:
        """What makes two messages the same phase.

        A message of the form `[tag #n] ...` is per-node instrumentation
        by construction -- the graph builder emits one for every AST node
        it visits -- so the TAG alone is the key and the rest is a count.
        Anything else folds only when it matches with its numbers blanked,
        which catches `aot: lowering N region(s) for shell X` without
        merging genuinely different phases.
        """
        tagged = _TAGGED.match(message)
        if tagged:
            return tagged.group(1)
        return _DIGITS.sub("#", message)[:CompileProbe.FOLD_WIDTH]

    def note(self, message: str) -> None:
        now = time.perf_counter() - self.started
        message = str(message)
        if not self.phases:
            # The first progress message can arrive long after the start:
            # measured, 10.3 s of a 10.5 s lowering happened before any
            # phase announced itself. Unrecorded, that reads as a compile
            # that took no time and then finished.
            self.phases.append(Phase("(before the first phase message)", 0.0,
                                     now))
        else:
            self.phases[-1].took_s = now - self.phases[-1].at_s
        last = self.phases[-1]
        if last.repeats and self._fold_key(message) == self._fold_key(
                last.message):
            last.repeats += 1
            return
        self.phases.append(Phase(message, now, repeats=1))

    def lower(self, source: str, entrypoint: str, *, name: str = "probe",
              extraction_contract: Any = None, **kwargs):
        """`lower_ast_source_to_ssa` with every hook wired.

        A refusal is KEPT, not re-raised: the payload is the finding, and
        the report prints it whole.
        """
        from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

        if extraction_contract is None:
            import compile_contract
            extraction_contract = compile_contract.contract()
        try:
            result = lower_ast_source_to_ssa(
                source, entrypoint, name=name,
                extraction_contract=extraction_contract,
                progress=self.note,
                compilation_unit_plan_sink=self.unit_plans.append,
                **kwargs)
            self.note("done")
            return result
        except BaseException as error:        # noqa: BLE001 - kept, reported
            self.note(f"REFUSED: {type(error).__name__}")
            self.refusal = error
            self.refusal_traceback = traceback.format_exc()
            return None

    # -- the report ----------------------------------------------------
    def report(self) -> str:
        out: list[str] = []
        add = out.append
        total = (self.finished or time.perf_counter()) - self.started
        add("=" * 78)
        add(f"COMPILE PROBE   {total:.1f} s total")
        add("=" * 78)

        add("")
        add("PHASES -- where the time went")
        for phase in self.phases:
            tail = f"  (x{phase.repeats})" if phase.repeats > 1 else ""
            add(f"  {phase.at_s:8.1f}s  +{phase.took_s:7.1f}s  "
                f"{phase.message}{tail}")

        add("")
        add("STATE EFFECTS -- what the compiler can and cannot model")
        by_mode = collections.Counter(e.mode.value for e in self.effects)
        add(f"  {len(self.effects)} effects: {dict(by_mode)}")
        opaque = [e for e in self.effects if e.mode.value == "opaque"]
        if opaque:
            add("")
            add(f"  OPAQUE -- {len(opaque)} state transitions with no model.")
            add("  `opaque` is the DEFAULT, not a detection. This list is the")
            add("  specification for what a record has to cover.")
            no_output = sum(1 for e in opaque if e.state_output_id is None)
            add(f"  {no_output} of {len(opaque)} have no state_output_id --")
            add("  each is a loop recurrence that cannot close.")
            add("")
            add(f"  {'receiver':<44}{'n':>4}  operators")
            by_state = collections.Counter(e.state_name for e in opaque)
            for state_name, count in by_state.most_common():
                ops = sorted({e.operator for e in opaque
                              if e.state_name == state_name})
                add(f"  {str(state_name):<44}{count:>4}  {', '.join(ops)}")

        if self.abi:
            add("")
            add("ABI -- what each declaration form answered")
            for answer in self.abi:
                add(f"  {answer.kind:<8} fn={answer.function!r} "
                    f"owner={answer.method_owner!r}")
                add(f"           parameters={answer.parameters!r}")
                add(f"           answered  ={answer.answered!r}")

        if self.cache:
            add("")
            add("CHECKPOINT CACHE -- a hit can be the whole startup")
            for event in self.cache:
                add(f"  {event.phase:<20}{event.status:<44}"
                    f"{event.seconds:8.1f}s  {event.identity}")

        messages = [str(w.message) for w in self.caught]
        if messages:
            add("")
            add(f"WARNINGS -- {len(messages)}, kept in full")
            for message in dict.fromkeys(messages):
                add(f"  {message}")

        if self.refusal is not None:
            add("")
            add("REFUSAL -- entire payload, nothing elided")
            add(f"  {type(self.refusal).__name__}: {self.refusal}")
            add("")
            add("  traceback:")
            for line in self.refusal_traceback.splitlines():
                add(f"  {line}")
        return "\n".join(out)
