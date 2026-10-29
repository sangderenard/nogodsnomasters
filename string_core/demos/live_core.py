"""Live whole-core native demo. No native algorithm or FFI lives in this UI.

Run from string_core: python -u demos/live_core.py
Headless acceptance: python -u demos/live_core.py --verify --backends c,llvm
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid

PROJECT = Path(__file__).resolve().parents[1]
TURING = Path(os.environ.get("STRING_CORE_TURING_ROOT", str(PROJECT.parent / "turing"))).resolve()
for path in (PROJECT, TURING):
    sys.path.insert(0, str(path))

FIELDS = ("failures", "checks", "epoch", "rows", "atoms", "cover_tokens",
          "last_token", "last_length", "last_cover", "checksum", "fingerprint")
METHODS = ("Whole manager acceptance", "Single-item interning", "Expand / length / cover",
           "Slice / concat / atom_at")
PREFIX = "@@LIVE "


def event(kind, **values):
    print(PREFIX + json.dumps({"kind": kind, **values}), flush=True)


def frame_sizes(limit):
    values = [0]
    size = 1
    while size < limit:
        values.append(size)
        size *= 2
    if limit:
        values.append(limit)
    return values


def check_result(result, expected, rounds, collision):
    if tuple(result) != tuple(expected):
        differences = {key: {"native": actual, "python": want}
                       for key, actual, want in zip(FIELDS, result, expected)
                       if actual != want}
        raise AssertionError(f"Exact native/Python disagreement: {differences}")
    metrics = dict(zip(FIELDS, result))
    for key, want in (("failures", 0), ("epoch", 2 * rounds),
                      ("rows", 2 * rounds), ("atoms", 5 * rounds),
                      ("checksum", 2 * rounds * rounds + rounds)):
        if metrics[key] != want:
            raise AssertionError(f"Independent law {key}: {metrics[key]} != {want}")
    if collision and metrics["fingerprint"] != 0:
        raise AssertionError("Forced-collision fingerprints must be zero")
    return metrics


def check_manager_result(actual, expected, rounds):
    """Read the existing acceptance checksum; label its checked invariants.

    Only checksum is exported here. The tracked program explicitly verifies
    the counts below before returning its supplied golden checksum. These
    values are therefore not advertised as independently exported telemetry.
    """
    if expected != 3 * rounds * rounds + 7000172 * rounds:
        raise AssertionError("Python manager violates supplied golden law")
    if actual != expected:
        raise AssertionError(f"Native manager {actual} != Python/golden {expected}")
    from token_orchard.contract import AGGREGATE_BASE
    return {"failures": 0, "checks": 5 * rounds + 9,
            "epoch": 2 * rounds, "rows": 2 * rounds,
            "atoms": 5 * rounds, "cover_tokens": 5 * rounds,
            "last_token": AGGREGATE_BASE + 2 * (rounds - 1) if rounds else 0,
            "last_length": 0, "last_cover": 0, "checksum": actual,
            "fingerprint": None}


class NativeCore:
    def __init__(self, backends, profile="manager"):
        import numpy as np
        from src.compiler.extraction_contract import ExtractionContract
        from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
        from src.compiler.identity_concordance import CorrelationTable
        from src.compiler.project_compilation_product import compiler_toolchain_fingerprint
        self.np = np
        self.backends = {}
        self.directory = PROJECT / "build" / "live_core" / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        self.directory.mkdir(parents=True)
        self.profile = profile
        if profile == "manager":
            entry = "repeated_manager"
            source_path = PROJECT / "acceptance/manager_workload.py"
            from acceptance.manager_workload import repeated_manager
            self.reference = repeated_manager
        elif profile == "queries":
            entry = "live_workload"
            source_path = PROJECT / "demos/live_workload.py"
            from demos.live_workload import live_workload
            self.reference = live_workload
        else:
            entry = "live_intern_workload"
            source_path = PROJECT / "demos/live_intern_workload.py"
            from demos.live_intern_workload import live_intern_workload
            self.reference = live_intern_workload
        source = source_path.read_text(encoding="utf-8")
        self.manifest = {
            "source": str(source_path), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "compiler_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=TURING, text=True).strip(),
            "core_sources": {name: hashlib.sha256((PROJECT / "token_orchard" / name).read_bytes()).hexdigest()
                             for name in ("core.py", "kernel.py", "contract.py")},
            "backends": list(backends), "status": "compiling", "frames": [],
            "compiler_sha256": compiler_toolchain_fingerprint()["sha256"],
            "product": profile, "capability_failures": [],
        }
        self.save()
        event("phase", message=f"Lowering the real {self.manifest['product']} core", directory=str(self.directory))
        contract = ExtractionContract(TURING / "extraction_contracts/program_extraction.yaml").with_sources(
            [("token_orchard." + name, PROJECT / "token_orchard" / (name + ".py"))
             for name in ("contract", "kernel", "core")]
        ).with_execution_file(TURING / "extraction_contracts/vehicle_full_native_execution.yaml")

        def progress(message):
            print(message, flush=True)
            if "phase:" in message or "stage " in message or "ssa-program" in message:
                event("progress", message=message[-280:])

        started = time.perf_counter()
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, entry, name="live_string_core", extraction_contract=contract,
            progress=progress, compile_profile=True)
        findings = CorrelationTable.build(module).findings(module)
        if findings:
            raise RuntimeError(f"Concordance refused native admission: {findings}")
        if len(exports) != 1:
            raise RuntimeError(f"Expected one demo entry, received {exports}")
        self.export = exports[0]
        root = module.functions[self.export]
        self.parameters = dict(root.metadata["parameter_names"])
        returns = {tuple(int(value.id) for value in instruction.args)
                   for block in root.blocks.values() for instruction in block.instrs
                   if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")}
        if len(returns) != 1 or len(next(iter(returns))) != (1 if profile == "manager" else len(FIELDS)):
            raise RuntimeError(f"Unsettled demo return layout: {returns}")
        self.outputs = next(iter(returns))
        self.manifest.update(lowering_seconds=time.perf_counter() - started,
                             functions=len(module.functions), concordance_findings=0)
        for backend in backends:
            event("phase", message=f"Building {backend.upper()} native library")
            if backend == "c":
                from src.compiler.ssa_c_backend import emit_ssa_to_c
                artifact = emit_ssa_to_c(module, self.export)
                if not artifact.complete:
                    raise RuntimeError(str(artifact.shortfalls))
                artifact.compile(self.directory / "c")
                self.backends[backend] = artifact.prepare_execution
            else:
                from src.compiler.ssa_llvm_backend import (
                    emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
                emitted = emit_ssa_function_to_llvm(module, self.export)
                if emitted.shortfalls:
                    raise RuntimeError(str(emitted.shortfalls))
                artifact = compile_artifact(emitted, directory=self.directory / "llvm")
                self.backends[backend] = lambda feeds, artifact=artifact: prepare_artifact_execution(artifact, feeds)
        if compiler_toolchain_fingerprint()["sha256"] != self.manifest["compiler_sha256"]:
            raise RuntimeError("Compiler sources changed during this build; compile again on a stable checkout")
        self.manifest["status"] = "ready"
        self.save()
        event("ready", **{key: self.manifest[key] for key in
                          ("compiler_commit", "functions", "lowering_seconds", "concordance_findings")})

    def save(self):
        temporary = self.directory / "run.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "run.json")

    def run(self, rounds, method, collision):
        if not 0 <= rounds <= 3000 or method not in range(4) or collision not in (0, 1):
            raise ValueError("Rounds must be 0..3000, method 0..3, collision 0 or 1")
        python_times = []
        for _ in range(3):
            started = time.perf_counter()
            expected = self.reference(rounds) if self.profile == "manager" else self.reference(rounds, method, collision)
            python_times.append(time.perf_counter() - started)
        if self.profile == "manager":
            if expected != 3 * rounds * rounds + 7000172 * rounds:
                raise AssertionError("Python manager violates supplied golden law")
        else:
            check_result(expected, expected, rounds, collision)
        feeds = {self.parameters[name]: self.np.asarray([value], dtype=self.np.int64)
                 for name, value in (("rounds", rounds), ("method", method), ("collision", collision))
                 if name in self.parameters}
        for backend, prepare in self.backends.items():
            # Each preparation allocates a fresh invocation's context using
            # the compiler's existing runtime. Warmup is never timed.
            prepare(feeds).run()
            native_times, total_times = [], []
            for _ in range(3):
                whole_started = time.perf_counter()
                execution = prepare(feeds)
                started = time.perf_counter()
                execution.run()
                native_times.append(time.perf_counter() - started)
                total_times.append(time.perf_counter() - whole_started)
                actual = tuple(int(execution.buffers[value].reshape(-1)[0]) for value in self.outputs)
                if self.profile == "manager":
                    metrics = check_manager_result(actual[0], expected, rounds)
                else:
                    metrics = check_result(actual, expected, rounds, collision)
            result = {"backend": backend, "rounds": rounds, "method": method, "collision": collision,
                      "metrics": metrics, "python_seconds": statistics.median(python_times),
                      "native_seconds": statistics.median(native_times),
                      "native_with_workspace_seconds": statistics.median(total_times),
                      "repetitions": 3, "exact_match": True}
            result["measurement"] = "native-checked invariants" if self.profile == "manager" else "exported native metrics"
            self.manifest["frames"].append(result)
            self.save()
            event("result", **result)


def worker(backends, verify=False, methods=(0, 1, 2, 3), verify_collisions=(0, 1), verify_rounds=(0, 1, 3)):
    # Retain construction diagnostics even if the compiler refuses before
    # __init__ can return a ready native core.
    core = NativeCore.__new__(NativeCore)
    try:
        core.__init__(backends)
        alternatives = {}
        refused_products = {}

        def exercise(rounds, method, collision):
            if method == 0 and collision == 0:
                core.run(rounds, method, collision)
                return True
            profile = "interning" if method < 2 else "queries"
            if profile not in alternatives and profile not in refused_products:
                candidate = NativeCore.__new__(NativeCore)
                try:
                    candidate.__init__(backends, profile=profile)
                    alternatives[profile] = candidate
                except Exception as error:
                    traceback.print_exc()
                    refused_products[profile] = f"{type(error).__name__}: {error}"
                    if hasattr(candidate, "manifest"):
                        candidate.manifest.update(status="failed", error=refused_products[profile])
                        candidate.save()
            if profile in refused_products:
                failure = {"method": method, "message": refused_products[profile]}
                if failure not in core.manifest["capability_failures"]:
                    core.manifest["capability_failures"].append(failure)
                    core.save()
                event("method_failure", **failure)
                return False
            alternatives[profile].run(rounds, method, collision)
            return True

        if verify:
            refused = False
            for collision in verify_collisions:
                for method in methods:
                    for rounds in verify_rounds:
                        if not exercise(rounds, method, collision):
                            refused = True
                            break
            count = len(core.manifest["frames"]) + sum(len(item.manifest["frames"]) for item in alternatives.values())
            core.manifest.update(status="partial" if refused else "verified", verified_frames=count)
            core.save()
            event("partial" if refused else "verified", frames=count, directory=str(core.directory))
            return 1 if refused else 0
        else:
            for line in sys.stdin:
                request = json.loads(line)
                exercise(int(request["rounds"]), int(request["method"]), int(request["collision"]))
    except Exception as error:
        if hasattr(core, "manifest"):
            core.manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            core.save()
        traceback.print_exc()
        event("failure", message=f"{type(error).__name__}: {error}")
        return 1
    return 0


def window(backends, observe_run=None):
    import tkinter as tk
    from tkinter import ttk
    app = tk.Tk()
    app.title("String Core · live native workbench")
    app.geometry("1240x840")
    app.configure(bg="#101925")
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#101925")
    style.configure("TLabel", background="#101925", foreground="#dbe9f4", font=("Segoe UI", 11))
    style.configure("Treeview", background="#162436", fieldbackground="#162436",
                    foreground="#dbe9f4", rowheight=25, font=("Segoe UI", 10))
    style.configure("Treeview.Heading", background="#22364b", foreground="#dbe9f4")
    style.map("Treeview", background=[("selected", "#31567b")])
    outer = ttk.Frame(app, padding=22)
    outer.pack(fill="both", expand=True)
    ttk.Label(outer, text="STRING CORE  /  NATIVE WORKBENCH", font=("Segoe UI", 21, "bold")).pack(anchor="w")
    ttk.Label(outer, text="Real core source → concordance → native library → exact Python comparison").pack(anchor="w", pady=(4, 12))
    ttk.Label(outer, text="New sequences mint tokens. Repeats reuse tokens. Forced collisions must keep the same answers.",
              font=("Segoe UI", 10)).pack(anchor="w")
    state = tk.StringVar(value="Choose Compile whole core to build the real native libraries.")
    ttk.Label(outer, textvariable=state, wraplength=1040).pack(anchor="w", pady=6)
    controls = ttk.Frame(outer)
    controls.pack(fill="x", pady=12)
    choice = ttk.Combobox(controls, values=METHODS, state="readonly", width=31)
    choice.current(0)
    choice.pack(side="left")
    rounds = tk.IntVar(value=32)
    ttk.Label(controls, text="  Target rounds ").pack(side="left")
    ttk.Spinbox(controls, from_=1, to=3000, textvariable=rounds, width=6).pack(side="left")
    collisions = tk.BooleanVar(value=False)
    ttk.Checkbutton(controls, text="Force hash collisions", variable=collisions).pack(side="left", padx=12)
    ttk.Label(outer, text="Each frame replays a larger history in a NEW native context. Persistence is inside a frame.",
              font=("Segoe UI", 10)).pack(anchor="w")
    summary = tk.StringVar(value="Waiting for a compiled artifact. No Python fallback.")
    ttk.Label(outer, textvariable=summary, font=("Consolas", 12), wraplength=1040).pack(anchor="w", pady=12)
    chart = tk.Canvas(outer, height=190, bg="#162436", highlightthickness=0)
    chart.pack(fill="x", pady=8)
    columns = ("backend", "rounds", "rows", "atoms", "checks", "token", "python", "native", "workspace", "result")
    table = ttk.Treeview(outer, columns=columns, show="headings", height=7)
    table.tag_configure("failed", foreground="#ff9c9c")
    table.tag_configure("matched", foreground="#65d8b0")
    labels = ("Backend", "Rounds", "Rows", "Atoms", "Checks", "Last token", "Python ms", "Native ms", "+ workspace ms", "Exact")
    for column, label in zip(columns, labels):
        table.heading(column, text=label)
        table.column(column, width=100, anchor="e")
    table.pack(fill="x", pady=8)
    log = tk.Text(outer, height=8, bg="#0b111b", fg="#94aec5", font=("Consolas", 9), wrap="word")
    log.pack(fill="both", expand=True)
    messages = queue.Queue()
    history = deque(maxlen=100)
    pending = deque()
    ready = False
    busy = False
    process = None
    observed_frames = 0
    observed_state = None

    def pump(child):
        for line in child.stdout:
            print(line, end="", flush=True)
            if line.startswith(PREFIX):
                messages.put(json.loads(line[len(PREFIX):]))
        messages.put({"kind": "exit", "code": child.wait()})

    def compile_core():
        nonlocal process, ready, observe_run
        if process is not None and process.poll() is None:
            return
        ready = False
        observe_run = None
        run_button.configure(state="normal")
        all_button.configure(state="normal")
        state.set("Starting canonical source compiler…")
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--worker", "--backends", ",".join(backends),
                                    "--compiler-root", str(TURING)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        threading.Thread(target=pump, args=(process,), daemon=True).start()

    ttk.Button(controls, text="Compile whole core", command=compile_core).pack(side="left", padx=5)

    def send_next():
        nonlocal busy
        if pending:
            busy = True
            request = pending.popleft()
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
        else:
            busy = False

    def start(all_methods=False):
        if not ready or busy:
            return
        try:
            target = rounds.get()
            if not 1 <= target <= 3000:
                raise ValueError("Choose 1..3000 rounds")
        except (ValueError, tk.TclError) as error:
            state.set(str(error))
            return
        history.clear()
        for item in table.get_children():
            table.delete(item)
        for method in (range(4) if all_methods else (choice.current(),)):
            for size in frame_sizes(target):
                pending.append({"rounds": size, "method": method, "collision": int(collisions.get())})
        send_next()

    run_button = ttk.Button(controls, text="Run live", command=start)
    run_button.pack(side="left", padx=5)
    all_button = ttk.Button(controls, text="Try all methods", command=lambda: start(True))
    all_button.pack(side="left", padx=5)
    ttk.Button(controls, text="Stop after frame", command=pending.clear).pack(side="left", padx=5)
    if observe_run is not None:
        run_button.configure(state="disabled")
        all_button.configure(state="disabled")
        state.set(f"Observing real native verification: {observe_run}")

    def draw():
        chart.delete("all")
        width = max(chart.winfo_width(), 400)
        maximum = max([row["metrics"]["atoms"] for row in history] + [1])
        chart.create_text(14, 14, text="NATIVE-CHECKED STORAGE  ·  exact atoms / aggregate rows", anchor="nw", fill="#a9c4da")
        for key, color in (("atoms", "#60d8c0"), ("rows", "#70a8ff")):
            points = []
            for i, row in enumerate(history):
                points.extend((25 + i * (width - 50) / max(1, len(history) - 1),
                               165 - row["metrics"][key] * 120 / maximum))
            if len(points) >= 4:
                chart.create_line(*points, fill=color, width=3)

    def tick():
        nonlocal ready, busy, observed_frames, observed_state
        if observe_run is not None:
            try:
                receipt = json.loads((observe_run / "run.json").read_text(encoding="utf-8"))
                for frame in receipt.get("frames", ())[observed_frames:]:
                    messages.put({"kind": "result", **frame})
                    observed_frames += 1
                status = receipt["status"]
                if status != observed_state:
                    observed_state = status
                    if status == "failed":
                        messages.put({"kind": "failure", "message": receipt["error"]})
                    else:
                        messages.put({"kind": "phase", "message":
                                      f"Observed native verification: {status} · compiler {receipt['compiler_commit'][:12]} · "
                                      f"{observed_frames} exact frames"})
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # The selected run has not published its first receipt yet.
        for _ in range(200):
            try:
                message = messages.get_nowait()
            except queue.Empty:
                break
            kind = message["kind"]
            if kind in ("progress", "phase", "failure"):
                state.set(message["message"])
                log.insert("end", message["message"] + "\n")
                if int(log.index("end-1c").split(".")[0]) > 120:
                    log.delete("1.0", "25.0")
                log.see("end")
            if kind == "ready":
                ready = True
                state.set(f"READY · {message['functions']} compiled functions · 0 concordance findings · {message['compiler_commit'][:12]}")
            if kind == "failure":
                ready = False
                pending.clear()
                summary.set("FAILED — see the actual compiler/runtime error above. No fallback was run.")
            if kind == "method_failure":
                state.set(f"{METHODS[message['method']]} refused: {message['message']}")
                log.insert("end", state.get() + "\n")
                log.see("end")
                summary.set("METHOD REFUSED — the last successful frame remains below. Select manager acceptance to continue.")
                table.insert("", "end", tags=("failed",),
                             values=("compiler", "—", "—", "—", "—", "—", "—", "—", "—", "REFUSED"))
                table.yview_moveto(1)
                # Skip the remaining sizes of this refused method, but let
                # Try all methods expose the other capabilities as requested.
                remaining = [request for request in pending if request["method"] != message["method"]]
                pending.clear()
                pending.extend(remaining)
                app.after(350, send_next)
            if kind == "result":
                metrics = message["metrics"]
                history.append(message)
                speed = message["python_seconds"] / max(message["native_with_workspace_seconds"], 1e-12)
                summary.set(f"{METHODS[message['method']]} · {message['rounds']} rounds · epoch {metrics['epoch']} · "
                            f"{metrics['checks']} checks · exact MATCH\n"
                            f"{message.get('measurement', 'exported native metrics')} · checksum {metrics['checksum']} · "
                            f"Python / native incl. workspace: {speed:.2f}× (median of 3)")
                table.insert("", "end", tags=("matched",), values=(message["backend"], message["rounds"], metrics["rows"], metrics["atoms"],
                             metrics["checks"], metrics["last_token"], f"{message['python_seconds'] * 1000:.3f}",
                             f"{message['native_seconds'] * 1000:.3f}",
                             f"{message['native_with_workspace_seconds'] * 1000:.3f}", "MATCH"))
                table.yview_moveto(1)
                draw()
                if message["backend"] == backends[-1]:
                    app.after(350, send_next)
            if kind == "exit":
                ready = False
                busy = False
                state.set(f"Native worker exited: {message['code']}. Restart the demo to compile again.")
        app.after(80, tick)

    def close():
        if process is not None and process.poll() is None:
            process.terminate()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", close)
    chart.bind("<Configure>", lambda _event: draw())
    tick()
    app.mainloop()


def main():
    global TURING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="c,llvm")
    parser.add_argument("--verify", action="store_true", help="compile and execute all method/collision cases without a window")
    parser.add_argument("--compiler-root", type=Path, default=TURING,
                        help="Turing checkout to use (the exact commit is displayed and recorded)")
    parser.add_argument("--observe-run", type=Path,
                        help="watch the actual receipts of an ongoing --verify run without compiling again")
    parser.add_argument("--methods", default="0,1,2,3", help="headless verification method numbers")
    parser.add_argument("--collision-modes", default="0,1", help="headless fingerprint modes (0 normal, 1 forced collisions)")
    parser.add_argument("--verify-rounds", default="0,1,3", help="headless workload sizes")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    TURING = args.compiler_root.resolve()
    sys.path.insert(0, str(TURING))
    backends = tuple(dict.fromkeys(args.backends.split(",")))
    if not backends or set(backends) - {"c", "llvm"}:
        parser.error("--backends accepts c,llvm")
    methods = tuple(int(value) for value in args.methods.split(","))
    if not methods or set(methods) - set(range(4)):
        parser.error("--methods accepts 0,1,2,3")
    collision_modes = tuple(int(value) for value in args.collision_modes.split(","))
    if not collision_modes or set(collision_modes) - {0, 1}:
        parser.error("--collision-modes accepts 0,1")
    sizes = tuple(int(value) for value in args.verify_rounds.split(","))
    if not sizes or any(value < 0 or value > 3000 for value in sizes):
        parser.error("--verify-rounds accepts sizes 0..3000")
    return worker(backends, args.verify, methods, collision_modes, sizes) if args.worker or args.verify else window(backends, args.observe_run)


if __name__ == "__main__":
    raise SystemExit(main())
