"""Native LLVM-compiled one-pole IIR filter for the audio synth's hot
per-sample loops (see engine_sound.py's noise-coloring/bleed loops, which
run inside the realtime PortAudio callback -- toy_shared.make_audio_callback
-- where a pure-Python per-sample for loop is a genuine underrun risk).

Routed through the repository's real, currently-used Python-to-LLVM
deployment pipeline: lower_ast_source_to_ssa -> emit_ssa_function_to_llvm ->
compile_artifact -> prepare_artifact_execution (fortran_c_shell.py /
ssa_llvm_backend.py) -- the same path repository_ssa_dispatch.py's
production dispatch and the compiled-eigh precedent use. Not the
AbstractTensor tape path, not compile_ast_aot (both wrong for this).

Compiles once, lazily, at first call, and caches the native artifact for
the process lifetime. Falls back to the exact same recurrence run in plain
Python if compilation isn't available, so audio still plays either way.
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

_LEAKY_INTEGRATOR_SOURCE = """
def leaky_integrator(x, n, alpha, state, y):
    prev = state[0]
    for k in range(n):
        prev = prev + alpha * (x[k] - prev)
        y[k] = prev
    state[0] = prev
    return y
"""

_native_ready: bool | None = None
_native_parts = None


def _build_native() -> None:
    global _native_ready, _native_parts
    from compile_contract import contract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm,
        compile_artifact,
        prepare_artifact_execution,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _outputs, _exports = lower_ast_source_to_ssa(
            _LEAKY_INTEGRATOR_SOURCE, "leaky_integrator", name="engine_toy_audio",
            extraction_contract=contract(),
        )
    qualified = "engine_toy_audio__leaky_integrator"
    function = module.functions[qualified]
    parameters = dict(function.metadata["parameter_names"])

    artifact = emit_ssa_function_to_llvm(module, qualified)
    if artifact.shortfalls:
        raise RuntimeError(f"leaky_integrator shortfalls: {artifact.shortfalls}")
    native = compile_artifact(
        artifact, directory=Path(tempfile.mkdtemp()) / "engine_toy_audio"
    )
    _native_parts = (parameters, function, native, prepare_artifact_execution)
    _native_ready = True


def _ensure_native() -> bool:
    global _native_ready
    if _native_ready is None:
        try:
            _build_native()
        except Exception:
            _native_ready = False
    return bool(_native_ready)


def leaky_integrator(x: np.ndarray, alpha: float, state: float) -> tuple[np.ndarray, float]:
    """prev += alpha * (x[k] - prev) per sample; returns (filtered, new_state).
    Native-compiled when available, an exact Python fallback otherwise."""
    if _ensure_native():
        parameters, function, native, prepare_artifact_execution = _native_parts
        x64 = np.ascontiguousarray(x, dtype=np.float64)
        n = len(x64)
        y64 = np.zeros(n, dtype=np.float64)
        state_arr = np.array([state], dtype=np.float64)
        feed = {
            parameters["x"]: x64,
            parameters["n"]: np.array([n]),
            parameters["alpha"]: np.array([alpha], dtype=np.float64),
            parameters["state"]: state_arr,
            parameters["y"]: y64,
        }
        for formal in function.args:
            identifier = int(formal.id)
            if identifier not in feed:
                feed[identifier] = (
                    np.array([0]) if formal.dtype == "int" else np.zeros(1)
                )
        execution = prepare_artifact_execution(native, feed)
        execution.run()
        y_out = np.asarray(execution.buffers[parameters["y"]]).copy()
        new_state = float(np.asarray(execution.buffers[parameters["state"]])[0])
        return y_out, new_state

    prev = state
    out = np.empty_like(x, dtype=np.float64)
    for k in range(len(x)):
        prev += alpha * (x[k] - prev)
        out[k] = prev
    return out, prev
