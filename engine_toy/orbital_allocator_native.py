"""Fixed-shape native box-constrained least squares for actuator allocation.

The numerical source is lowered by the sanctioned source compiler. Each
active-set Newton step uses the repository's canonical retained-loop solve
source, unchanged except for its declared dimensions. No SciPy or eager
linear-solve fallback is involved. Nonlinear wrench laws and physical
inequalities remain the caller's responsibility: this solves a bounded
linearized Gauss-Newton subproblem.
"""
from __future__ import annotations

import functools
import hashlib
import json
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))


_PREFIX = """
def bounded_solve(operator, target, lower, upper, initial, options, linear_term, x, hessian, drive, gradient, blocked, matrix, rhs, current_row, pivot_row_values, current_rhs, pivot_rhs, pivot_rows, pivot_magnitudes, telemetry):
    for initialize_column in range(__N__):
        x[initialize_column] = min(upper[initialize_column], max(lower[initialize_column], initial[initialize_column]))
        drive_total = -linear_term[initialize_column]
        for drive_row in range(__M__):
            drive_total = drive_total + operator[drive_row * __N__ + initialize_column] * target[drive_row]
        drive[initialize_column] = drive_total
        for hessian_column in range(__N__):
            hessian_total = options[0] if initialize_column == hessian_column else 0.0
            for hessian_row in range(__M__):
                hessian_total = hessian_total + operator[hessian_row * __N__ + initialize_column] * operator[hessian_row * __N__ + hessian_column]
            hessian[initialize_column * __N__ + hessian_column] = hessian_total
    telemetry[0] = 0.0
    for iteration in range(__ITERATIONS__):
        projected_norm = 0.0
        for gradient_column in range(__N__):
            gradient_total = -drive[gradient_column]
            for gradient_inner in range(__N__):
                gradient_total = gradient_total + hessian[gradient_column * __N__ + gradient_inner] * x[gradient_inner]
            gradient[gradient_column] = gradient_total
            at_lower = x[gradient_column] <= lower[gradient_column] and gradient_total >= 0.0
            at_upper = x[gradient_column] >= upper[gradient_column] and gradient_total <= 0.0
            is_blocked = lower[gradient_column] == upper[gradient_column] or at_lower or at_upper
            blocked[gradient_column] = 1.0 if is_blocked else 0.0
            projected_component = 0.0 if is_blocked else abs(gradient_total)
            projected_norm = max(projected_norm, projected_component)
        frozen = projected_norm <= options[1]
        telemetry[0] = telemetry[0] if frozen else (iteration + 1) * 1.0
        for reduced_row in range(__N__):
            row_blocked = frozen or blocked[reduced_row] > 0.0
            rhs[reduced_row] = 0.0 if row_blocked else -gradient[reduced_row]
            for reduced_column in range(__N__):
                column_blocked = frozen or blocked[reduced_column] > 0.0
                unit_entry = 1.0 if reduced_row == reduced_column else 0.0
                matrix[reduced_row * __N__ + reduced_column] = unit_entry if row_blocked or column_blocked else hessian[reduced_row * __N__ + reduced_column]
"""

_SUFFIX = """
        step_fraction = 1.0
        for fraction_column in range(__N__):
            positive_fraction = (upper[fraction_column] - x[fraction_column]) / rhs[fraction_column] if rhs[fraction_column] > 0.0 else 1.0
            negative_fraction = (lower[fraction_column] - x[fraction_column]) / rhs[fraction_column] if rhs[fraction_column] < 0.0 else 1.0
            step_fraction = min(step_fraction, positive_fraction, negative_fraction)
        # A released bound can oppose the coupled Newton direction. Its
        # zero feasible step uses the projected steepest-descent direction;
        # exact quadratic line minimization keeps the same objective.
        use_projected = step_fraction <= 0.0
        for safeguard_column in range(__N__):
            descent_component = -gradient[safeguard_column] if blocked[safeguard_column] == 0.0 else 0.0
            rhs[safeguard_column] = descent_component if use_projected else rhs[safeguard_column]
        descent_product = 0.0
        curvature = 0.0
        step_fraction = 1.0
        for search_column in range(__N__):
            descent_product = descent_product + gradient[search_column] * rhs[search_column]
            curvature_row = 0.0
            for curvature_column in range(__N__):
                curvature_row = curvature_row + hessian[search_column * __N__ + curvature_column] * rhs[curvature_column]
            curvature = curvature + rhs[search_column] * curvature_row
            search_positive = (upper[search_column] - x[search_column]) / rhs[search_column] if rhs[search_column] > 0.0 else 1.0
            search_negative = (lower[search_column] - x[search_column]) / rhs[search_column] if rhs[search_column] < 0.0 else 1.0
            step_fraction = min(step_fraction, search_positive, search_negative)
        minimum_fraction = -descent_product / curvature if curvature > 0.0 else 1.0
        step_fraction = max(0.0, min(step_fraction, minimum_fraction))
        for update_column in range(__N__):
            x[update_column] = min(upper[update_column], max(lower[update_column], x[update_column] + step_fraction * rhs[update_column]))
    final_norm = 0.0
    for final_column in range(__N__):
        final_gradient = -drive[final_column]
        for final_inner in range(__N__):
            final_gradient = final_gradient + hessian[final_column * __N__ + final_inner] * x[final_inner]
        final_lower = x[final_column] <= lower[final_column] and final_gradient >= 0.0
        final_upper = x[final_column] >= upper[final_column] and final_gradient <= 0.0
        final_blocked = lower[final_column] == upper[final_column] or final_lower or final_upper
        final_component = 0.0 if final_blocked else abs(final_gradient)
        final_norm = max(final_norm, final_component)
    telemetry[1] = final_norm
    telemetry[2] = 1.0 if final_norm <= options[1] else 0.0
    return x, telemetry
"""


def allocation_source(rows: int, columns: int, max_iterations: int) -> str:
    """Compose the canonical solve's body inside the bounded Newton loop."""
    from src.common.tensors.linalg_kernels import SOLVE_SOURCE

    # Preserve the canonical algorithm verbatim; only remove its function
    # boundary and final return so its existing scratch ABI nests in this law.
    solve_lines = SOLVE_SOURCE.strip().splitlines()
    solve_body = "\n".join(solve_lines[1:-1])
    source = _PREFIX + textwrap.indent(solve_body, "    ") + "\n" + _SUFFIX
    for token, value in (("__BATCH__", 1), ("__M__", rows),
                         ("__N__", columns), ("__ITERATIONS__", max_iterations)):
        source = source.replace(token, str(value))
    return source


def _shapes(rows: int, columns: int) -> dict[str, tuple[int, ...]]:
    vectors = ("lower", "upper", "initial", "linear_term", "x", "drive",
               "gradient", "blocked", "rhs", "current_row", "pivot_row_values",
               "current_rhs", "pivot_rhs")
    return {
        "operator": (rows * columns,), "target": (rows,), "options": (2,),
        "telemetry": (3,), **{name: (columns,) for name in vectors},
        **{name: (columns * columns,) for name in
           ("hessian", "matrix", "pivot_rows", "pivot_magnitudes")},
    }


@dataclass
class NativeBoundedLeastSquares:
    """One prepared ABI, reused on every call; an instance is not reentrant."""

    rows: int
    columns: int
    max_iterations: int
    artifact: Any
    input_value_ids: dict[str, int]
    compiler: Any = None
    _execution: Any = field(default=None, init=False, repr=False)

    def prepare(self) -> "NativeBoundedLeastSquares":
        from src.compiler.ssa_llvm_backend import prepare_artifact_execution

        if self._execution is None:
            feeds = {self.input_value_ids[name]: np.zeros(shape, np.float64)
                     for name, shape in _shapes(self.rows, self.columns).items()
                     if self.input_value_ids[name] in self.artifact.buffer_order}
            self._execution = prepare_artifact_execution(self.artifact, feeds)
        return self

    def _buffer(self, name):
        return self._execution.buffers[self.input_value_ids[name]]

    @property
    def converged(self) -> bool:
        return bool(self._buffer("telemetry")[2]) if self._execution else False

    @property
    def iterations(self) -> int:
        return int(self._buffer("telemetry")[0]) if self._execution else 0

    @property
    def projected_gradient(self) -> float:
        return float(self._buffer("telemetry")[1]) if self._execution else np.inf

    def solve(self, matrix, rhs, lower, upper, initial=None,
              regularization: float = 1.0e-10, *, linear_term=None,
              tolerance: float = 1.0e-10) -> np.ndarray:
        """Minimize .5||matrix*x-rhs||² + .5*regularization*||x||² + c*x.

        ``linear_term`` is c (zero by default). Bounds may pin a variable.
        A zero regularization requires full column rank; otherwise supply a
        positive damping explicitly. The returned vector aliases prepared
        storage and is overwritten on the next call. Inspect ``converged``
        and ``projected_gradient`` when the iteration cap is reached.
        """
        arrays = {"operator": np.asarray(matrix, np.float64),
                  "target": np.asarray(rhs, np.float64),
                  "lower": np.asarray(lower, np.float64),
                  "upper": np.asarray(upper, np.float64)}
        expected = {"operator": (self.rows, self.columns),
                    "target": (self.rows,), "lower": (self.columns,),
                    "upper": (self.columns,)}
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arrays[name].shape}")
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"{name}: all entries must be finite")
        if np.any(arrays["lower"] > arrays["upper"]):
            raise ValueError("lower bounds exceed upper bounds")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and nonnegative")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        self.prepare()
        for name, value in arrays.items():
            self._buffer(name)[...] = value.reshape(-1)
        for name, value in (("initial", initial), ("linear_term", linear_term)):
            if value is None:
                self._buffer(name).fill(0.0)
            else:
                value = np.asarray(value, np.float64)
                if value.shape != (self.columns,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{name}: expected a finite vector of length {self.columns}")
                self._buffer(name)[...] = value
        self._buffer("options")[...] = regularization, tolerance
        self._execution.run()
        result = self._buffer("x")
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("native bounded solve failed; rank-deficient models require positive regularization")
        return result


def compile_bounded_least_squares(directory: str | Path, *, rows: int,
                                  columns: int, max_iterations: int = 128
                                  ) -> NativeBoundedLeastSquares:
    """Lower, emit, compile and prepare one fixed shape, with visible progress."""
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.native_law_kernels import route_compiler_record
    from src.compiler.ssa_llvm_backend import (
        compile_artifact, emit_ssa_function_to_llvm)

    rows, columns, max_iterations = int(rows), int(columns), int(max_iterations)
    if min(rows, columns, max_iterations) < 1:
        raise ValueError("rows, columns and max_iterations must be positive")
    name = f"orbital_box_lsq_m{rows}_n{columns}_i{max_iterations}"
    shapes = _shapes(rows, columns)
    abi = {"records": {}, "bindings": [], "values": [
        {"function": "bounded_solve", "parameter": parameter,
         "storage": "span", "dtype": "float64", "rank": 1,
         "shape": list(shape),
         "python_type": "src.common.tensors.abstraction.AbstractTensor"}
        for parameter, shape in shapes.items()]}
    policy = ExtractionContract(
        _TURING_ROOT / "extraction_contracts" / "program_extraction.yaml"
    ).with_execution_file(
        _TURING_ROOT / "extraction_contracts" / "vehicle_full_native_execution.yaml"
    ).with_program_abi(abi)
    def report(message):
        print(f"[orbital-allocation] {name}: {message}", flush=True)
    report("lowering fixed-shape active-set solve")
    module, _outputs, _exports = lower_ast_source_to_ssa(
        allocation_source(rows, columns, max_iterations), "bounded_solve",
        name=name, extraction_contract=policy, progress=report)
    qualified = f"{name}__bounded_solve"
    ids = {str(parameter): int(value_id) for parameter, value_id in
           module.functions[qualified].metadata["parameter_names"]}
    report("emitting LLVM")
    artifact = emit_ssa_function_to_llvm(module, qualified, entry_name=name)
    if artifact.shortfalls:
        raise RuntimeError(f"{name}: LLVM shortfalls {artifact.shortfalls!r}")
    report("compiling native library")
    artifact = compile_artifact(artifact, directory=Path(directory))
    solver = NativeBoundedLeastSquares(rows, columns, max_iterations, artifact,
                                      ids, route_compiler_record()).prepare()
    report("prepared reusable buffers")
    return solver


@functools.lru_cache(maxsize=64)
def get_bounded_least_squares(rows: int, columns: int, max_iterations: int = 128
                               ) -> NativeBoundedLeastSquares:
    """Process/disk cache by dimensions, source and the canonical compiler record."""
    from src.compiler.native_law_kernels import (
        PieceCompilerRecord, piece_staleness, post_piece_book)
    from src.compiler.ssa_llvm_backend import LLVMFunctionArtifact

    source = allocation_source(rows, columns, max_iterations)
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]
    directory = Path(tempfile.gettempdir()) / "orbital_allocation_native" / key
    manifest = directory / "contract.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        compiler = PieceCompilerRecord(saved["compiler"]["digest"],
                                       tuple(tuple(r) for r in saved["compiler"]["modules"]))
        artifact = LLVMFunctionArtifact(
            name=saved["entry"], llvm_ir=saved["llvm_ir"],
            buffer_order=tuple(saved["buffer_order"]),
            buffer_shapes=tuple(tuple(r) for r in saved["buffer_shapes"]),
            buffer_dtypes=tuple(saved["buffer_dtypes"]),
            extent_order=tuple(tuple(r) for r in saved["extent_order"]),
            shortfalls=(), library_path=(directory / saved["library"]).resolve())
        solver = NativeBoundedLeastSquares(int(rows), int(columns), int(max_iterations),
                                          artifact, saved["input_ids"], compiler)
        changed = piece_staleness(solver)
        if not changed and artifact.library_path.is_file():
            print(f"[orbital-allocation] loaded {artifact.name}", flush=True)
            return solver.prepare()
        if changed:
            print(f"[orbital-allocation] rebuilding {artifact.name}: compiler changed {changed[:8]}", flush=True)
            post_piece_book(directory, artifact.name, 1, key, stale=changed,
                            stale_record=compiler, decision="rebuild_required")
    solver = compile_bounded_least_squares(directory, rows=rows, columns=columns,
                                           max_iterations=max_iterations)
    a = solver.artifact
    manifest.write_text(json.dumps({
        "entry": a.name, "library": Path(a.library_path).name, "llvm_ir": a.llvm_ir,
        "buffer_order": list(a.buffer_order),
        "buffer_shapes": [list(s) for s in a.buffer_shapes],
        "buffer_dtypes": list(a.buffer_dtypes),
        "extent_order": [list(r) for r in a.extent_order],
        "input_ids": solver.input_value_ids,
        "compiler": {"digest": solver.compiler.digest,
                     "modules": solver.compiler.modules}}), encoding="utf-8")
    post_piece_book(directory, a.name, 1, key, built=solver)
    return solver


bounded_least_squares_solver = get_bounded_least_squares


__all__ = ["NativeBoundedLeastSquares", "compile_bounded_least_squares",
           "get_bounded_least_squares", "bounded_least_squares_solver",
           "allocation_source"]
