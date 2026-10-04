"""Actual native execution isolation; independent of the production row bank."""
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import sympy as sp

import orbital_collocation as oc
from src.compiler import ssa_llvm_backend
from src.compiler.identity_concordance import concordance_report
from src.compiler.process_graph_autograd import reverse_compile_book


def test_native_rows_share_artifact_but_not_feeds_seeds_or_outputs(tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr(oc, "CACHE_DIRECTORY", tmp_path / "rows")
    x, y = sp.symbols("ownership_x ownership_y")
    modules = []
    emit = ssa_llvm_backend.emit_ssa_function_to_llvm

    def capture(module, *args, **kwargs):
        modules.append(module)
        return emit(module, *args, **kwargs)

    monkeypatch.setattr(ssa_llvm_backend, "emit_ssa_function_to_llvm", capture)
    with reverse_compile_book():
        original = oc.compile_reverse_rows(
            "orbital_planning_execution_ownership", (x * y + x**2, y - x), (x, y))
    receipt = concordance_report(modules[0], limit=2)
    print(receipt, flush=True)
    assert receipt.splitlines()[0].endswith(", 0 finding(s)")
    assert original.artifact.emission is not None
    first, second = replace(original), replace(original)
    assert first.artifact is second.artifact is original.artifact
    ex1, arena1 = first._execution
    ex2, arena2 = second._execution
    assert ex1 is not ex2 and not np.shares_memory(arena1, arena2)
    barrier = threading.Barrier(2)

    def run(rows, offset):
        for index in range(100):
            a, b = offset + index / 8, -offset + index / 16
            feeds = {x.name: a, y.name: b}
            barrier.wait(timeout=5.0)
            values, J = rows.jacobian(feeds)
            expected = np.array((a * b + a**2, b - a))
            np.testing.assert_array_equal(values, expected)
            columns = {x.name: (b + 2 * a, -1.), y.name: (a, 1.)}
            np.testing.assert_array_equal(
                J, np.asarray([columns[name] for name in rows._grad_names]).T)
            np.testing.assert_array_equal(rows.values(feeds), expected)
        return index + 1

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = [workers.submit(run, first, 1.0),
                   workers.submit(run, second, 20.0)]
        assert [result.result(timeout=15.0) for result in results] == [100, 100]
    print("two native executions / one artifact: 200 distinct feed/seed "
          "Jacobians and forward calls; disjoint scalar arenas", flush=True)
