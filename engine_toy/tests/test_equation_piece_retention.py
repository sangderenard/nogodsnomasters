"""Real native cache loads keep audit archives off the game execution path."""

import hashlib

import numpy as np
import pytest
import sympy as sp

import honorary_engine_equation_catalogue as catalogue
from src.compiler.native_law_kernels import LLVMPiece, piece_staleness


@pytest.mark.parametrize("runtime_first", (True, False))
def test_runtime_cache_preserves_archive_and_native_columns(tmp_path, monkeypatch,
                                                          runtime_first):
    monkeypatch.setattr(catalogue, "_PIECE_MEMORY_CACHE", {})
    piece_id = f"retention_native_{int(runtime_first)}"
    x = sp.Symbol("x")
    equations = (sp.Eq(sp.Symbol("y_next"), 2 * x + 1, evaluate=False),
                 sp.Eq(sp.Symbol("constant_next"), 7, evaluate=False))
    options = dict(batch=3, cache_dir=tmp_path)
    initial = catalogue.equation_piece(
        piece_id, equations, retain_compilation=not runtime_first, **options)
    archive = next(tmp_path.rglob(f"{piece_id}.piece"))
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    runtime = catalogue.equation_piece(
        piece_id, equations, retain_compilation=False, **options)
    runtime_path = archive.with_name(f"{piece_id}.runtime.piece")
    assert runtime_path.is_file()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_digest
    assert runtime_path.stat().st_size < archive.stat().st_size
    assert runtime.compiler == initial.compiler
    assert not piece_staleness(runtime)
    assert runtime.module is None and runtime.source is None
    assert runtime.outputs is None and runtime.artifact.emission is None
    assert runtime.entry == initial.entry
    if not runtime_first:
        assert initial.module is not None and initial.artifact.emission is not None

    # Loading the execution file must not deserialize the full book again.
    # A repeated call also exercises the persistent native column binding.
    catalogue._PIECE_MEMORY_CACHE.clear()
    reloaded = catalogue.equation_piece(
        piece_id, equations, retain_compilation=False, **options)
    values = np.array([-2.0, 0.0, 3.5])
    destination = np.zeros(3)
    reloaded.instantiate({"x": values}, outputs={"y_next": destination})
    for _ in range(2):
        result = dict(zip(reloaded.output_names, reloaded(values)))
        np.testing.assert_array_equal(result["y_next"], 2 * values + 1)
        np.testing.assert_array_equal(result["constant_next"], np.full(3, 7.0))
        assert result["y_next"] is destination
        values += 0.5

    full = LLVMPiece.load(archive)
    assert full.module is not None and full.artifact.emission is not None
    assert full.compiler == reloaded.compiler
    assert catalogue.equation_piece(
        piece_id, equations, retain_compilation=True, **options).module is not None
    print(f"{piece_id}: full={archive.stat().st_size} bytes, "
          f"runtime={runtime_path.stat().st_size} bytes", flush=True)
