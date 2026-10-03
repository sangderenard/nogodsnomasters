"""Cache provenance checks; no native build or execution required."""
import json
from types import SimpleNamespace

import orbital_collocation as oc
from src.compiler.native_law_kernels import route_compiler_record


def _stored_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "CACHE_DIRECTORY", tmp_path / "cache")
    library = tmp_path / "rows.dll"
    library.write_bytes(b"cache serialization fixture")
    artifact = SimpleNamespace(
        library_path=library, name="rows", buffer_order=(1, 2, 3, 4),
        buffer_shapes=((),) * 4, buffer_dtypes=("float64",) * 4,
        extent_order=())
    rows = oc.ReverseRows("rows", artifact, {"x": 1}, (2,), (3,),
                          {"x": 4}, compiler=route_compiler_record())
    oc._store_rows(rows, "key")
    return rows, oc.CACHE_DIRECTORY / "key" / "contract.json"


def test_cache_round_trip_keeps_the_build_compiler(tmp_path, monkeypatch):
    rows, _manifest = _stored_rows(tmp_path, monkeypatch)
    loaded = oc._load_cached_rows("rows", "key")
    assert loaded.compiler == rows.compiler
    assert loaded.input_ids == rows.input_ids
    assert loaded.gradient_ids == rows.gradient_ids
    assert (oc.CACHE_DIRECTORY / "key" / "rows.book.log").is_file()


def test_legacy_cache_requires_a_rebuild_with_a_book_receipt(tmp_path, monkeypatch):
    _rows, manifest = _stored_rows(tmp_path, monkeypatch)
    record = json.loads(manifest.read_text())
    del record["compiler"]
    manifest.write_text(json.dumps(record))
    assert oc._load_cached_rows("rows", "key") is None
    receipt = (manifest.parent / "rows.stale.book.log").read_text()
    assert "<no compiler record>" in receipt
    assert "rebuild_required" in receipt


def test_cache_rejects_changed_compiler_sources(tmp_path, monkeypatch):
    _rows, manifest = _stored_rows(tmp_path, monkeypatch)
    record = json.loads(manifest.read_text())
    record["compiler"] = {"digest": "old", "modules": [
        ["src.compiler.changed", "src/compiler/missing_cache_fixture.py", "old"]]}
    manifest.write_text(json.dumps(record))
    assert oc._load_cached_rows("rows", "key") is None
    assert "src.compiler.changed" in (
        manifest.parent / "rows.stale.book.log").read_text()
