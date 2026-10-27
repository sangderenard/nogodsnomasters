"""Run the authored whole-core audit against the isolated Turing worktree."""

from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
source_path = project_root / "tools" / "audit_turing_whole_core.py"
source = source_path.read_text(encoding="utf-8")
original = 'TURING_ROOT = WORKSPACE_ROOT / "turing"'
replacement = 'TURING_ROOT = WORKSPACE_ROOT / "turing-record-view"'
assert source.count(original) == 1
source = source.replace(original, replacement, 1)
exec(compile(source, str(source_path), "exec"), {
    "__file__": str(source_path),
    "__name__": "__main__",
})
