"""What already exists, in one grep-able place.

Five times in one working session I built something this project already
had: pistons and connecting rods (cylinder_ports builds them), a
procedural header designer (exhaust_header), pipe routing
(fluid_routing), intake runner tuning (engines.IntakeSystem) and a
supercharger model (engine_cycle_sim._step_forced_induction). Each time
the prior art was better than the replacement, and each time I found it
only after writing the duplicate.

The cause is not carelessness in the moment. It is that "does this
already exist?" has no cheap answer in a hundred-file project. Grepping
for a word I guessed finds nothing when the author chose a different
word -- I searched for "piston" in crank_mesh and concluded there were
no pistons, when they were in cylinder_ports under the same name I was
searching for, in a file I had not thought to look in.

So this builds the answer instead of relying on recall: every public
class and function in the project, with the first line of its docstring,
in one text blob that can be searched in a single call. A term that
appears nowhere in it is genuinely new. A term that appears is a file to
go read before writing anything.

WHAT MAKES IT WORTH TRUSTING is that it is generated, not maintained.
A hand-written inventory goes stale the first time someone adds a module
and forgets to update it, which makes it worse than nothing -- it would
say "no prior art" with false authority. This reads the source every
time it is asked.

    python capability_index.py piston rod          # does this exist?
    python capability_index.py --build             # dump the whole index
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"__pycache__", ".git", "frames", "tests"}


def _first_line(node) -> str:
    doc = ast.get_docstring(node) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def module_entries(path: str) -> list:
    """Public classes, functions and module-level registries in one file."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    name = os.path.relpath(path, HERE).replace("\\", "/")
    out = [(name, "module", name, _first_line(tree))]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            out.append((name, kind, node.name, _first_line(node)))
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and not sub.name.startswith("_"):
                        out.append((name, "method", f"{node.name}.{sub.name}",
                                    _first_line(sub)))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # module-level registries and constant tables: the places
            # this project keeps its declared data, and exactly what a
            # "does X exist" question is usually really asking about
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 3:
                    out.append((name, "table", t.id, ""))
    return out


def build(root: str = HERE) -> list:
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                rows.extend(module_entries(os.path.join(dirpath, fn)))
    return rows


def search(*terms: str, root: str = HERE) -> list:
    """Every entry matching ANY of these terms, in name or summary.

    Deliberately OR rather than AND, and deliberately substring rather
    than whole-word: the failure mode this exists to prevent is a term
    that ALMOST matches, so it should over-report. A long list of
    near-misses costs one read; a confident empty result costs a
    duplicate module."""
    needles = [t.lower() for t in terms if t]
    if not needles:
        return []
    hits = []
    for mod, kind, name, doc in build(root):
        hay = f"{mod} {name} {doc}".lower()
        if any(n in hay for n in needles):
            hits.append((mod, kind, name, doc))
    return hits


def report(*terms: str, limit: int = 40) -> str:
    hits = search(*terms)
    if not hits:
        return (f"nothing matches {terms!r} -- this looks genuinely new. "
                "Note that a different author's word for the same idea will "
                "not match, so try a synonym before concluding.")
    lines = [f"{len(hits)} existing things match {terms!r}:"]
    for mod, kind, name, doc in hits[:limit]:
        lines.append(f"  {mod:26s} {kind:7s} {name:34s} {doc[:60]}")
    if len(hits) > limit:
        lines.append(f"  ... and {len(hits) - limit} more")
    return "\n".join(lines)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--build"]
    if "--build" in sys.argv[1:] or not args:
        for mod, kind, name, doc in build():
            print(f"{mod}\t{kind}\t{name}\t{doc}")
    else:
        print(report(*args))
