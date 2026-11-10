"""Trap every post that writes value 148 of repeated_manager, then run the audit.

Run from string_core/tools:   python trap148.py
Writes trap148.txt next to this script (the audit's own output still prints).
"""
import os
import runpy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
TURING_ROOT = PROJECT_ROOT.parent / "turing"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TURING_ROOT))

OUT = open(HERE / "trap148.txt", "w", encoding="utf-8")
WANT = 148
PAGES = {"ssa_value", "value_type_abi", "value_shape", "ssa_shape_materialization",
         "sequence_contract_concordance", "source_span"}

from src.compiler import identity_concordance as ic

original = ic.IdentityBook.post


def post(self, page, row, *args, **kwargs):
    result = original(self, page, row, *args, **kwargs)
    try:
        name = getattr(page, "name", page)
        flat = repr(row)
        if (name in PAGES and "repeated_manager" in flat
                and isinstance(row, tuple) and WANT in _ints(row)):
            fact = args[0] if args else kwargs.get("fact")
            OUT.write(f"POST {name} row={flat[:300]}\n  fact={str(fact)[:600]}\n"
                      f"  stage={kwargs.get('stage')} mode={kwargs.get('mode')}\n")
            for f in traceback.extract_stack()[-14:-1]:
                OUT.write(f"    {Path(f.filename).name}:{f.lineno} {f.name}\n")
            OUT.flush()
    except Exception as error:  # the trap must never change the run
        OUT.write(f"TRAP-ERROR {error!r}\n")
    return result


def _ints(value):
    if isinstance(value, bool):
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, tuple):
        return tuple(i for item in value for i in _ints(item))
    return ()


ic.IdentityBook.post = post
runpy.run_path(str(HERE / "audit_turing_whole_core.py"), run_name="__main__")
