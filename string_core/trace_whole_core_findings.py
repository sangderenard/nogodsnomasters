"""Print every final detector finding from the real whole-core audit."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "turing"))
sys.path.insert(0, str(ROOT))

from src.compiler import fortran_c_shell
from src.compiler.identity_concordance import CorrelationTable

lower = fortran_c_shell.lower_ast_source_to_ssa


def lower_with_findings(*args, **kwargs):
    result = lower(*args, **kwargs)
    module = result[0]
    findings = tuple(CorrelationTable.build(module).findings(module))
    print(f"WHOLE-CORE DETECTOR: {len(findings)} finding(s)", flush=True)
    for number, finding in enumerate(findings, 1):
        print(
            f"{number}. [{finding.kind}] {finding.function} "
            f"value {finding.value_id}: {finding.detail}",
            flush=True,
        )
    return result


fortran_c_shell.lower_ast_source_to_ssa = lower_with_findings
try:
    runpy.run_path(str(ROOT / "tools" / "audit_turing_whole_core.py"), run_name="__main__")
finally:
    fortran_c_shell.lower_ast_source_to_ssa = lower
