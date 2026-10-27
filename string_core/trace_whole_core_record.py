"""Print the two exact record claims at the whole-core audit frontier.
This is an ignored, read-only diagnostic. Run it from string_core instead of
audit_turing_whole_core.py when tracing the current receiver-view loss.
"""

from collections.abc import Mapping
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "tools" / "audit_turing_whole_core.py"


def field_receipt(record):
    if not isinstance(record, Mapping):
        return record
    fields = record.get("fields") or ()
    if isinstance(fields, Mapping):
        return fields.get("exact_begin")
    return tuple(
        field for field in fields
        if isinstance(field, Mapping) and field.get("name") == "exact_begin"
    )


def trace_specialization(frame, event, unused_arg):
    if event == "line" and frame.f_lineno == 28480:
        values = frame.f_locals
        if values.get("parameter") != "self":
            return trace_specialization
        original = values.get("original")
        caller = values.get("caller")
        callee_name = str(original.G.graph.get("function_name", ""))
        caller_name = str(caller.G.graph.get("function_name", ""))
        if "_cover_span" in callee_name and "_intern_validated_span" in caller_name:
            print("RECORD RECEIPTS AT SPECIALIZATION", flush=True)
            print("caller:", caller_name, "callee:", callee_name, flush=True)
            print("callsite:", values.get("node_id"), "source:", values.get("parent"), flush=True)
            print("declared exact_begin:", field_receipt(values.get("declared_record_abi")), flush=True)
            print("callsite exact_begin:", field_receipt(values.get("callsite_record_abi")), flush=True)
    return trace_specialization


def trace_linker(frame, event, arg):
    if event == "exception" and isinstance(arg[1], ValueError) and "exact_begin" in str(arg[1]):
        values = frame.f_locals
        print("RECORD RECEIPTS AT LINKER REFUSAL", flush=True)
        for key in ("caller_symbol", "record_id", "caller_record_id", "member", "field", "caller_field", "caller_sequence"):
            print(f"{key}: {values.get(key)!r}", flush=True)
    return trace_linker


def trace_materialize(frame, event, unused_arg):
    values = frame.f_locals
    symbol = str(values.get("symbol", ""))
    if "_intern_validated_span" not in symbol:
        return trace_materialize
    if event == "call":
        graph = values.get("graph")
        receipt = dict(graph.graph.get("parameter_record_abi") or {}).get("self")
        print("RECORD RECEIPT AT MATERIALIZER ENTRY", flush=True)
        print("symbol:", symbol, flush=True)
        print("exact_begin:", field_receipt(receipt), flush=True)
    if event == "line" and frame.f_lineno == 37134 and values.get("field_name") == "exact_begin":
        print("RECORD FIELD AT MATERIALIZATION", flush=True)
        print("field:", values.get("field"), flush=True)
        print("storage:", values.get("storage"), "access:", values.get("access_receipts"), flush=True)
    return trace_materialize


def trace_call(frame, event, unused_arg):
    if event != "call":
        return None
    filename = frame.f_code.co_filename.replace("\\", "/")
    if filename.endswith("/glsl_deployment_strategy.py") and frame.f_code.co_name == "_callsite_specialized_shell_type":
        return trace_specialization
    if filename.endswith("/fortran_c_shell.py") and frame.f_code.co_name == "_linked_caller_member":
        return trace_linker
    if filename.endswith("/fortran_c_shell.py") and frame.f_code.co_name == "materialize_parameter_record_abi":
        return trace_materialize
    return None


sys.settrace(trace_call)
try:
    runpy.run_path(str(AUDIT), run_name="__main__")
finally:
    sys.settrace(None)
