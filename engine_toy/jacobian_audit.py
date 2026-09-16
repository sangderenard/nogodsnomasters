"""Constants that change nothing: the parameters with zero Jacobian.

A declared constant is a claim that something depends on it. When the
partial derivative of every output with respect to that constant is
EXACTLY zero, the claim is false, and it is false in one of several
ways that are worth telling apart:

  DEAD          nothing reads it. It survives because deleting things
                feels riskier than leaving them, and each one makes the
                next reader believe the model is richer than it is.

  IMPORT-BAKED  read once at module import to compute something else,
                so changing it afterwards moves nothing. Not a defect --
                but it means the constant cannot be tuned, swept, or
                fitted at runtime, and anyone who tries will conclude it
                does not matter.

  MASKED        read, but downstream of a clamp, a min/max, or a branch
                that never selects it at the operating points tested. A
                masked constant is the dangerous kind: it does nothing
                HERE and may do everything somewhere else, so a zero
                reading is a statement about the probe, not the model.

  UNPROBED      genuinely used, on a path this audit does not exercise.
                A zero here is the audit's fault, and the honest
                response is to widen the probe, not to delete anything.

WHY EXACTLY ZERO IS THE INTERESTING THRESHOLD, and not "small". A tiny
sensitivity is ordinary -- most constants matter a little. A sensitivity
of precisely 0.0, in floating point, after a 1% perturbation, cannot
happen by accident to a number that is genuinely multiplied into an
answer. It means the number never reached the answer at all.

This was worth building because the measurement has already been made
once, narrowly, and found something: an identifiability check on
derived_torque's 27 scalars reported two with exactly 0.0% sensitivity.
Two constants, declared and documented, moving nothing. That check
covered one module. This covers the project.

HOW IT PROBES. Perturbation, not symbolic differentiation: set the
constant to 1.01x, evaluate a battery of pure outputs, restore it, and
take the largest relative change across the battery. The battery is
deliberately made of functions that need no simulator -- a full sim run
per constant would make a project-wide sweep take hours, and the
functions that carry the physics are pure anyway.
"""
from __future__ import annotations

import importlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

#: Relative perturbation. Large enough to clear floating-point noise in
#: any real dependence, small enough to stay inside the regime a
#: constant was chosen for.
PERTURBATION = 0.01

#: Constants whose zero reading is expected and uninteresting: unit
#: conversions and physical definitions that the battery does not touch.
SKIP_PREFIXES = ("TEST_", "DEBUG_", "DEFAULT_LOG")


def _probe_engines():
    import engines
    wanted = ("amc-258-jeep-i6", "cat-c18-industrial-diesel",
              "mazda-b6ze-miata-1990", "25cc-two-stroke-trimmer")
    out = []
    for w in wanted:
        try:
            out.append(engines.get(w))
        except Exception:
            continue
    return out or list(engines.BY_IDENTITY.values())[:3]


def battery(probes) -> list:
    """Every number this audit watches.

    Pure functions only, spanning the physics that actually carries the
    model: the torque fraction the sim integrates, the independent
    derivation, combustion completeness, the harm chain, absorber
    capacity and cycle efficiency. A constant that moves none of these
    moves nothing anyone reads."""
    vals: list = []
    import engine_sim
    import derived_torque
    import combustion_efficiency
    import engine_harm
    import dyno_brakes
    import thermo_cycles
    for e in probes:
        arch = e.architecture
        for rpm_frac in (0.25, 0.5, 0.8, 1.0):
            rpm = max(100.0, e.redline_rpm * rpm_frac)
            try:
                vals.append(engine_sim.torque_fraction(e, rpm))
            except Exception:
                vals.append(0.0)
            try:
                vals.append(derived_torque.derive(e, rpm).torque_nm)
            except Exception:
                vals.append(0.0)
            try:
                vals.append(engine_harm.over_rev(e, rpm).inertia_force_n)
            except Exception:
                vals.append(0.0)
        for phi in (1.0, 1.25):
            try:
                vals.append(combustion_efficiency.for_engine(e, phi).efficiency)
            except Exception:
                vals.append(0.0)
        try:
            lub = engine_harm.lubrication(4e-4, arch.bore_m, arch.stroke_m, 800.0, 8.0)
            vals.extend([lub.specific_film, lub.friction_coefficient, lub.friction_heat_w])
            fit = engine_harm.running_fit(arch.bore_m, 450.0, 390.0)
            vals.extend([fit.hot_clearance_m, fit.cold_clearance_m])
        except Exception:
            vals.extend([0.0] * 5)
        try:
            for kind in ("eddy-current", "water-brake"):
                a = dyno_brakes.for_engine(e, kind)
                vals.append(a.capacity_nm(max(500.0, e.torque_peak_rpm)))
        except Exception:
            vals.extend([0.0, 0.0])
        try:
            cyc = thermo_cycles.cycle_of(e)
            import engines as _eng
            vals.append(thermo_cycles.efficiency(
                cyc, era_key=_eng.era_of(e), bore_m=arch.bore_m,
                compression_ratio=float(getattr(e, "compression_ratio", 9.0) or 9.0)).realised)
        except Exception:
            vals.append(0.0)
    return [v if isinstance(v, (int, float)) and math.isfinite(v) else 0.0 for v in vals]


def _relative_change(a: list, b: list) -> float:
    worst = 0.0
    for x, y in zip(a, b):
        denom = max(abs(x), 1e-12)
        worst = max(worst, abs(y - x) / denom)
    return worst


def candidates(modules: list) -> list:
    """Module-level numeric constants, by the project's own convention.

    UPPERCASE module-level names are where this project keeps its
    declared data -- capability_index already indexes them as "table"
    entries for exactly that reason."""
    out = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr in dir(mod):
            if not attr.isupper() or len(attr) <= 3 or attr.startswith("_"):
                continue
            if any(attr.startswith(p) for p in SKIP_PREFIXES):
                continue
            v = getattr(mod, attr, None)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if v == 0:
                continue      # cannot perturb zero multiplicatively
            out.append((name, attr, float(v)))
    return out


def audit(modules: list | None = None, probes=None) -> dict:
    if modules is None:
        modules = ["engine_sim", "engines", "derived_torque", "port_flow",
                   "thermo_cycles", "two_stroke", "engine_harm",
                   "combustion_efficiency", "dyno_brakes", "surface_wetting"]
    probes = probes or _probe_engines()
    base = battery(probes)
    rows = []
    for mod_name, attr, value in candidates(modules):
        mod = importlib.import_module(mod_name)
        try:
            setattr(mod, attr, value * (1.0 + PERTURBATION))
            moved = _relative_change(base, battery(probes))
        except Exception as exc:
            moved = float("nan")
            rows.append({"module": mod_name, "name": attr, "value": value,
                         "sensitivity": moved, "error": str(exc)[:50]})
            setattr(mod, attr, value)
            continue
        finally:
            setattr(mod, attr, value)
        rows.append({"module": mod_name, "name": attr, "value": value,
                     "sensitivity": moved})
    dead = [r for r in rows if r.get("sensitivity") == 0.0]
    rows.sort(key=lambda r: -(r.get("sensitivity") or 0.0))
    return {"probed": len(rows), "zero": len(dead), "rows": rows, "dead": dead,
            "battery_size": len(base)}


def report(modules: list | None = None) -> str:
    res = audit(modules)
    out = [f"zero-Jacobian audit: {res['probed']} constants against a "
           f"{res['battery_size']}-number battery",
           f"perturbation {PERTURBATION * 100:.0f}%; a sensitivity of EXACTLY 0.0 "
           "means the constant never reached any output",
           ""]
    live = [r for r in res["rows"] if (r.get("sensitivity") or 0.0) > 0.0]
    out.append(f"  {len(live)} constants move something. Top ten:")
    for r in live[:10]:
        out.append(f"    {r['module']:24s} {r['name']:34s} {r['sensitivity'] * 100:8.3f}%")
    out.append("")
    out.append(f"  {res['zero']} constants move NOTHING:")
    for r in res["dead"]:
        out.append(f"    {r['module']:24s} {r['name']:34s} = {r['value']!r}")
    out.append("")
    out.append("  A zero here is a question, not a verdict. Classify each before "
               "touching it: dead,")
    out.append("  import-baked (read once at module load, so a runtime change "
               "cannot move it),")
    out.append("  masked behind a clamp or branch, or simply on a path this "
               "battery does not walk.")
    return "\n".join(out)


if __name__ == "__main__":
    print(report())
