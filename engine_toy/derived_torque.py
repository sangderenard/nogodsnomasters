"""Torque derived from the air the engine can trap, and made to prove it.

peak_torque_nm is a catalogued number. Everything downstream scales it,
which means the engine's own hardware -- its valves, its manifold, its
compression ratio, its fuel -- cannot actually decide how much torque it
makes. Port it, cam it, feed it something with more energy in it, and
the peak does not move, because the peak was typed in.

This derives it instead, by the chain a real engine actually follows:

    air trapped  =  VE  x  swept volume  x  charge density
    fuel         =  air / air-fuel ratio
    heat         =  fuel x heating value x combustion efficiency
    indicated    =  heat x thermal efficiency
    brake        =  indicated - friction
    torque       =  brake mean effective pressure x volume / 4.pi

Every term on the right is hardware or fuel, and every one of them is
already declared somewhere in this project. Nothing here is fitted to a
torque figure.

AND THEN IT HAS TO PROVE IT. A derivation that lands nowhere near the
catalogue is a nicer-looking lie than the constant it replaced, so
`prove()` runs the derivation against the published figure for every
engine and reports the error. Where the two disagree badly, that is
information -- either the derivation is missing a mechanism or the
catalogue entry is wrong, and both are worth knowing. The verdict is
the spread, not any single engine.

WHAT THE FOUR-PI IS. A four-stroke fires once every two revolutions, so
one cycle spans 4.pi radians of crank rotation. Work per cycle divided
by that is torque. A two-stroke fires every revolution and gets 2.pi,
which is the entire reason a two-stroke of the same size makes more
torque and the same reason it drinks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

AIR_DENSITY_KG_M3 = 1.184          # 25 C, sea level
#: Share of the air-standard Otto efficiency a real engine actually
#: reaches. It is not 1.0 because the ideal cycle burns instantly, loses
#: no heat through the walls and opens the exhaust valve at bottom dead
#: centre. Real engines land near three quarters, and this is the one
#: number here that is fitted rather than derived -- which is why
#: prove() exists to check whether one value serves the whole catalogue.
OTTO_REALISATION = 0.78
#: Compression-ignition runs leaner and with a longer expansion, and
#: reaches a higher share of its own ideal.
DIESEL_REALISATION = 0.82


def air_standard_efficiency(compression_ratio: float, gamma: float = 1.35) -> float:
    """The ceiling, from the compression ratio alone.

    1 - r^(1-gamma). This is why raising compression raises efficiency
    and why knock -- which limits compression -- is the single biggest
    constraint on a petrol engine's efficiency."""
    r = max(1.5, float(compression_ratio))
    return 1.0 - r ** (1.0 - float(gamma))


def friction_mep_pa(mean_piston_speed_m_s: float, peak_pressure_pa: float = 5e6,
                    diesel: bool = False) -> float:
    """Friction, as a mean effective pressure, from piston speed.

    A Chen-Flynn-shaped correlation: a constant term (bearings, seals,
    valvetrain, oil pump), a term linear in piston speed (ring drag) and
    a term in its square (hydrodynamic losses). The square is what makes
    high-revving engines lose so much of their own output to themselves,
    and it is why friction, not breathing, sets the practical redline of
    a big slow engine."""
    sp = max(0.0, float(mean_piston_speed_m_s))
    base = 110_000.0 if diesel else 75_000.0
    load = 0.006 * max(0.0, float(peak_pressure_pa))
    return base + load + 4_800.0 * sp + 400.0 * sp * sp


@dataclass
class DerivedPoint:
    rpm: float
    ve: float
    air_kg_per_cycle: float
    fuel_kg_per_cycle: float
    imep_pa: float
    fmep_pa: float
    bmep_pa: float
    torque_nm: float
    power_kw: float


def _fuel(engine):
    import working_fluids as wf
    key = getattr(engine, "preferred_fuel_profile", None)
    f = wf.WORKING_FLUIDS.get(key) if key else None
    if f is None:
        f = wf.WORKING_FLUIDS["pump-gasoline-87"]
    return f


def volumetric_efficiency(engine, rpm: float) -> float:
    """What fraction of its own swept volume the engine actually fills.

    Three real effects, each from a part that already exists: the valve
    and port area (port_flow), the intake runner's resonance
    (IntakeSystem, which owns the tuning), and the exhaust's scavenging
    assist (ExhaustSystem). Nothing is invented here -- this composes
    what the engine already declares."""
    arch = engine.architecture
    import port_flow as pf
    import cylinder_ports as cp
    try:
        n_valves = cp.valves_per_cylinder(engine)
    except Exception:
        n_valves = 2
    lift = getattr(getattr(engine, "lifter_spring", None), "valve_lift_mm", 9.5) / 1000.0
    ports = pf.PortSet(side="intake", bore_m=arch.bore_m, stroke_m=arch.stroke_m,
                       valves_total=n_valves, lift_m=lift)
    # THE BASELINE IS NOT 1.0. PortSet.volumetric_efficiency returns 1.0
    # when the port is not choking, which is the right answer to the
    # question it asks -- "is gas speed limiting yet" -- and the wrong
    # baseline for trapping. A real engine never traps its full swept
    # volume even with a wide-open unchoked port: valve overlap blows
    # some charge back out, residual burnt gas takes up room, and the
    # incoming air is heated by the port on the way in. Two-valve heads
    # land near 0.82 and four-valve near 0.88 at their best.
    #
    # Starting from 1.0 and then multiplying by BOTH the intake
    # resonance gain AND the exhaust scavenging assist gave every
    # naturally-aspirated engine a VE around 1.45, which no
    # naturally-aspirated engine has ever achieved.
    ve = ports.volumetric_efficiency(rpm) * (0.88 if n_valves >= 4 else 0.82)
    fpr = engine.firing_events_per_rev()
    try:
        ve *= engine.intake_system.resonance_gain(rpm, fpr)
    except Exception:
        pass
    try:
        ve *= 1.0 + engine.exhaust_system.scavenging_assist_frac(rpm, fpr)
    except Exception:
        pass
    # forced induction multiplies the charge density outright
    fi = getattr(engine, "forced_induction", None)
    if fi is not None and getattr(fi, "kind", None) in ("turbo", "supercharger"):
        ve *= 1.0 + float(getattr(fi, "peak_boost_bar", 0.0) or 0.0) / 1.01325
    return max(0.05, ve)


def derive(engine, rpm: float) -> DerivedPoint:
    """One point on the torque curve, from hardware and fuel only."""
    arch = engine.architecture
    disp_m3 = float(engine.displacement_l) / 1000.0
    two_stroke = bool(getattr(arch, "two_stroke", False))
    fuel = _fuel(engine)
    gamma = float(getattr(fuel, "gamma", 1.35) or 1.35)
    cr = float(getattr(arch, "compression_ratio", 9.0) or 9.0)
    # COMPRESSION RATIO IS UNDECLARED ACROSS ALMOST THE WHOLE CATALOGUE.
    # 34 of 36 engines carry EngineArchitecture's dataclass default of
    # 10.0 -- a 1900s hit-and-miss that really runs about 4:1, a
    # low-compression emissions-era six at 8.6, a heavy industrial
    # diesel at 16.5 and a slow-speed marine diesel at 19 all claim the
    # same number. See undeclared_compression() below.
    #
    # It matters here more than anywhere else, because compression ratio
    # sets the air-standard efficiency directly, and because it is the
    # only thing distinguishing a diesel from a petrol engine in this
    # derivation. With every engine at 10.0, no diesel in the catalogue
    # is treated as one.
    diesel = cr >= 14.0

    ve = volumetric_efficiency(engine, rpm)
    air = ve * disp_m3 * AIR_DENSITY_KG_M3
    afr = float(getattr(fuel, "stoich_afr", 14.7) or 14.7)
    # a diesel runs lean at anything but full fuelling; a petrol engine
    # at wide-open throttle runs slightly RICH for power, which is why
    # best-power mixture is not best-economy mixture
    afr *= 1.15 if diesel else 0.92
    burn = air / max(1e-9, afr)
    lhv = float(getattr(fuel, "energy_density_j_per_kg", 43.4e6) or 43.4e6)
    heat = burn * lhv * float(getattr(engine, "combustion_efficiency", 0.85) or 0.85)

    eta = air_standard_efficiency(cr, gamma) * (DIESEL_REALISATION if diesel
                                                else OTTO_REALISATION)
    imep = heat * eta / max(1e-9, disp_m3)
    sp = 2.0 * arch.stroke_m * max(0.0, rpm) / 60.0
    fmep = friction_mep_pa(sp, peak_pressure_pa=imep * 1.8, diesel=diesel)
    bmep = max(0.0, imep - fmep)
    span = (2.0 if two_stroke else 4.0) * math.pi
    torque = bmep * disp_m3 / span
    return DerivedPoint(rpm=rpm, ve=ve, air_kg_per_cycle=air,
                        fuel_kg_per_cycle=burn, imep_pa=imep, fmep_pa=fmep,
                        bmep_pa=bmep, torque_nm=torque,
                        power_kw=torque * rpm * 2.0 * math.pi / 60.0 / 1000.0)


def curve(engine, lo: float | None = None, hi: float | None = None,
          steps: int = 32) -> list:
    """Sweep this engine's own range.

    `lo` is a FRACTION of redline, not a fixed rpm. A fixed 600 rpm
    floor is above the redline of every large slow-speed marine engine
    there is -- the Wartsila turns 102 rpm flat out -- so the sweep ran
    backwards and reported a twentieth of its real output."""
    hi = float(hi if hi is not None else engine.redline_rpm)
    lo = float(lo if lo is not None else max(60.0, hi * 0.15))
    if hi <= lo:
        hi = lo * 1.5
    return [derive(engine, lo + (hi - lo) * i / max(1, steps - 1))
            for i in range(steps)]


def prove(engine) -> dict:
    """Derived peak against the catalogued one. The honesty check.

    Reports both peaks and where they occur, because landing the right
    NUMBER at the wrong SPEED is a different failure from landing the
    wrong number, and the two want different fixes."""
    pts = curve(engine)
    if not pts:
        return {"identity": getattr(engine, "identity", "?"), "ok": False}
    best_t = max(pts, key=lambda p: p.torque_nm)
    best_p = max(pts, key=lambda p: p.power_kw)
    claim_t = float(getattr(engine, "peak_torque_nm", 0.0) or 0.0)
    claim_p = float(getattr(engine, "peak_power_kw", 0.0) or 0.0)
    return {
        "identity": getattr(engine, "identity", "?"),
        "derived_nm": best_t.torque_nm, "claimed_nm": claim_t,
        "torque_err": (best_t.torque_nm / claim_t - 1.0) if claim_t > 0 else float("nan"),
        "derived_nm_rpm": best_t.rpm,
        "claimed_nm_rpm": float(getattr(engine, "torque_peak_rpm", 0.0) or 0.0),
        "derived_kw": best_p.power_kw, "claimed_kw": claim_p,
        "power_err": (best_p.power_kw / claim_p - 1.0) if claim_p > 0 else float("nan"),
        "derived_kw_rpm": best_p.rpm,
        "claimed_kw_rpm": float(getattr(engine, "power_peak_rpm", 0.0) or 0.0),
        "ve_at_peak": best_t.ve,
        "bmep_bar": best_t.bmep_pa / 1e5, "fmep_bar": best_t.fmep_pa / 1e5,
    }


def prove_catalogue(only_combustion: bool = True) -> dict:
    """Run the proof over every engine there is.

    The verdict is the SPREAD. One engine agreeing proves nothing; a
    whole catalogue agreeing to within a sensible band means the chain
    is carrying the physics rather than being tuned to one case."""
    import engines as _eng
    rows = []
    for ident, e in _eng.BY_IDENTITY.items():
        if only_combustion and getattr(e, "kind", "") != "combustion":
            continue
        if not getattr(e.architecture, "bore_m", 0.0):
            continue
        try:
            r = prove(e)
        except Exception as exc:
            rows.append({"identity": ident, "error": str(exc)[:60]})
            continue
        rows.append(r)
    good = [r for r in rows if "error" not in r and r.get("claimed_nm", 0) > 0
            and not math.isnan(r.get("torque_err", float("nan")))]
    errs = sorted(abs(r["torque_err"]) for r in good)
    within = lambda f: sum(1 for e in errs if e <= f) / max(1, len(errs))
    return {"engines": len(rows), "checkable": len(good),
            "median_abs_err": errs[len(errs) // 2] if errs else float("nan"),
            "within_20pct": within(0.20), "within_35pct": within(0.35),
            "within_50pct": within(0.50), "rows": rows}


def undeclared_compression() -> dict:
    """Which engines never declared a compression ratio.

    Kept as a function rather than a comment because it is checkable and
    will stop being true as entries get filled in. A derivation is only
    as good as what it is derived FROM, and this is the largest single
    hole in the inputs -- bigger than anything in the arithmetic above.

    Deliberately not filled in here. Reconstructing thirty-four
    compression ratios from memory would make prove() agree with the
    catalogue for the wrong reason, which is worse than disagreeing for
    the right one."""
    import dataclasses
    import engines as _eng
    default = None
    for f in dataclasses.fields(_eng.EngineArchitecture):
        if f.name == "compression_ratio":
            default = f.default
    rows = []
    for ident, e in _eng.BY_IDENTITY.items():
        cr = getattr(e.architecture, "compression_ratio", None)
        if cr == default:
            rows.append(ident)
    return {"default": default, "undeclared": sorted(rows),
            "count": len(rows), "of": len(_eng.BY_IDENTITY),
            "why": "compression ratio sets air-standard efficiency directly and is the "
                   "only thing here that distinguishes a diesel from a petrol engine. "
                   "While it is defaulted, no derived output can be better than the "
                   "default"}
