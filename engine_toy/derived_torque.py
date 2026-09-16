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
#: Compression-ignition runs leaner and with a longer expansion, and
#: reaches a higher share of its own ideal.


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
    # A TWO-STROKE HAS NO INTAKE VALVE, so none of the poppet-valve
    # machinery below applies to it. Its cylinder is filled through
    # ports in the liner by something pushing air in, and what it ends
    # up holding is a scavenging question -- delivery ratio times
    # trapping efficiency -- not a curtain-area question.
    import two_stroke as ts
    sc = ts.scavenge_of(engine)
    if sc:
        fi_ = getattr(engine, "forced_induction", None)
        boost = 0.0
        if fi_ is not None and getattr(fi_, "kind", None) in ("turbo", "supercharger"):
            boost = max(0.0, float(getattr(fi_, "max_boost_frac", 0.0) or 0.0))
        rl = max(1.0, float(getattr(engine, "redline_rpm", 4000.0)))
        return ts.charge(sc, boost_frac=boost, rpm_frac=rpm / rl,
                         bore_m=arch.bore_m, stroke_m=arch.stroke_m,
                         rpm=rpm).charging
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
    ve = port_limited_ve(ports, n_valves, rpm)
    fpr = engine.firing_events_per_rev()
    try:
        ve *= engine.intake_system.resonance_gain(rpm, fpr)
    except Exception:
        pass
    # NO EXHAUST TERM. This was `ve *= 1.0 + scavenging_assist_frac(...)`,
    # which is wrong twice over and is worth leaving described rather
    # than silently deleted, because it looked completely reasonable:
    #
    #  - WRONG QUANTITY. scavenging_assist_frac returns a fraction of
    #    BACKPRESSURE relieved near the tuned rpm, to apply to a solved
    #    pressure. Read as a VE multiplier its 0.35 becomes a 35%
    #    volumetric-efficiency gain, which no engine has ever made.
    #  - WRONG STATE. It was called without a gas temperature, so it
    #    fell through to 293 K and tuned every pipe in the catalogue as
    #    though the engine had never run. The AMC's manifold resolved
    #    to 1143 rpm instead of its real hot 2002, and this function
    #    then returned 1.219 -- 22% ABOVE peak torque -- at 1200 rpm on
    #    an engine whose peak is at 1800.
    #
    # The exhaust's real effect on torque is pumping work against
    # backpressure the fluid circuit solves, applied in
    # engine_cycle_sim where that solved value exists. See
    # engine_sim.torque_fraction's note for the same decision.
    # FORCED INDUCTION MULTIPLIES THE CHARGE OUTRIGHT, and the field is
    # max_boost_frac -- a fraction of an atmosphere, so 1.2 means the
    # manifold sees 2.2 atmospheres absolute.
    #
    # This read a field called peak_boost_bar, which does not exist on
    # ForcedInduction. getattr returned the 0.0 default every time, so
    # every blown and turbocharged engine in the catalogue was being
    # derived as though naturally aspirated -- which is most of why the
    # Wartsila, running 2.5 atmospheres of boost, came out at a
    # twentieth of its real output.
    # AND BOOST IS NOT CONSTANT. max_boost_frac is the CEILING, reached
    # at the top of the range and nowhere near it at the bottom.
    # Applying it flat across the rev range gave every supercharged
    # radial its full wartime manifold pressure at idle.
    #
    # The two kinds build it differently, and the difference is the
    # whole reason people argue about them. A supercharger is geared to
    # the crank, so its boost rises roughly in step with engine speed
    # and is there the instant the throttle is. A turbo is driven by
    # exhaust energy, which goes up far faster than linearly, so it
    # makes almost nothing low down and then arrives all at once.
    fi = getattr(engine, "forced_induction", None)
    if fi is not None and getattr(fi, "kind", None) in ("turbo", "supercharger"):
        ceiling = max(0.0, float(getattr(fi, "max_boost_frac", 0.0) or 0.0))
        redline = max(1.0, float(getattr(engine, "redline_rpm", 4000.0)))
        f = max(0.0, min(1.0, rpm / redline))
        if getattr(fi, "kind") == "supercharger" and                 str(getattr(fi, "blower_type", "roots")) == "centrifugal":
            # A CENTRIFUGAL BLOWER IS NOT A PUMP, IT IS A FAN. It does
            # not trap and carry a fixed volume; it flings air outward,
            # and the pressure it raises goes with the SQUARE of tip
            # speed. So its boost is nearly absent low down and all
            # there at the top -- the opposite behaviour to a Roots, and
            # the reason a Merlin makes its power high up.
            #
            # ForcedInduction.blower_type already declares this, and its
            # own docstring names "a Merlin's two-stage wheelcase
            # blower" as the example. The catalogue had every radial and
            # the Merlin correctly marked centrifugal and the drag
            # motors as roots. Treating them all as positive-
            # displacement put their torque peaks 50 to 80 per cent
            # below where they belong.
            ramp = f * f
        elif getattr(fi, "kind") == "supercharger":
            # RPM CANCELS. A positive-displacement blower is geared to
            # the crank, so rotor speed and engine speed rise together
            # and the ratio of displacements -- which is what sets the
            # pressure ratio -- does not change with speed. A Roots or
            # screw blower makes very nearly the same boost everywhere,
            # which is the entire reason people fit them.
            #
            # engine_cycle_sim._step_forced_induction already says this
            # explicitly and solves it properly, through a real belt
            # with stiffness and damping so the boost LAGS rather than
            # tracking rpm instantly. A ramp here was not a
            # simplification of that, it was a contradiction of it.
            ramp = 1.0
        else:
            # a turbo is driven by exhaust energy, which rises far
            # faster than linearly, so it makes almost nothing low down
            # and then arrives all at once
            ramp = min(1.0, (f / 0.55) ** 2)
        boost_applied = ceiling * ramp
        ve *= 1.0 + boost_applied
    return max(0.05, ve)


def port_limited_ve(ports, n_valves: int, rpm: float) -> float:
    """The project's ONE definition of volumetric efficiency.

    port_flow.PortSet.volumetric_efficiency is NOT this. It returns the
    port CHOKE fraction -- 1.0 meaning "not choked" -- so reading it as
    volumetric efficiency reports 1.00 for any engine below its choke
    rpm, which is nearly all of them nearly all of the time. A dyno
    profile printed exactly that before this function existed.

    Real VE is that choke fraction times what a head of this
    construction achieves when it is NOT choked: no engine fills its
    cylinder completely at atmospheric pressure, because overlap blows
    some charge back out, residual burnt gas takes up room, and the
    port heats the incoming air. Two-valve heads land near 0.82 and
    four-valve near 0.88 at their best.

    Exposed rather than left inline so that anything reporting a VE
    reports the same one this derivation uses."""
    return ports.volumetric_efficiency(rpm) * (0.88 if n_valves >= 4 else 0.82)


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
    fi = getattr(engine, "forced_induction", None)
    boost_applied = 0.0
    # a two-stroke's boost is already inside its delivery ratio
    two_stroke_scav = bool(__import__("two_stroke").scavenge_of(engine))
    if (not two_stroke_scav) and fi is not None and getattr(fi, "kind", None) == "supercharger":
        ceiling = max(0.0, float(getattr(fi, "max_boost_frac", 0.0) or 0.0))
        boost_applied = ceiling
    afr = float(getattr(fuel, "stoich_afr", 14.7) or 14.7)
    # a diesel runs lean at anything but full fuelling; a petrol engine
    # at wide-open throttle runs slightly RICH for power, which is why
    # best-power mixture is not best-economy mixture
    afr *= 1.15 if diesel else 0.92
    burn = air / max(1e-9, afr)
    lhv = float(getattr(fuel, "energy_density_j_per_kg", 43.4e6) or 43.4e6)
    heat = burn * lhv * float(getattr(engine, "combustion_efficiency", 0.85) or 0.85)

    # THE CYCLE AND THE ERA, instead of one fitted constant.
    #
    # OTTO_REALISATION was doing two jobs at once: bridging the ideal
    # cycle to a real engine, AND expressing every difference between a
    # hit-and-miss and a modern turbo diesel. thermo_cycles splits those
    # -- the ideal comes from whichever cycle this engine actually runs,
    # the era factor from what that technology achieved, and the scale
    # factor from the engine's own bore, because heat loss goes with
    # area and work goes with volume.
    import thermo_cycles as tc
    import engines as _eng
    cyc = tc.cycle_of(engine)
    pr, regen = _eng.TURBINE_CYCLES.get(getattr(engine, "identity", ""), (14.0, 0.0))
    eta = tc.efficiency(
        cyc, era_key=_eng.era_of(engine), bore_m=arch.bore_m,
        compression_ratio=cr, gamma=gamma,
        pressure_ratio=pr, regenerator=regen,
    ).realised
    imep = heat * eta / max(1e-9, disp_m3)
    sp = 2.0 * arch.stroke_m * max(0.0, rpm) / 60.0
    fmep = friction_mep_pa(sp, peak_pressure_pa=imep * 1.8, diesel=diesel)
    # A BLOWER HAS TO BE DRIVEN, and on a heavily supercharged engine
    # that is not a rounding error. Compressing the charge is real work
    # taken off the crank before anything reaches the propeller:
    #
    #     w = cp . T_in . (PR^((g-1)/g) - 1) / eta
    #
    # per kilogram of air, and a two-stage wartime aero engine can be
    # spending a sixth of its own output this way. Derived from the
    # boost and the air mass already computed rather than absorbed into
    # a realisation factor -- which is what the era factor was silently
    # doing, and why applying full boost made the radials read high.
    #
    # A turbo pays for its drive differently, in exhaust back-pressure
    # rather than shaft work, and that is already carried by the
    # exhaust system's own restriction. Only the belt-driven case is
    # charged here.
    pump_mep = 0.0
    if fi is not None and getattr(fi, "kind", None) == "supercharger":
        pr = 1.0 + boost_applied
        if pr > 1.001:
            work_per_kg = (1005.0 * 288.0
                           * (pr ** ((1.4 - 1.0) / 1.4) - 1.0) / 0.68)
            pump_mep = air * work_per_kg / max(1e-9, disp_m3)
    bmep = max(0.0, imep - fmep - pump_mep)
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


# ---------------------------------------------------------------------
# CONSERVATION, WHICH MATTERS MORE THAN AGREEMENT
# ---------------------------------------------------------------------
#
# prove_catalogue measures distance from the published figure, and that
# is a weaker test than it looks. The catalogue is not a ground truth --
# thirty-four of its entries carried a defaulted compression ratio until
# this week -- so tuning toward it rewards matching a number rather than
# obeying a law, and a model can be made to agree with it by cheating in
# ways that would be obvious if anyone checked the books.
#
# These check the books. Every one is a law rather than a target, so
# failing any of them is a defect regardless of how close the torque
# figure lands, and passing them all says the model is at least honest
# even where it is inaccurate.
#
#   FIRST LAW        work out cannot exceed the chemical energy in.
#   CYCLE CEILING    realised efficiency cannot exceed the ideal cycle
#                    it claims to be running.
#   CARNOT           and the ideal cycle itself cannot exceed Carnot
#                    between the same temperature limits.
#   FRICTION SIGN    brake work cannot exceed indicated work. An engine
#                    cannot be helped by its own friction.
#   BREATHING        a naturally aspirated engine cannot trap more than
#                    ram tuning can give it; anything more needs a
#                    blower, and the blower has to be declared.
#   TRAPPING         scavenging must lie between perfect displacement
#                    and perfect mixing. Outside those two bounds is not
#                    a scavenging system, it is an arithmetic error.

#: Peak cycle temperature used for the Carnot comparison. Real
#: peak-charge temperatures reach this and no engine sustains it.
AMBIENT_TEMP_K = 293.15
#: Best volumetric efficiency ram tuning alone has ever delivered on a
#: naturally aspirated engine. Past this something is pumping.
MAX_NA_VOLUMETRIC = 1.15


def conservation_check(engine, rpm: float) -> list:
    """Every law this operating point might be breaking. Empty is good."""
    import thermo_cycles as tc
    import engines as _eng
    faults = []
    arch = engine.architecture
    p = derive(engine, rpm)
    fuel = _fuel(engine)
    lhv = float(getattr(fuel, "energy_density_j_per_kg", 43.4e6) or 43.4e6)
    disp_m3 = float(engine.displacement_l) / 1000.0

    # FIRST LAW
    chemical_j = p.fuel_kg_per_cycle * lhv
    indicated_j = p.imep_pa * disp_m3
    if indicated_j > chemical_j + 1e-9:
        faults.append(f"first law: {indicated_j:.0f} J indicated from {chemical_j:.0f} J "
                      "of fuel -- work created from nothing")

    # CYCLE CEILING and CARNOT
    cr = float(getattr(arch, "compression_ratio", 9.0) or 9.0)
    cyc = tc.cycle_of(engine)
    pr, regen = _eng.TURBINE_CYCLES.get(getattr(engine, "identity", ""), (14.0, 0.0))
    res = tc.efficiency(cyc, era_key=_eng.era_of(engine), bore_m=arch.bore_m,
                        compression_ratio=cr, pressure_ratio=pr, regenerator=regen)
    if res.realised > res.ideal + 1e-9:
        faults.append(f"cycle ceiling: realised {res.realised:.3f} exceeds ideal "
                      f"{cyc} {res.ideal:.3f}")
    carnot = 1.0 - AMBIENT_TEMP_K / CYCLE_PEAK_TEMP_K
    if res.ideal > carnot + 1e-9:
        faults.append(f"carnot: ideal {cyc} {res.ideal:.3f} exceeds Carnot {carnot:.3f}")

    # FRICTION SIGN
    if p.bmep_pa > p.imep_pa + 1e-9:
        faults.append(f"friction sign: brake {p.bmep_pa/1e5:.2f} bar exceeds indicated "
                      f"{p.imep_pa/1e5:.2f} bar")

    # BREATHING
    fi = getattr(engine, "forced_induction", None)
    blown = fi is not None and getattr(fi, "kind", None) in ("turbo", "supercharger")
    if not blown and p.ve > MAX_NA_VOLUMETRIC:
        faults.append(f"breathing: VE {p.ve:.2f} with no declared blower "
                      f"(ram tuning tops out near {MAX_NA_VOLUMETRIC})")

    # TRAPPING, for anything scavenged
    import two_stroke as ts
    sc = ts.scavenge_of(engine)
    if sc:
        boost = float(getattr(fi, "max_boost_frac", 0.0) or 0.0) if blown else 0.0
        r = ts.charge(sc, boost_frac=boost, bore_m=arch.bore_m,
                      stroke_m=arch.stroke_m, rpm=rpm)
        lo = ts.perfect_mixing_trapping(r.delivery_ratio)
        hi = ts.perfect_displacement_trapping(r.delivery_ratio)
        if not (lo - 1e-9 <= r.trapping <= hi + 1e-9):
            faults.append(f"trapping {r.trapping:.3f} outside the Hopkinson bounds "
                          f"[{lo:.3f}, {hi:.3f}] at DR {r.delivery_ratio:.2f}")
    return faults


def audit_catalogue() -> dict:
    """Run every law over every engine at several speeds.

    This is the test that matters. Distance from the catalogue is
    informative; a broken conservation law is disqualifying."""
    import engines as _eng
    bad = {}
    checked = 0
    for ident, e in _eng.BY_IDENTITY.items():
        if getattr(e, "kind", "") != "combustion" or not getattr(e.architecture, "bore_m", 0.0):
            continue
        rl = max(1.0, float(getattr(e, "redline_rpm", 4000.0)))
        for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
            checked += 1
            try:
                f = conservation_check(e, rl * frac)
            except Exception as exc:
                f = [f"raised: {str(exc)[:70]}"]
            if f:
                bad.setdefault(ident, []).extend(
                    f"@{rl*frac:.0f} rpm: {x}" for x in f)
    return {"points_checked": checked, "engines_with_faults": len(bad),
            "faults": bad}
