"""Torque from three contributors, in the order they can be trusted.

An engine's torque curve is one number at every speed, and there are
three genuinely different ways to know it. This composes them instead
of making them compete:

    torque(rpm) = anchor  x  shape(rpm)  x  residual(rpm)

  ANCHOR      the declared bmep_pa, per engine, from industry figures.
              Brake mean effective pressure is the right thing to
              declare because it is INTENSIVE -- it does not depend on
              how big the engine is, so it is comparable across the
              whole catalogue and a wrong one is obvious. This supplies
              MAGNITUDE and nothing else.

  SHAPE       engine_sim.torque_fraction -- THE PROJECT'S OWN, the one
              the live sim integrates, not a second opinion computed
              beside it. Intake runner resonance against the catalogued
              manifold, port/valve-curtain choke, and the real FMEP
              friction correlation, each as a ratio against its own
              value at the torque peak. Supplies EVERYTHING ABOUT THE
              CURVE and deliberately nothing about its height.

              This used to call derived_torque.derive, a parallel
              offline model. That was a mistake worth naming: it
              recomputed what the sim already solves, and it called the
              harmonics with DEFAULT arguments, so every pipe in the
              catalogue was tuned as though it were at room
              temperature. The AMC 258 then "peaked" at 1128 rpm
              against a declared 1800 and read 1.219 -- 22% ABOVE peak
              torque -- at 1200 rpm. The same geometry at a real 900 K
              exhaust tunes to 2002 rpm, and the catalogue figure was
              right all along. A reported shape defect was a
              disconnected model.

  RESIDUAL    an optional learned correction, per engine, fitted to
              whatever high-detail data can be scavenged. Defaults to
              exactly 1.0 everywhere, so an engine with no data gets
              pure physics and nothing silently invented for it.

AND WHAT THIS IS NOT: A MEASURED CURVE. Everything here is a
steady-state estimate at wide-open throttle. It has no exhaust term at
all, because the exhaust's effect on torque is pumping work against a
backpressure only the fluid circuit can solve, and the circuit is
stateful. For the real answer you run the engine:
EngineCycleSim.start_wot_dyno_pull() spins a real drum through real
gears with real solved gas in the pipes. `pull()` below is that, and it
is what a power curve for this project actually means.

WHY THE SPLIT IS NOT COSMETIC. Fitting absolute torque needs the model
to get magnitude AND shape right from the same parameters, and with one
scalar per engine that problem is underdetermined -- measurably so: two
of the constants in the previous arrangement moved the catalogue error
by exactly zero per cent, meaning the data could not see them at all.
Normalising removes the magnitude degree of freedom entirely. Constants
that only scale the answer stop mattering, and what remains to be
identified is the curve, which is what the physics is actually good at.

AND WHY THE RESIDUAL LEARNS THE GAP, NOT THE ANSWER. A network fitted
to torque directly would happily replace the physics, and then the
model knows a number without knowing why. Fitted to the RESIDUAL it can
only ever express what the derivation failed to capture -- so its size
is a direct readout of how much is still unexplained, it cannot exceed
its declared bounds, and deleting it returns the honest physics rather
than nothing. This is the shape wear_profile.INDUSTRIAL_SPECS already
uses: a small table of real anchors, deliberately sparse, with the
provenance next to the number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------
# 1. the anchor
# ---------------------------------------------------------------------

def anchor_nm(engine) -> float:
    """Peak torque from the declared bmep. The industry figure.

    Engine.peak_torque_nm is already exactly this -- bmep times swept
    volume over the cycle's radians -- so this does not recompute it,
    it names it. The declaration is the anchor and nothing here is
    allowed to move it."""
    return float(getattr(engine, "peak_torque_nm", 0.0) or 0.0)


def anchor_rpm(engine) -> float:
    return float(getattr(engine, "torque_peak_rpm", 0.0) or 0.0)


# ---------------------------------------------------------------------
# 2. the shape
# ---------------------------------------------------------------------

def shape(engine, rpm: float) -> float:
    """The project's own WOT torque fraction, at this speed.

    engine_sim.torque_fraction is already exactly a normalised shape --
    every term in it is a ratio against that term's own value at
    torque_peak_rpm, and the intake is normalised against the
    CATALOGUED manifold, so it returns exactly 1.0 for an unmodified
    engine at its declared peak and moves in absolute terms when
    hardware changes. There is nothing left for this module to
    normalise, and re-normalising it here would undo that."""
    import engine_sim as es
    return es.torque_fraction(engine, rpm)


# ---------------------------------------------------------------------
# 3. the residual
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Residual:
    """A learned correction for one engine, and where it came from.

    `points` are (rpm, multiplier) pairs -- what the real curve does
    that the derivation did not. Between them it interpolates; outside
    them it holds the end value rather than extrapolating, because
    extrapolating a fit past its data is how a correction becomes an
    invention.

    `bound` caps how far it may ever move the answer. A residual that
    wants more than this is not correcting the physics, it is replacing
    it, and the right response is to fix the physics."""
    source: str
    points: tuple = ()
    bound: float = 0.25

    def at(self, rpm: float) -> float:
        if not self.points:
            return 1.0
        pts = sorted(self.points)
        r = float(rpm)
        if r <= pts[0][0]:
            v = pts[0][1]
        elif r >= pts[-1][0]:
            v = pts[-1][1]
        else:
            v = pts[-1][1]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x0 <= r <= x1:
                    t = (r - x0) / max(1e-9, x1 - x0)
                    v = y0 + (y1 - y0) * t
                    break
        return max(1.0 - self.bound, min(1.0 + self.bound, v))


#: DELIBERATELY EMPTY, exactly as wear_profile.INDUSTRIAL_SPECS was.
#:
#: An entry here should come from a real measured curve -- a
#: manufacturer's published torque plot, a dyno sheet, an SAE paper --
#: digitised into (rpm, measured/derived) pairs. Inventing entries would
#: make the blend agree with itself and prove nothing, which is worse
#: than having none, because the agreement would look like validation.
#:
#: The right first candidates are engines with published full curves
#: rather than a single peak figure, since one curve is twenty points
#: against the one a peak value gives.
RESIDUALS: dict[str, Residual] = {}


def residual(engine) -> Residual:
    return RESIDUALS.get(getattr(engine, "identity", ""),
                         Residual(source="none: pure derivation"))


# ---------------------------------------------------------------------
# the composition
# ---------------------------------------------------------------------

@dataclass
class CurvePoint:
    rpm: float
    torque_nm: float
    power_kw: float
    anchor_nm: float
    shape: float
    residual: float

    # SI is what is computed; these are reading conveniences, derived on
    # demand so they can never drift out of step with the real figure.
    @property
    def bhp(self) -> float:
        import units
        return units.kw_to_hp(self.power_kw)

    @property
    def lb_ft(self) -> float:
        import units
        return units.lb_ft(self.torque_nm)

    def describe(self) -> str:
        import units
        return (f"{self.rpm:5.0f} rpm  {units.torque(self.torque_nm)}  "
                f"{units.power(self.power_kw)}")


def torque_nm(engine, rpm: float) -> float:
    a = anchor_nm(engine)
    if a <= 0.0:
        return 0.0
    return a * shape(engine, rpm) * residual(engine).at(rpm)


def point(engine, rpm: float) -> CurvePoint:
    a, s = anchor_nm(engine), shape(engine, rpm)
    r = residual(engine).at(rpm)
    t = a * s * r
    return CurvePoint(rpm=rpm, torque_nm=t,
                      power_kw=t * rpm * 2.0 * math.pi / 60.0 / 1000.0,
                      anchor_nm=a, shape=s, residual=r)


def curve(engine, steps: int = 24) -> list:
    """The steady-state ESTIMATE across the rev range.

    Cheap, stateless, and exhaust-blind -- see the module docstring.
    Use it to compare hardware or to sanity-check a catalogue entry.
    Do not report it as this engine's power curve; for that, pull()."""
    hi = float(getattr(engine, "redline_rpm", 4000.0) or 4000.0)
    lo = max(60.0, hi * 0.15)
    return [point(engine, lo + (hi - lo) * i / max(1, steps - 1))
            for i in range(steps)]


def pull(engine, max_seconds: float = 90.0) -> dict:
    """A REAL measurement: run the engine on the dyno.

    READ THIS BEFORE USING THE TORQUE NUMBER. This is the one-button WOT
    INERTIA pull -- settle, shift, run the final gear to redline -- and
    it is the right instrument for PEAK POWER and the wrong one for a
    torque curve. The drum's torque is only tracked while the clutch is
    genuinely locked, which on the AMC 258 means a sampled window of
    roughly 3140-4300 crank rpm. That engine's declared torque peak is
    at 1800 rpm, and the pull never goes there under lock. So
    peak_torque_nm here is the largest torque IN THE SAMPLED WINDOW, not
    the engine's peak, and `rpm_span` is reported next to it so the
    number cannot be read without its own range.

    A real torque curve needs a SWEPT-LOAD pull that holds each speed
    while the brake absorbs the difference -- a different instrument,
    not a different wrapper around this one. Not built yet; said here
    rather than implied by a function name.

    This is what a power curve means here. It starts the same one-button
    WOT inertia pull the dashboard does -- settle, shift, run the final
    gear to redline -- and reports what the drum actually saw. Along the
    way the fluid circuits solve real exhaust pressure and temperature,
    that temperature sets the speed of sound, the pipe's own geometry
    turns it into a tuned rpm, and the assist tapers the solved
    backpressure into real pumping work. None of which the estimate
    above can see, because none of it exists without running.

    Note also that the drum is POST-GEARING: its torque is the crank's
    times the ratio, so a drum figure must never be compared against a
    catalogue crank figure. Both are reported separately below for
    exactly that reason.

    Costs real simulated seconds; that is the price of measuring
    instead of asserting."""
    import engine_cycle_sim as ecs
    import units as _u
    sim = ecs.EngineCycleSim(engine=engine)
    sim.start()
    sim.start_wot_dyno_pull()
    dt = ecs.FIXED_PHYSICS_DT_S
    samples, t = [], 0.0
    while t < max_seconds:
        sim.step(dt)
        t += dt
        st = sim.state
        if (st.dyno_pull_state == "pulling" and st.brake_clutch_locked
                and st.rpm > 0.0):
            # crank AND drum, never conflated: (crank rpm, crank torque,
            # drum rpm, drum torque)
            samples.append((st.rpm, st.current_torque_nm,
                            st.dyno_rpm, st.dyno_torque_nm))
        if st.dyno_pull_complete_flag:
            break
    st = sim.state
    span = ((min(x[0] for x in samples), max(x[0] for x in samples))
            if samples else (0.0, 0.0))
    return {"identity": getattr(engine, "identity", "?"),
            "completed": bool(st.dyno_pull_complete_flag),
            "seconds": t,
            "rpm_span": span,
            "crank_peak_torque_nm": max((x[1] for x in samples), default=0.0),
            "peak_torque_nm": st.dyno_pull_peak_torque_nm,
            "peak_torque_rpm": st.dyno_pull_peak_torque_rpm,
            "peak_power_kw": st.dyno_pull_peak_power_kw,
            "peak_power_bhp": _u.kw_to_hp(st.dyno_pull_peak_power_kw),
            "peak_power_rpm": st.dyno_pull_peak_power_rpm,
            "peak_torque_lb_ft": _u.lb_ft(st.dyno_pull_peak_torque_nm),
            "samples": samples,
            "what": "measured at the drum, POST-GEARING, on an inertia "
                    "pull over rpm_span only -- peak power is the "
                    "trustworthy figure here; peak torque is bounded by "
                    "that window, and drum torque is not a crank figure"}


# ---------------------------------------------------------------------
# what the derivation is still FOR
# ---------------------------------------------------------------------

def audit_anchor(engine) -> dict:
    """Does the physics reproduce the declared bmep?

    The derivation no longer sets magnitude, so this is not a scoring
    metric any more -- it is a DISAGREEMENT DETECTOR. Where the two
    differ badly one of them is wrong, and both answers are worth
    having: a missing mechanism in the physics, or a bad catalogue
    entry.

    It has already earned this. Both derivations independently said the
    Fairbanks-Morse hit-and-miss should make more than its declared 2.2
    bar, and real engines of that type make four to five."""
    a, ar = anchor_nm(engine), anchor_rpm(engine)
    if a <= 0.0 or ar <= 0.0:
        return {"identity": getattr(engine, "identity", "?"), "checkable": False}
    import derived_torque as dt
    derived = dt.derive(engine, ar).torque_nm
    return {"identity": getattr(engine, "identity", "?"), "checkable": True,
            "declared_nm": a, "derived_nm": derived,
            "disagreement": derived / a - 1.0,
            "declared_bmep_bar": float(getattr(engine, "bmep_pa", 0.0)) / 1e5,
            "has_residual": bool(RESIDUALS.get(getattr(engine, "identity", ""))),
            }


def audit_catalogue() -> dict:
    """Every engine's disagreement, worst first."""
    import engines as _eng
    rows = []
    for ident, e in _eng.BY_IDENTITY.items():
        if getattr(e, "kind", "") != "combustion":
            continue
        if not getattr(e.architecture, "bore_m", 0.0):
            continue
        try:
            r = audit_anchor(e)
        except Exception as exc:
            rows.append({"identity": ident, "checkable": False,
                         "error": str(exc)[:60]})
            continue
        if r.get("checkable"):
            rows.append(r)
    rows.sort(key=lambda r: -abs(r.get("disagreement", 0.0)))
    d = [abs(r["disagreement"]) for r in rows if "disagreement" in r]
    return {"engines": len(rows),
            "median_disagreement": sorted(d)[len(d) // 2] if d else float("nan"),
            "rows": rows,
            "why": "disagreement is a flag, not a score. The anchor supplies "
                   "magnitude; this asks whether the physics can independently "
                   "arrive at it, and a large gap means one of the two is wrong"}
