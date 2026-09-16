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

  SHAPE       derived physics, normalised to exactly 1.0 at the
              declared torque peak. Valve and port area, the Mach
              index, overlap loss at low speed, runner resonance,
              exhaust scavenging, the way a centrifugal blower's boost
              climbs with the square of speed, two-stroke port
              scavenging. This supplies EVERYTHING ABOUT THE CURVE and
              deliberately nothing about its height.

  RESIDUAL    an optional learned correction, per engine, fitted to
              whatever high-detail data can be scavenged. Defaults to
              exactly 1.0 everywhere, so an engine with no data gets
              pure physics and nothing silently invented for it.

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

def raw_shape(engine, rpm: float) -> float:
    """Unnormalised derived torque at this speed.

    Everything in derived_torque, used for its CURVE rather than its
    height. Any constant that merely scales this cancels in the
    normalisation below, which is the whole point."""
    import derived_torque as dt
    return dt.derive(engine, rpm).torque_nm


def shape(engine, rpm: float) -> float:
    """Derived torque, normalised to 1.0 at the declared torque peak.

    Division by the value at the anchor speed is what makes this a pure
    shape. An engine whose physics says it makes twice as much torque as
    declared still gets the right curve, because that factor of two
    appears in numerator and denominator alike and leaves."""
    peak_rpm = anchor_rpm(engine)
    if peak_rpm <= 0.0:
        return 1.0
    at_peak = raw_shape(engine, peak_rpm)
    if at_peak <= 1e-9:
        return 1.0
    return raw_shape(engine, rpm) / at_peak


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
    hi = float(getattr(engine, "redline_rpm", 4000.0) or 4000.0)
    lo = max(60.0, hi * 0.15)
    return [point(engine, lo + (hi - lo) * i / max(1, steps - 1))
            for i in range(steps)]


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
    derived = raw_shape(engine, ar)
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
