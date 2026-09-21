"""Knees: the parametric control surface almost every regime change is.

A diode turning on, a core saturating, a relief valve cracking, an
alternator cutting in, a material yielding, a magnetron striking -- all
of these are one shape.  Something is quiescent, then it is not, and the
interesting parameters are WHERE that happens and HOW ABRUPTLY.  Written
out ad hoc, each becomes a threshold somebody typed; written once, they
become a declared surface with a location and a sharpness that other
parts of the system can read, differentiate and compile.

WHY NOT AN ``if``.  Three reasons, all of them the tree's own:

  * a branch does not lower.  The symbolic lane spells everything
    branch-free for exactly this reason -- ``symbolic_atmosphere``'s
    module docstring is explicit that ``sympy.sign`` has no LLVM
    definition and is not used.
  * a branch has no derivative at the branch, so a VJP through it is
    either wrong or undefined precisely where the interesting physics is.
  * real knees are not sharp.  A diode's turn-on is rounded over a few
    tens of millivolts because that roundness IS ``kT/q``; pretending
    otherwise throws away the one parameter that is not a modelling
    choice.

THE SHARPNESS IS SOMETIMES PHYSICS.  :class:`DiodeKnee` does not take a
knee voltage at all.  It takes a saturation current and an ideality, and
the familiar 0.7 V falls out of the Shockley equation at an ampere --
along with 0.53 V at a milliamp, because a diode's "knee voltage" was
never a constant.  Where a sharpness has a physical origin, this module
takes the origin and derives the sharpness.

EVERY FUNCTION HERE IS SMOOTH, BOUNDED AND ARRAY-SAFE.  They are written
in the stable spellings -- softplus through ``logaddexp``, the logistic
through ``tanh`` -- so that a large argument saturates rather than
overflowing.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

#: Boltzmann's constant over the elementary charge, in volts per kelvin.
BOLTZMANN_OVER_CHARGE_V_K = 1.380649e-23 / 1.602176634e-19


def softplus(x, sharpness: float = 1.0):
    """``s ln(1 + exp(x/s))`` -- a hinge with its corner rounded off.

    As ``sharpness`` goes to zero this becomes ``max(x, 0)`` exactly, and
    for any positive sharpness it is smooth everywhere with derivative
    :func:`logistic`.  Evaluated through ``logaddexp`` so a large
    argument returns ``x`` rather than overflowing ``exp``.
    """
    if sharpness <= 0.0:
        return np.maximum(np.asarray(x, dtype=float), 0.0)
    scaled = np.asarray(x, dtype=float) / sharpness
    return sharpness * np.logaddexp(scaled, 0.0)


def logistic(x, sharpness: float = 1.0):
    """A smooth gate from 0 to 1, and the derivative of :func:`softplus`.

    Spelled through ``tanh`` because ``tanh`` saturates gracefully at
    both ends where ``1/(1+exp(-z))`` overflows at one of them.
    """
    if sharpness <= 0.0:
        return (np.asarray(x, dtype=float) > 0.0).astype(float)
    return 0.5 * (1.0 + np.tanh(np.asarray(x, dtype=float) / (2.0 * sharpness)))


def harmonic_blend(a, b, epsilon: float = 1e-30):
    """``a b / sqrt(a^2 + b^2)``: the blend the atmosphere law uses.

    This is the branch-free join ``symbolic_atmosphere`` puts between
    Stokes and Newton drag.  Where the two are well separated it returns
    the smaller one, which is why it reads as a minimum.

    IT IS NOT A MINIMUM AT THE CROSSOVER.  With ``a == b`` it returns
    ``a / sqrt(2)`` -- measured 3.5355 against 5 -- an undershoot of 29%
    exactly where the two mechanisms are equally active.  For competing
    physical regimes that is arguably the point, since both are running
    and neither is dominant.  For a limit, a cap or a supply bound it is
    wrong, and :func:`smooth_minimum` is the function that means it.

    Positive arguments only: with mixed signs the expression is not
    ordered and the result is meaningless.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a * b / np.sqrt(a * a + b * b + epsilon)


def smooth_minimum(a, b, sharpness: float = 0.0):
    """A minimum that actually is one, rounded over ``sharpness``.

    ``-s ln(exp(-a/s) + exp(-b/s))``.  Unlike :func:`harmonic_blend` this
    approaches the true minimum everywhere: with ``a == b`` it returns
    ``a - s ln 2``, so the error is bounded by the sharpness rather than
    being a fixed fraction of the value.  ``sharpness = 0`` gives the
    exact, non-differentiable minimum.

    Spelled through ``logaddexp`` so neither branch overflows.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if sharpness <= 0.0:
        return np.minimum(a, b)
    return -sharpness * np.logaddexp(-a / sharpness, -b / sharpness)


def smooth_maximum(a, b, sharpness: float = 0.0):
    """The same, the other way up: ``s ln(exp(a/s) + exp(b/s))``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if sharpness <= 0.0:
        return np.maximum(a, b)
    return sharpness * np.logaddexp(a / sharpness, b / sharpness)


@dataclass(frozen=True)
class Knee:
    """A regime change at ``location``, rounded over ``sharpness``.

    ``sharpness`` is in the same units as ``location`` and is the width
    over which the transition happens, not a dimensionless steepness --
    so it can be read off a datasheet or a measurement rather than
    tuned.  A sharpness of zero is admissible and gives the hard hinge,
    which is useful for a reference answer but does not differentiate.
    """

    identity: str
    location: float
    sharpness: float = 0.0
    units: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.sharpness < 0.0:
            raise ValueError(f"{self.identity}: sharpness cannot be negative")

    # ---- the surface itself -------------------------------------------

    def gate(self, value):
        """How far past the knee we are, as a fraction from 0 to 1."""
        return logistic(np.asarray(value, dtype=float) - self.location,
                        self.sharpness)

    def excess(self, value):
        """How much is past the knee: smoothly ``max(value - location, 0)``."""
        return softplus(np.asarray(value, dtype=float) - self.location,
                        self.sharpness)

    def deficit(self, value):
        """How far short of the knee: smoothly ``max(location - value, 0)``."""
        return softplus(self.location - np.asarray(value, dtype=float),
                        self.sharpness)

    def slope(self, value):
        """``d(excess)/d(value)`` -- which is exactly :meth:`gate`."""
        return self.gate(value)

    # ---- what the knee implies ----------------------------------------

    @property
    def hardness(self) -> float:
        """Location over sharpness: how knee-like this knee actually is.

        A large number is a sharp corner and a stiff system.  Below
        about ten the transition is a sizeable fraction of the location
        itself and calling it a knee starts to flatter it.
        """
        if self.sharpness <= 0.0:
            return math.inf
        return abs(self.location) / self.sharpness

    def resolving_dt(self, rate: float) -> float:
        """The timestep needed to see the crossing, at this rate of approach.

        Crossing a transition of width ``sharpness`` while the driving
        quantity moves at ``rate`` takes ``sharpness / rate``; resolving
        it takes rather less.  This is what a knee contributes to a
        ``dt_limit``: stiffness with a number attached, rather than a
        solver discovering it the hard way.
        """
        speed = abs(float(rate))
        if speed <= 0.0:
            return math.inf
        if self.sharpness <= 0.0:
            return 0.0
        return 0.1 * self.sharpness / speed

    def graph_attributes(self) -> dict:
        return {
            "knee_identity": self.identity,
            "knee_location": float(self.location),
            "knee_sharpness": float(self.sharpness),
            "knee_hardness": float(self.hardness),
            "knee_units": self.units,
        }


@dataclass(frozen=True)
class Ceiling:
    """An approach to a limit rather than a departure from a floor.

    Saturation is a knee seen from the other side: the output tracks the
    input until it cannot, and then it stops.  ``limit * tanh(x / limit)``
    is the standard smooth spelling -- unit slope at the origin,
    asymptotic to the limit, smooth throughout -- and it is what a
    magnetic core's B-H curve looks like when nobody is pretending it
    has a corner.
    """

    identity: str
    limit: float
    units: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.limit <= 0.0:
            raise ValueError(f"{self.identity}: a ceiling must be positive")

    def apply(self, value):
        value = np.asarray(value, dtype=float)
        return self.limit * np.tanh(value / self.limit)

    def headroom(self, value):
        """How much of the ceiling is left, from 1 down to 0.

        NOT ``1 - tanh(u)``.  That subtracts nearly equal numbers:
        ``tanh`` reaches 1.0 in float64 by about ``u = 19`` and the
        difference collapses to exactly zero, throwing away how deep
        into saturation a part is just as that starts to matter.
        Measured at ``u = 25.6``: the naive form gives 0.0 where the
        answer is 1.16e-22.

        ``1 - tanh(u) = 2 / (e^2u + 1)`` subtracts nothing and stays
        meaningful past ``u = 300``.

        THIS IS RANGE REDUCTION, NOT A SUBSTITUTE FOR WIDTH.  The
        repository's proof cores evaluate ``tanh`` at any limb width but
        only inside a PROVEN radius -- ``CORE_RADII["tanh"]`` is 0.5 --
        and refuse beyond it rather than extrapolate, because a core
        outside its interval "approximates nothing".  So an argument
        this large has to be reduced before extra limbs are even
        available, and the exponential identity is that reduction;
        ``signal_math.tanh`` splits at the same 0.5 for the same reason.
        Reduce first, then widen if the reduced form still needs it.
        """
        depth = 2.0 * np.abs(np.asarray(value, dtype=float)) / self.limit
        return 2.0 / (np.exp(np.minimum(depth, 709.0)) + 1.0)

    def utilisation(self, value):
        """The fraction of the ceiling in use, before saturation is applied."""
        return np.abs(np.asarray(value, dtype=float)) / self.limit

    @property
    def knee(self) -> Knee:
        """Where the curve stops being straight.

        ``tanh`` departs from unit slope by a few percent at about a
        third of its limit and is visibly bent by two thirds, so the
        bend is not a point and this reports it as a knee with a width
        rather than as a threshold.
        """
        return Knee(f"{self.identity}.bend", 0.66 * self.limit,
                    0.22 * self.limit, self.units,
                    note="the tanh bend, which is a region and not a point")


@dataclass(frozen=True)
class DiodeKnee:
    """Shockley: the knee voltage is an OUTPUT, not a parameter.

    ``i = Is (exp(v / (n Vt)) - 1)`` with ``Vt = kT/q``.  Nothing here
    is told that a silicon diode turns on at 0.7 V; that number appears
    at an ampere, 0.53 V appears at a milliamp, and the fact that they
    differ is the whole reason a "knee voltage" should be asked for at a
    current rather than declared.
    """

    identity: str
    saturation_current_a: float = 1.0e-12
    ideality: float = 1.0
    temperature_k: float = 300.0
    series_dies: int = 1

    def __post_init__(self) -> None:
        if self.saturation_current_a <= 0.0:
            raise ValueError(f"{self.identity}: saturation current must be positive")
        if self.ideality <= 0.0:
            raise ValueError(f"{self.identity}: ideality must be positive")

    @property
    def thermal_voltage_v(self) -> float:
        """``n kT / q`` -- the sharpness, and it is physics, not a choice."""
        return self.ideality * BOLTZMANN_OVER_CHARGE_V_K * self.temperature_k

    def current_a(self, voltage_v):
        """Current for a voltage across the whole string."""
        per_die = np.asarray(voltage_v, dtype=float) / self.series_dies
        # expm1 keeps the reverse-bias limit exact: at large negative
        # bias this returns -Is rather than cancelling to zero.
        return self.saturation_current_a * np.expm1(
            np.clip(per_die / self.thermal_voltage_v, -700.0, 700.0))

    def voltage_v(self, current_a):
        """The voltage the string sits at when carrying this current."""
        current = np.asarray(current_a, dtype=float)
        return self.series_dies * self.thermal_voltage_v * np.log1p(
            np.maximum(current, -self.saturation_current_a * 0.999999)
            / self.saturation_current_a)

    def knee_at(self, current_a: float) -> Knee:
        """The knee this diode presents when asked to carry ``current_a``.

        Location is the voltage it sits at; sharpness is the thermal
        voltage, because that is the width over which the exponential
        turns on.
        """
        return Knee(
            f"{self.identity}.knee",
            float(self.voltage_v(current_a)),
            self.series_dies * self.thermal_voltage_v,
            units="V",
            note=f"Shockley at {current_a:g} A, {self.temperature_k:g} K")

    def graph_attributes(self) -> dict:
        return {
            "part_role": "semiconductor-junction",
            "saturation_current_a": float(self.saturation_current_a),
            "ideality": float(self.ideality),
            "thermal_voltage_v": float(self.thermal_voltage_v),
            "series_dies": int(self.series_dies),
        }


__all__ = [
    "BOLTZMANN_OVER_CHARGE_V_K", "softplus", "logistic", "smooth_min",
    "Knee", "Ceiling", "DiodeKnee",
]
