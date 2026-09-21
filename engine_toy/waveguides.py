"""Waveguides as parametric geometry with analytic modes.

WHY THIS IS NOT A FIELD SOLVE.  A waveguide is one of the few shapes
whose fields are known exactly.  Its modes are closed form, its cutoffs
are roots of Bessel functions or simple ratios, and its propagation
constants follow from those by one square root.  Discretising a guide
onto a grid replaces all of that with staircasing error and a
resolution limit, and buys nothing.

The optics stack next door makes the same division deliberately: exact
compound lenses, propagation spans and cached modal operators run as
closed-form parametric transport, and the volumetric field engine is
reserved for what closed form cannot express.  A penetration through a
shield wall is a waveguide, so it belongs on the parametric side.

WHAT FALLS OUT RATHER THAN BEING TOLD.  The trade's rule that a round
penetration attenuates "32 dB per diameter" below cutoff is not an
empirical fit: it is ``(20/ln 10) * 1.8412 * 2 = 31.98``, the evanescent
decay of the TE11 mode over a length of two radii.  This module computes
that from the mode, so the rule of thumb becomes a check rather than an
input.

WHAT IS EXACT AND WHAT IS NOT.  Two distinct products live here and
they are deliberately not fused, because they hold on opposite
geometries and a single blended number would hide which one is
speaking:

  * A guide's two-port is EXACT for the mode it carries.  A uniform
    guide section is a transmission line, and a line's port admittances
    are closed form in ``cosh`` and ``sinh`` of ``gamma * l``.  Nothing
    is sampled, fitted or tabulated.  Its one simplification is
    declared: it carries the dominant mode alone, which holds when the
    tunnel is long against its bore.
  * An aperture's coupling is BETHE's small-hole result, a leading-order
    expansion in ``a/lambda``.  It is an approximation with a stated
    range -- hole small against the wavelength, screen thin against the
    hole -- and it is the term the two-port cannot see, because a
    zero-length line has no attenuation at all while a zero-thickness
    hole still shields enormously.

A real penetration has both: coupling at each mouth, decay down the
tunnel.  Composing them is the caller's business and is explicit, so
that a result can always be traced back to which physics produced it.

CONVENTIONS.  Frequencies in hertz, lengths in metres, attenuation
reported in decibels of field amplitude.  A mode below its cutoff does
not propagate; it decays, and the decay constant is real.  Guides are
assumed non-magnetic, so a fill is a permittivity and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass
import cmath
import math

import numpy as np

from circuit_graph import PortNetwork
from em_materials import VACUUM_PERMEABILITY_H_M, VACUUM_PERMITTIVITY_F_M

SPEED_OF_LIGHT = 299792458.0
NEPERS_TO_DB = 20.0 / math.log(10.0)

#: sqrt(mu0/eps0), 376.73 ohm.  Every guide impedance is this number
#: divided or multiplied by the same cutoff factor, which is a useful
#: check on both branches.
FREE_SPACE_IMPEDANCE_OHM = math.sqrt(
    VACUUM_PERMEABILITY_H_M / VACUUM_PERMITTIVITY_F_M)

#: First zeros of the Bessel derivative J'_m (TE modes) and of J_m (TM
#: modes) in a circular guide.  TE11 is the dominant mode of a round
#: pipe, which is why a drilled hole's cutoff is set by 1.8412.
BESSEL_TE_ROOTS = {(1, 1): 1.8411838, (2, 1): 3.0542369,
                   (0, 1): 3.8317060, (1, 2): 5.3314428}
BESSEL_TM_ROOTS = {(0, 1): 2.4048256, (1, 1): 3.8317060,
                   (2, 1): 5.1356223, (0, 2): 5.5200781}


@dataclass(frozen=True)
class GuideMode:
    """One mode of a guide: what it is called and where it turns on."""

    family: str                 # TE | TM
    m: int
    n: int
    cutoff_hz: float
    #: The medium's speed, not vacuum's.  Both ``k`` and ``k_c`` must be
    #: taken in the SAME medium or the fill cancels out of the cutoff
    #: while surviving everywhere else -- which looks correct in exactly
    #: the one test a careless suite writes.
    phase_speed_m_s: float = SPEED_OF_LIGHT

    @property
    def label(self) -> str:
        return f"{self.family}{self.m}{self.n}"

    @property
    def relative_permittivity(self) -> float:
        """Recovered from the phase speed; guides here are non-magnetic."""
        return (SPEED_OF_LIGHT / self.phase_speed_m_s) ** 2

    def propagates_at(self, frequency_hz: float) -> bool:
        return frequency_hz > self.cutoff_hz

    def wavenumber_1_m(self, frequency_hz: float) -> complex:
        """``beta = sqrt(k^2 - k_c^2)``, real above cutoff, imaginary below.

        Returning it as a complex number keeps one expression for both
        regimes: a propagating mode advances in phase, an evanescent one
        decays, and which happens is the sign under the root rather than
        a branch in the caller.
        """
        speed = self.phase_speed_m_s
        k = 2.0 * math.pi * float(frequency_hz) / speed
        k_c = 2.0 * math.pi * self.cutoff_hz / speed
        squared = k * k - k_c * k_c
        if squared >= 0.0:
            return complex(math.sqrt(squared), 0.0)
        return complex(0.0, math.sqrt(-squared))

    def propagation_constant_1_m(self, frequency_hz: float) -> complex:
        """``gamma = alpha + j beta``, the field's ``exp(-gamma z)`` rate.

        This is :meth:`wavenumber_1_m` said the way the rest of physics
        says it, and the form the transmission-line algebra needs.  The
        two carry the same content: propagating puts it on the imaginary
        axis, evanescent on the real one.
        """
        split = self.wavenumber_1_m(frequency_hz)
        return complex(split.imag, split.real)

    def wave_impedance_ohm(self, frequency_hz: float) -> complex:
        """Transverse E over transverse H: ``j w mu / gamma`` for TE.

        THE SIGN OF THE IMAGINARY PART IS THE PHYSICS.  Above cutoff
        this is real, and a real impedance carries power away.  Below
        cutoff it is purely imaginary -- inductive for TE, capacitive
        for TM -- and a purely reactive impedance carries none.  An
        evanescent mode stores energy and hands it back; it does not
        dissipate.  So a sub-cutoff penetration shields by REFLECTION,
        and writing it as a resistance would give the right decibels
        while lying about where the energy went.
        """
        omega = 2.0 * math.pi * float(frequency_hz)
        gamma = self.propagation_constant_1_m(frequency_hz)
        if gamma == 0.0:
            raise ValueError(
                f"{self.label}: the wave impedance is singular exactly at "
                "cutoff -- TE diverges, TM vanishes; evaluate to either side")
        if self.family == "TE":
            return 1j * omega * VACUUM_PERMEABILITY_H_M / gamma
        permittivity = VACUUM_PERMITTIVITY_F_M * self.relative_permittivity
        return gamma / (1j * omega * permittivity)

    def attenuation_db(self, frequency_hz: float, length_m: float) -> float:
        """Amplitude lost over ``length_m``, in dB.

        Zero for a propagating mode: this accounts for being below
        cutoff, not for wall loss, which is a separate and much smaller
        effect handled by surface resistance.
        """
        beta = self.wavenumber_1_m(frequency_hz)
        if beta.imag <= 0.0:
            return 0.0
        return NEPERS_TO_DB * beta.imag * float(length_m)

    def graph_attributes(self) -> dict:
        return {
            "mode": self.label,
            "cutoff_hz": float(self.cutoff_hz),
        }


def _line_port_network(identity: str, mode: GuideMode, frequency_hz: float,
                       length_m: float,
                       ports: tuple[str, str]) -> PortNetwork:
    """A guide section as the two-port its mode makes it.

    A uniform guide carrying one mode IS a transmission line, and a
    line of length ``l`` has exact port admittances

        Y11 = Y22 = coth(gamma l) / Z0
        Y12 = Y21 = -1 / (Z0 sinh(gamma l))

    ``eliminated_nodes`` is reported as zero because there was never an
    interior to eliminate: the reduction was done in closed form rather
    than by factoring a mesh, which is the whole advantage of knowing
    the geometry.  That is a stronger statement than a Kron reduction,
    not a weaker one -- a Kron reduction is exact for the network it was
    handed, while this is exact for the continuum.
    """
    length = float(length_m)
    if length <= 0.0:
        raise ValueError(
            f"{identity}: a line of zero length is not a model of a hole. "
            "A thin screen couples through aperture polarizability -- see "
            "CircularAperture -- which this two-port cannot represent")
    gamma = mode.propagation_constant_1_m(frequency_hz)
    extent = gamma * length
    if abs(extent) < 1e-12:
        raise ValueError(
            f"{identity}: the two-port is singular exactly at cutoff; "
            "evaluate to either side of it")
    impedance = mode.wave_impedance_ohm(frequency_hz)
    series = 1.0 / (impedance * cmath.sinh(extent))
    shunt = cmath.cosh(extent) / (impedance * cmath.sinh(extent))
    admittance = np.array([[shunt, -series], [-series, shunt]], dtype=complex)
    return PortNetwork(
        identity=identity,
        ports=tuple(ports),
        frequency_hz=float(frequency_hz),
        admittance=admittance,
        eliminated_nodes=0,
    )


@dataclass(frozen=True)
class CircularGuide:
    """A round pipe -- or a drilled penetration, which is the same thing.

    Every cable gland, vent, breather and shaft opening through a shield
    wall is one of these, with ``length_m`` the wall thickness.  That is
    the whole reason this class exists: the geometry is already declared
    by the part, so the attenuation is a consequence rather than a
    parameter.
    """

    identity: str
    radius_m: float
    length_m: float = 0.0
    relative_permittivity: float = 1.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError(f"{self.identity}: a guide needs a positive radius")
        if self.length_m < 0.0:
            raise ValueError(f"{self.identity}: length cannot be negative")

    @property
    def phase_speed_m_s(self) -> float:
        return SPEED_OF_LIGHT / math.sqrt(self.relative_permittivity)

    def cutoff_hz(self, family: str, m: int, n: int) -> float:
        roots = BESSEL_TE_ROOTS if family == "TE" else BESSEL_TM_ROOTS
        if (m, n) not in roots:
            raise ValueError(
                f"{self.identity}: no tabulated root for {family}{m}{n}")
        return roots[(m, n)] * self.phase_speed_m_s / (
            2.0 * math.pi * self.radius_m)

    def modes(self) -> tuple[GuideMode, ...]:
        """Every tabulated mode, in cutoff order."""
        speed = self.phase_speed_m_s
        found = [GuideMode("TE", m, n, self.cutoff_hz("TE", m, n), speed)
                 for (m, n) in BESSEL_TE_ROOTS]
        found += [GuideMode("TM", m, n, self.cutoff_hz("TM", m, n), speed)
                  for (m, n) in BESSEL_TM_ROOTS]
        return tuple(sorted(found, key=lambda mode: mode.cutoff_hz))

    @property
    def dominant_mode(self) -> GuideMode:
        """TE11: the first thing that gets through a round hole."""
        return GuideMode("TE", 1, 1, self.cutoff_hz("TE", 1, 1),
                         self.phase_speed_m_s)

    def attenuation_db(self, frequency_hz: float,
                       length_m: float | None = None) -> float:
        """Shielding from being below cutoff, over the wall's thickness.

        The dominant mode governs: anything higher decays faster, so a
        penetration leaks through TE11 or not at all.
        """
        through = self.length_m if length_m is None else float(length_m)
        return self.dominant_mode.attenuation_db(frequency_hz, through)

    def attenuation_db_per_diameter(self, frequency_hz: float) -> float:
        """The number the trade quotes, computed rather than looked up.

        Far below cutoff this tends to ``(20/ln10) * 1.8412 * 2``, which
        is 31.98 -- the familiar "32 dB per diameter".
        """
        return self.attenuation_db(frequency_hz, 2.0 * self.radius_m)

    def dominant_mode_holds(self, frequency_hz: float,
                            length_m: float | None = None,
                            tolerance: float = 3.0) -> bool:
        """Whether carrying TE11 alone is honest for this bore and wall.

        The next mode down the pipe decays faster, so after a few of its
        own decay lengths only TE11 is left.  Short of that the mouths'
        higher-order content has not died away and a single-mode
        two-port is understating the coupling.  ``tolerance`` is how
        many decay lengths of the SECOND mode the wall must be worth.

        This is the simplification stated as a predicate rather than
        left in a docstring, so a caller can ask before trusting.
        """
        through = self.length_m if length_m is None else float(length_m)
        second = self.modes()[1]
        alpha = second.wavenumber_1_m(frequency_hz).imag
        return alpha * through >= tolerance

    def port_network(self, frequency_hz: float,
                     length_m: float | None = None,
                     ports: tuple[str, str] | None = None) -> PortNetwork:
        """This penetration as a circuit element, ready to stamp.

        The product is the same ``PortNetwork`` a Kron reduction yields,
        so a penetration enters a chamber's circuit exactly as any other
        reduced block does -- which is the point: the field problem and
        the circuit problem were the same operator all along.

        What this DOES NOT include is the coupling at either mouth; see
        the module docstring.  For a wall thin against its bore this
        term is not merely incomplete, it is the smaller one.
        """
        through = self.length_m if length_m is None else float(length_m)
        names = ports if ports is not None else (f"{self.identity}/inside",
                                                 f"{self.identity}/outside")
        return _line_port_network(f"{self.identity}/guide",
                                  self.dominant_mode, frequency_hz,
                                  through, names)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "circular-waveguide",
            "radius_m": float(self.radius_m),
            "length_m": float(self.length_m),
            "te11_cutoff_hz": float(self.cutoff_hz("TE", 1, 1)),
        }


@dataclass(frozen=True)
class RectangularGuide:
    """A rectangular guide, and the reference case for cavity work.

    ``TE_mn`` cutoff is ``(v/2) sqrt((m/a)^2 + (n/b)^2)``, which is the
    same expression a rectangular cavity's resonances are built from --
    one more index and the box is closed.
    """

    identity: str
    width_m: float
    height_m: float
    length_m: float = 0.0
    relative_permittivity: float = 1.0

    def __post_init__(self) -> None:
        if min(self.width_m, self.height_m) <= 0.0:
            raise ValueError(f"{self.identity}: a guide needs positive sides")

    @property
    def phase_speed_m_s(self) -> float:
        return SPEED_OF_LIGHT / math.sqrt(self.relative_permittivity)

    def cutoff_hz(self, family: str, m: int, n: int) -> float:
        if family == "TM" and (m < 1 or n < 1):
            raise ValueError(
                f"{self.identity}: TM{m}{n} does not exist; a TM mode needs "
                "both indices at least one")
        if m < 0 or n < 0 or (m == 0 and n == 0):
            raise ValueError(f"{self.identity}: no mode {family}{m}{n}")
        return 0.5 * self.phase_speed_m_s * math.hypot(
            m / self.width_m, n / self.height_m)

    def modes(self, max_index: int = 2) -> tuple[GuideMode, ...]:
        speed = self.phase_speed_m_s
        found = []
        for m in range(max_index + 1):
            for n in range(max_index + 1):
                if m or n:
                    found.append(GuideMode("TE", m, n,
                                           self.cutoff_hz("TE", m, n), speed))
                if m and n:
                    found.append(GuideMode("TM", m, n,
                                           self.cutoff_hz("TM", m, n), speed))
        return tuple(sorted(found, key=lambda mode: mode.cutoff_hz))

    @property
    def dominant_mode(self) -> GuideMode:
        """TE10 for any guide wider than it is tall, which is most of them."""
        speed = self.phase_speed_m_s
        wide = self.width_m >= self.height_m
        return (GuideMode("TE", 1, 0, self.cutoff_hz("TE", 1, 0), speed) if wide
                else GuideMode("TE", 0, 1, self.cutoff_hz("TE", 0, 1), speed))

    def attenuation_db(self, frequency_hz: float,
                       length_m: float | None = None) -> float:
        through = self.length_m if length_m is None else float(length_m)
        return self.dominant_mode.attenuation_db(frequency_hz, through)

    def port_network(self, frequency_hz: float,
                     length_m: float | None = None,
                     ports: tuple[str, str] | None = None) -> PortNetwork:
        """This guide run as a circuit element; see :meth:`CircularGuide.port_network`."""
        through = self.length_m if length_m is None else float(length_m)
        names = ports if ports is not None else (f"{self.identity}/near",
                                                 f"{self.identity}/far")
        return _line_port_network(f"{self.identity}/guide",
                                  self.dominant_mode, frequency_hz,
                                  through, names)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "rectangular-waveguide",
            "width_m": float(self.width_m),
            "height_m": float(self.height_m),
            "length_m": float(self.length_m),
            "te10_cutoff_hz": float(self.dominant_mode.cutoff_hz),
        }


@dataclass(frozen=True)
class CircularAperture:
    """A small hole in a thin wall, by Bethe's polarizabilities.

    WHY THIS IS NOT THE GUIDE ABOVE.  Shrink a guide's length to zero
    and its attenuation goes to zero with it: a tunnel of no depth
    cannot attenuate.  But a hole in a zero-thickness screen still
    shields enormously, and the mechanism is different in kind.  The
    hole is far too small to support any mode, so nothing propagates
    through it at all; instead the incident field induces effective
    dipoles in the opening and THOSE radiate into the far side.

    Bethe's 1944 result gives the dipoles from the hole's geometry
    alone::

        alpha_m = 4 a^3 / 3      magnetic, from tangential H
        alpha_e = 2 a^3 / 3      electric, from normal E

    Radiated power goes as ``k^4`` times the square of a dipole, and a
    dipole here goes as ``a^3``, so transmission goes as ``k^4 a^6`` --
    the famous sixth power, or equivalently transmitted FRACTION as
    ``(a/lambda)^4``.  That exponent is the physics and is exact in the
    limit; the leading coefficient is the part that is a first term.

    THE APPROXIMATION IS DECLARED, NOT BURIED.  Two conditions, both
    checkable, both exposed as predicates: the hole must be small
    against the wavelength, and the screen thin against the hole.  Ask
    :meth:`is_valid_at` before believing the number.
    """

    identity: str
    radius_m: float
    thickness_m: float = 0.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError(f"{self.identity}: an aperture needs a radius")

    @property
    def magnetic_polarizability_m3(self) -> float:
        """``4 a^3 / 3`` -- the dominant term for a hole in a conductor."""
        return 4.0 * self.radius_m ** 3 / 3.0

    @property
    def electric_polarizability_m3(self) -> float:
        """``2 a^3 / 3`` -- half the magnetic one, and usually secondary.

        A wall carries tangential H and, being a conductor, very little
        normal E, so the magnetic term normally dominates.  Both are
        kept because which one leads depends on where in the cavity the
        hole sits, and that is the caller's geometry, not ours.
        """
        return 2.0 * self.radius_m ** 3 / 3.0

    def free_space_wavenumber_1_m(self, frequency_hz: float) -> float:
        return 2.0 * math.pi * float(frequency_hz) / SPEED_OF_LIGHT

    def transmission_cross_section_m2(self, frequency_hz: float) -> float:
        """``64 k^4 a^6 / (27 pi^2)``, normal incidence, thin PEC screen."""
        k = self.free_space_wavenumber_1_m(frequency_hz)
        return 64.0 * k ** 4 * self.radius_m ** 6 / (27.0 * math.pi ** 2)

    def transmission_coefficient(self, frequency_hz: float) -> float:
        """Transmitted power as a fraction of what lands on the opening."""
        area = math.pi * self.radius_m ** 2
        return self.transmission_cross_section_m2(frequency_hz) / area

    def shielding_db(self, frequency_hz: float) -> float:
        """The aperture term, positive decibels of shielding."""
        transmitted = self.transmission_coefficient(frequency_hz)
        if transmitted <= 0.0:
            return math.inf
        return -10.0 * math.log10(transmitted)

    def size_parameter(self, frequency_hz: float) -> float:
        """``k a``.  Bethe is a leading term in this; it must be small."""
        return self.free_space_wavenumber_1_m(frequency_hz) * self.radius_m

    def is_valid_at(self, frequency_hz: float, *,
                    max_size_parameter: float = 0.5) -> bool:
        """Both conditions, asked rather than assumed.

        ``ka < 0.5`` puts the hole under a twelfth of a wavelength,
        where the leading term still carries the answer.  The thin-screen
        condition is ``t < a``; past that the opening is a tunnel and
        :class:`CircularGuide` is the better description -- and past a
        few diameters the two stop overlapping at all.
        """
        return (self.size_parameter(frequency_hz) < max_size_parameter
                and self.thickness_m < self.radius_m)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "circular-aperture",
            "radius_m": float(self.radius_m),
            "thickness_m": float(self.thickness_m),
            "magnetic_polarizability_m3": self.magnetic_polarizability_m3,
        }


__all__ = [
    "SPEED_OF_LIGHT", "NEPERS_TO_DB", "FREE_SPACE_IMPEDANCE_OHM",
    "BESSEL_TE_ROOTS", "BESSEL_TM_ROOTS",
    "GuideMode", "CircularGuide", "RectangularGuide", "CircularAperture",
]
