"""Bodies that absorb a field, and the states that absorption drives.

THE TERM.  There is no "radiated dewar" in the trade.  The structure
that couples microwave power into a load is an APPLICATOR, and that is
the right name for the object -- a dewar with a feed is a cryogenic
microwave applicator.  The deliberately lossy body you put inside to
heat something that will not absorb on its own is a SUSCEPTOR.  And
warming a cold surface to drive off what has condensed on it is
REGENERATION, which is exactly the cryopump word for defrosting a cold
trap.  Three different objects, three different words, and this module
has all three because they do different jobs.

WHY A CRYOGEN CANNOT BE MICROWAVE-HEATED.  Absorption needs a dipole
that can turn to follow the field.  Liquid nitrogen, argon and helium
are nonpolar and have essentially no loss, and cooling drops what little
there is.  So an applicator around a dewar does not warm the cryogen: it
warms a susceptor, or it warms water ice and adsorbed layers, or it
drives a mode for something inside to sit in.  Each of those is useful
and each is a different declared object; none of them is "boiling the
nitrogen", and the module refuses to pretend otherwise by making the
absorber something you have to declare.

SELECTIVITY IS THE WHOLE POINT.  Water ice absorbs about four thousand
times less than liquid water at 2.45 GHz, and vapour less again.  That
ratio is what lets a field pick out one state of one species in a vessel
full of things that ignore it -- and it is also why the process runs
away once melting starts, because the product of melting absorbs far
harder than the feedstock.  Both consequences fall out of the same
number and neither is coded in specially.

WHAT IS APPROXIMATE HERE, DECLARED.  A body is treated as a
PERTURBATION: it adds loss to the mode without moving the resonance.
That holds while the body is small or low-permittivity, and breaks for a
large high-permittivity load, which would pull the resonance badly.
:meth:`DielectricLoad.frequency_pull_fraction` says how far off it is
and :meth:`DielectricLoad.is_perturbation` refuses to guess.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from em_materials import EMMaterial, VACUUM_PERMEABILITY_H_M, \
    VACUUM_PERMITTIVITY_F_M

#: Solid straight to vapour: fusion plus vaporisation, at the triple
#: point.  A sublimating load never passes through liquid, which is the
#: entire reason freeze drying exists -- and the reason it is slow.
WATER_LATENT_HEAT_SUBLIMATION_J_KG = 2.834e6
WATER_LATENT_HEAT_FUSION_J_KG = 3.337e5
WATER_LATENT_HEAT_VAPORISATION_J_KG = 2.501e6


@dataclass(frozen=True)
class DielectricLoad:
    """A body in a cavity that turns field into heat.

    The physics is one line -- time-average dissipation density is
    ``(w eps0 eps'' / 2) |E|^2`` -- and everything else here is about
    being honest about where ``|E|`` came from and over what volume the
    integral was taken.

    A body absorbs in SERIES with the wall in the mode's equivalent
    circuit, because both see the same mode current.  It therefore has
    its own resistance and its own column in the ledger, and no step
    anywhere adds it to the wall's.
    """

    identity: str
    material: EMMaterial
    volume_m3: float
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fills_cavity: bool = False

    def __post_init__(self) -> None:
        if self.volume_m3 <= 0.0:
            raise ValueError(f"{self.identity}: a load needs a volume")

    @property
    def loss_permittivity(self) -> float:
        """``eps'' = eps' tan d``: the only material number that absorbs."""
        return self.material.relative_permittivity * self.material.loss_tangent

    def shape_factor(self, cavity, mode) -> float:
        """``|e|`` at the body, from the cavity's own pattern.

        One for a load that fills the box; otherwise the mode's
        normalised electric amplitude where the body sits.  A body at a
        wall sees nothing, which is why an oven has a turntable.
        """
        if self.fills_cavity:
            return 1.0
        return abs(cavity.electric_shape_at(mode, self.position_m)[1])

    def overlap_m3(self, cavity, mode) -> float:
        """``integral |e|^2 dV`` over the body.

        For a filling load this is the cavity's own ``V/4``.  For a
        smaller one the field is taken as uniform across it, which is a
        SMALL-BODY approximation: it holds while the body is small
        against the half-wavelength over which the pattern varies, and
        overstates a large off-centre body.
        """
        if self.fills_cavity:
            return cavity.electric_overlap_m3(mode)
        return self.shape_factor(cavity, mode) ** 2 * self.volume_m3

    def resistance_ohm(self, cavity, mode) -> float:
        """The series resistance this body puts in the mode loop.

        ``P = (w eps0 eps''/2) |E0|^2 * overlap`` and ``P = |i|^2 R / 2``
        with ``|E0| = |i| w mu / pi``, so the mode current divides out
        and the resistance is a property of body and cavity alone.
        """
        omega = 2.0 * math.pi * mode.frequency_hz
        scale = omega * VACUUM_PERMEABILITY_H_M / math.pi
        return (omega * VACUUM_PERMITTIVITY_F_M * self.loss_permittivity
                * scale ** 2 * self.overlap_m3(cavity, mode))

    def quality_factor(self, cavity, mode) -> float:
        """``w L / R``: the Q this body alone would give the mode.

        For a load that fills the cavity this comes out at exactly
        ``1 / tan d``, which is the textbook dielectric Q -- and is the
        check that the gauge connecting field to circuit is right.
        """
        resistance = self.resistance_ohm(cavity, mode)
        if resistance <= 0.0:
            return math.inf
        omega = 2.0 * math.pi * mode.frequency_hz
        return omega * cavity.mode_inductance_h(mode) / resistance

    def absorbed_w(self, cavity, mode, mode_current_a: complex) -> float:
        return 0.5 * self.resistance_ohm(cavity, mode) * abs(
            complex(mode_current_a)) ** 2

    def field_v_m(self, cavity, mode, mode_current_a: complex) -> float:
        """Peak field at the body, which is what sets its heating."""
        return (cavity.peak_electric_field_v_m(mode, mode_current_a)
                * self.shape_factor(cavity, mode))

    def power_density_w_m3(self, cavity, mode,
                           mode_current_a: complex) -> float:
        """Cross-checks against :meth:`EMMaterial.dielectric_loss_w_m3`.

        That method takes an RMS field, this convention carries peak
        phasors, so the two agree only once the root two is applied --
        which is stated here rather than left for someone to trip on.
        """
        return self.absorbed_w(cavity, mode, mode_current_a) / self.volume_m3

    def filling_factor(self, cavity, mode) -> float:
        return self.overlap_m3(cavity, mode) / cavity.electric_overlap_m3(mode)

    def frequency_pull_fraction(self, cavity, mode) -> float:
        """``-(eps' - 1)/2`` times the filling factor: the perturbation's
        own estimate of how badly it invalidates itself."""
        return (-0.5 * (self.material.relative_permittivity - 1.0)
                * self.filling_factor(cavity, mode))

    def is_perturbation(self, cavity, mode, tolerance: float = 0.01) -> bool:
        """Whether treating this body as loss-only is defensible.

        A load that pulls the resonance by more than a fraction of a
        linewidth is not a perturbation, and its absorbed power computed
        at the unperturbed frequency is wrong.  Asked, not assumed.
        """
        return abs(self.frequency_pull_fraction(cavity, mode)) <= tolerance

    def graph_attributes(self) -> dict:
        return {
            "part_role": "dielectric-load",
            "em_material": self.material.key,
            "volume_m3": float(self.volume_m3),
            "loss_permittivity": self.loss_permittivity,
            "absorbs_field": True,
        }


@dataclass(frozen=True)
class Susceptor(DielectricLoad):
    """A body put in the field ON PURPOSE to heat something else.

    The answer to a load that will not absorb: cryogens, dry powders,
    ice below its transition, anything nonpolar.  A susceptor absorbs
    strongly and is thermally bonded to what needs warming, so the
    microwave heats the susceptor and the susceptor heats the load by
    ordinary conduction.

    ``serves`` and ``conductance_w_k`` are what make it a susceptor
    rather than just another lossy body: it declares what it is heating
    and how well it is coupled to it.  Without those it is a part that
    gets hot for no stated reason.
    """

    serves: str = ""
    conductance_w_k: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.serves:
            raise ValueError(
                f"{self.identity}: a susceptor must declare what it heats; "
                "a lossy body with no stated job is a DielectricLoad")
        if self.conductance_w_k <= 0.0:
            raise ValueError(
                f"{self.identity}: a susceptor needs a thermal conductance "
                "to what it serves, or the heat it absorbs goes nowhere")

    def delivered_temperature_rise_k(self, absorbed_w: float) -> float:
        """Steady-state rise across the bond: ``P / G``.

        The susceptor always runs hotter than what it serves, and by how
        much is the bond's business.  A weak bond means a susceptor that
        glows while its load stays cold.
        """
        return float(absorbed_w) / self.conductance_w_k

    def graph_attributes(self) -> dict:
        declared = super().graph_attributes()
        declared.update({
            "part_role": "susceptor",
            "serves": self.serves,
            "conductance_w_k": float(self.conductance_w_k),
        })
        return declared


@dataclass(frozen=True)
class SublimationDuty:
    """Absorbed power driving a solid straight to vapour.

    Under vacuum a warmed solid does not melt, it sublimes, and the
    energy cost is fusion plus vaporisation together.  That is why this
    is slow and why it is worth doing with a field: the power goes into
    the ice throughout its volume rather than having to conduct in from
    a heated surface.
    """

    species: str = "water"
    latent_heat_j_kg: float = WATER_LATENT_HEAT_SUBLIMATION_J_KG

    def __post_init__(self) -> None:
        if self.latent_heat_j_kg <= 0.0:
            raise ValueError(f"{self.species}: latent heat must be positive")

    def mass_rate_kg_s(self, absorbed_w: float) -> float:
        return max(0.0, float(absorbed_w)) / self.latent_heat_j_kg

    def time_to_clear_s(self, mass_kg: float, absorbed_w: float) -> float:
        """How long a deposit takes to go, at this absorbed power.

        Infinite when nothing is being absorbed, which is the correct
        answer and a more useful one than a division error: a field that
        does not couple never clears the deposit, however long it runs.
        """
        rate = self.mass_rate_kg_s(absorbed_w)
        return math.inf if rate <= 0.0 else float(mass_kg) / rate

    def graph_attributes(self) -> dict:
        return {
            "part_role": "sublimation-duty",
            "species": self.species,
            "latent_heat_j_kg": float(self.latent_heat_j_kg),
        }


@dataclass(frozen=True)
class ApplicatorState:
    """What the inside of an applicator is doing, for anything watching.

    This is the published state the rest of the machine reacts to: a
    thermal system reads the absorbed powers as source terms, a fluid
    system reads the sublimation rate as a vapour source, an interlock
    reads the field strength.  It is a snapshot at one frequency and it
    carries no solver -- whoever is watching does not need one.
    """

    frequency_hz: float
    peak_field_v_m: float
    mode_current_a: float
    absorbed_w: dict          # load identity -> watts
    wall_loss_w: float
    radiated_w: float

    @property
    def total_absorbed_w(self) -> float:
        return sum(self.absorbed_w.values())

    def density_w_m3(self, load: DielectricLoad) -> float:
        return self.absorbed_w.get(load.identity, 0.0) / load.volume_m3

    def selectivity(self, load: DielectricLoad, other: DielectricLoad) -> float:
        """How much harder one body is absorbing than another, per volume.

        The number a selective process lives on.  For water ice against
        liquid water it is about four thousand, and that is what lets a
        field find the melt and leave the rest alone.
        """
        denominator = self.density_w_m3(other)
        return math.inf if denominator <= 0.0 else (
            self.density_w_m3(load) / denominator)

    def graph_attributes(self) -> dict:
        return {
            "part_role": "applicator-state",
            "frequency_hz": float(self.frequency_hz),
            "peak_field_v_m": float(self.peak_field_v_m),
            "total_absorbed_w": self.total_absorbed_w,
            "wall_loss_w": float(self.wall_loss_w),
            "radiated_w": float(self.radiated_w),
        }


__all__ = [
    "WATER_LATENT_HEAT_SUBLIMATION_J_KG", "WATER_LATENT_HEAT_FUSION_J_KG",
    "WATER_LATENT_HEAT_VAPORISATION_J_KG",
    "DielectricLoad", "Susceptor", "SublimationDuty", "ApplicatorState",
]
