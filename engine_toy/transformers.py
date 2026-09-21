"""Iron-core transformers, including the shunted kind that limits its own current.

WHAT A SHUNT IS FOR.  An ordinary power transformer is built to couple
its windings as tightly as it can: leakage is a defect, and a stiff
secondary voltage is the product.  A *magnetically shunted* transformer
is built the other way on purpose.  Steel shunts are set in the window
between primary and secondary, offering the flux a path that links one
winding and not the other, and the resulting leakage reactance is large
enough to dominate the secondary circuit.  The machine then behaves as a
CURRENT source: short the secondary and the current settles at
``V_oc / (omega L_leak)`` instead of rising until something fails.

That is the whole reason this is its own object rather than a turns
ratio.  A load that cannot regulate itself -- a welding arc, a magnetron
-- needs the supply to do the regulating, and the shunt is how the
supply does it.  Model such a transformer as ideal and the current is
bounded by nothing but the winding resistance, which is both wrong and
uninformative.

THE NUMBERS ARE RELATED, NOT ASSERTED.  A winding's turns, the core's
cross-section and the supply frequency together fix the peak flux
density through the transformer EMF equation

    V_rms = 4.44 f N A B_peak

so a declared design either fits inside its steel's saturation or it does
not, and :meth:`IronCoreTransformer.design_notes` says which.  Cheap
transformers are run deliberately close to saturation to save iron; that
is why they are noisy, lossy and hot, and the model should reproduce
that rather than hide it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from electrical_distribution import ElectricalService


# Laminated electrical steels.  Specific loss is quoted the way the
# industry quotes it: watts per kilogram at a stated peak flux and
# frequency, from which other operating points are scaled below.
@dataclass(frozen=True)
class CoreLamination:
    key: str
    label: str
    saturation_tesla: float
    density_kg_m3: float
    #: W/kg at ``loss_reference_tesla`` and ``loss_reference_hz``
    specific_loss_w_kg: float
    loss_reference_tesla: float = 1.5
    loss_reference_hz: float = 50.0
    spec: str = ""
    note: str = ""


CORE_STEELS: dict[str, CoreLamination] = {
    "m19": CoreLamination(
        "m19", "M19 non-oriented silicon steel", 1.95, 7650.0, 2.4,
        spec="ASTM A677 36F145",
        note="the ordinary choice for small mains transformers and motor "
             "laminations: cheap, isotropic, and tolerant of the flux "
             "wandering out of the rolling direction at a corner"),
    "m6": CoreLamination(
        "m6", "M6 grain-oriented silicon steel", 2.03, 7650.0, 1.1,
        spec="ASTM A876",
        note="half the loss of M19 along the rolling direction and worse "
             "across it; worth its price only where the flux path is "
             "straight, which an E-I core's corners are not"),
    "supermalloy": CoreLamination(
        "supermalloy", "80% nickel-iron", 0.78, 8700.0, 0.3,
        spec="ASTM A753 Type 4",
        note="saturates early and costs a fortune; it is here because a "
             "shield or an instrument transformer wants permeability, not "
             "flux capacity"),
}


@dataclass(frozen=True)
class Winding:
    """One winding: how many turns, of what wire, for what.

    ``role`` is declared rather than inferred from the turns count,
    because a filament winding and a control winding may have the same
    handful of turns and mean entirely different things.
    """

    role: str                      # primary | secondary | filament | control
    turns: int
    conductor_area_mm2: float
    mean_turn_length_m: float
    rated_current_a: float
    #: 20 C copper; windings run hot, and ``resistance_ohm_at`` corrects it
    resistivity_ohm_m: float = 1.724e-8

    def __post_init__(self) -> None:
        if self.turns <= 0:
            raise ValueError(f"{self.role} winding needs a positive turn count")
        if self.conductor_area_mm2 <= 0.0:
            raise ValueError(f"{self.role} winding needs a conductor area")

    @property
    def conductor_length_m(self) -> float:
        return self.turns * self.mean_turn_length_m

    @property
    def resistance_ohm(self) -> float:
        return (self.resistivity_ohm_m * self.conductor_length_m
                / (self.conductor_area_mm2 * 1e-6))

    def resistance_ohm_at(self, temperature_c: float) -> float:
        """Copper gains about 0.393% per kelvin, which is not a detail.

        A winding at 130 C carries a third more resistance than the
        table value, so copper loss -- and therefore the temperature --
        is a feedback loop rather than a constant.
        """
        return self.resistance_ohm * (1.0 + 0.00393 * (temperature_c - 20.0))

    @property
    def current_density_a_mm2(self) -> float:
        return self.rated_current_a / self.conductor_area_mm2

    def graph_attributes(self) -> dict:
        return {
            "winding_role": self.role,
            "turns": int(self.turns),
            "conductor_area_mm2": float(self.conductor_area_mm2),
            "winding_resistance_ohm": float(self.resistance_ohm),
            "rated_current_a": float(self.rated_current_a),
            "current_density_a_mm2": float(self.current_density_a_mm2),
        }


@dataclass(frozen=True)
class IronCoreTransformer:
    """One transformer, with or without current-limiting shunts.

    ``leakage_inductance_h`` is referred to the SECONDARY, because that
    is the side whose current it limits and the side a datasheet quotes.
    Leaving it ``None`` describes an ordinary tightly-coupled
    transformer, whose secondary current is then bounded by the load and
    the winding resistance alone.
    """

    identity: str
    service: ElectricalService
    core: str
    core_mass_kg: float
    core_area_m2: float
    primary: Winding
    secondary: Winding
    filament: Winding | None = None
    leakage_inductance_h: float | None = None
    ambient_c: float = 25.0
    label: str = "transformer"

    def __post_init__(self) -> None:
        if self.core not in CORE_STEELS:
            raise ValueError(f"unknown core steel: {self.core}")
        if self.service.frequency_hz <= 0.0:
            raise ValueError("a transformer needs an AC service")
        if self.core_area_m2 <= 0.0 or self.core_mass_kg <= 0.0:
            raise ValueError("a core needs a positive area and mass")

    # ---- magnetics -------------------------------------------------

    @property
    def lamination(self) -> CoreLamination:
        return CORE_STEELS[self.core]

    @property
    def primary_voltage_v(self) -> float:
        """What the primary actually sees, which is not always line voltage.

        A split-phase service supplies 120 V line-to-neutral and 240 V
        line-to-line; a small appliance takes the former.
        """
        if self.service.line_neutral_voltage_v is not None:
            return float(self.service.line_neutral_voltage_v)
        return float(self.service.line_voltage_v)

    @property
    def peak_flux_density_tesla(self) -> float:
        """``B = V / (4.44 f N A)`` -- the transformer EMF equation."""
        return self.primary_voltage_v / (
            4.44 * self.service.frequency_hz
            * self.primary.turns * self.core_area_m2)

    @property
    def saturation_margin(self) -> float:
        """Peak flux as a fraction of the steel's saturation."""
        return self.peak_flux_density_tesla / self.lamination.saturation_tesla

    @property
    def turns_ratio(self) -> float:
        return self.secondary.turns / self.primary.turns

    @property
    def open_circuit_secondary_v(self) -> float:
        return self.primary_voltage_v * self.turns_ratio

    @property
    def filament_voltage_v(self) -> float | None:
        if self.filament is None:
            return None
        return self.primary_voltage_v * self.filament.turns / self.primary.turns

    # ---- what the shunts do ----------------------------------------

    @property
    def leakage_reactance_ohm(self) -> float | None:
        if self.leakage_inductance_h is None:
            return None
        return (2.0 * math.pi * self.service.frequency_hz
                * self.leakage_inductance_h)

    @property
    def short_circuit_secondary_a(self) -> float | None:
        """The current a shorted secondary settles at, not runs away to.

        This is the number the shunt exists to produce.  Without it the
        only limit is winding resistance, which for a high-voltage
        secondary is a very large current indeed.
        """
        reactance = self.leakage_reactance_ohm
        if reactance is None:
            return None
        impedance = math.hypot(reactance, self.secondary.resistance_ohm)
        return self.open_circuit_secondary_v / impedance

    def secondary_voltage_at(self, load_current_a: float) -> float:
        """Terminal voltage sagging under load, through the leakage drop.

        Phasor, not arithmetic: the drop is taken in quadrature with the
        open-circuit EMF, so the terminal voltage follows a CIRCLE --
        ``V^2 + (Z I)^2 = V_oc^2`` -- holding up under light load and
        collapsing only as the current approaches the short-circuit
        value.  That shape is the regulation a shunted supply is built
        to provide.

        The winding resistance is folded into ``Z`` alongside the
        reactance, which is a good approximation only while the
        reactance dominates.  In a shunted transformer it does, by a
        wide margin -- 2639 against 60 ohms in the oven transformer
        this was written for.  For a tightly coupled transformer, where
        resistance is most of the impedance, this overstates the sag and
        the in-phase drop should be carried separately.
        """
        reactance = self.leakage_reactance_ohm or 0.0
        drop = math.hypot(reactance, self.secondary.resistance_ohm) * load_current_a
        remaining = self.open_circuit_secondary_v ** 2 - drop ** 2
        return math.sqrt(remaining) if remaining > 0.0 else 0.0

    # ---- losses, which become heat ---------------------------------

    @property
    def core_loss_w(self) -> float:
        """Steinmetz-scaled from the lamination's quoted operating point.

        Loss climbs with roughly the square of flux density and the
        1.5 power of frequency; running a cheap core near saturation to
        save iron is paid for here, every second it is energised.
        """
        steel = self.lamination
        flux_ratio = self.peak_flux_density_tesla / steel.loss_reference_tesla
        frequency_ratio = self.service.frequency_hz / steel.loss_reference_hz
        return (steel.specific_loss_w_kg * self.core_mass_kg
                * flux_ratio ** 2 * frequency_ratio ** 1.5)

    def copper_loss_w(self, secondary_current_a: float,
                      winding_temperature_c: float = 100.0) -> float:
        """``I^2 R`` in both windings, with the primary current reflected."""
        primary_current = secondary_current_a * self.turns_ratio
        return (primary_current ** 2
                * self.primary.resistance_ohm_at(winding_temperature_c)
                + secondary_current_a ** 2
                * self.secondary.resistance_ohm_at(winding_temperature_c))

    def heat_w(self, secondary_current_a: float,
               winding_temperature_c: float = 100.0) -> float:
        """Everything that does not leave as electrical power.

        This is the number a thermal system wants as a source term.
        """
        return self.core_loss_w + self.copper_loss_w(
            secondary_current_a, winding_temperature_c)

    # ---- is this design admissible? --------------------------------

    def design_notes(self) -> list[str]:
        """Complaints a builder would make about this design, if any."""
        notes: list[str] = []
        margin = self.saturation_margin
        if margin >= 1.0:
            notes.append(
                f"core saturates: {self.peak_flux_density_tesla:.2f} T against "
                f"{self.lamination.saturation_tesla:.2f} T for "
                f"{self.lamination.label}; the magnetising current will be "
                "peaky and the transformer will buzz and overheat")
        elif margin > 0.8:
            # Silicon steel's B-H curve bends well before nominal
            # saturation -- M19's knee is around 1.5 to 1.6 T against a
            # 1.95 T saturation, so roughly four fifths.  Past the knee
            # the magnetising current stops being sinusoidal and starts
            # being peaky, which is audible and expensive long before
            # anything is technically saturated.
            notes.append(
                f"core runs at {margin:.0%} of saturation, past the knee of "
                "the B-H curve: this is how a cheap transformer saves iron, "
                "and why it buzzes and runs hot")
        for winding in (self.primary, self.secondary, self.filament):
            if winding is None:
                continue
            if winding.current_density_a_mm2 > 5.0:
                notes.append(
                    f"{winding.role} winding at "
                    f"{winding.current_density_a_mm2:.1f} A/mm2 needs forced "
                    "cooling or a short duty cycle")
        if self.leakage_inductance_h is None:
            notes.append(
                "no shunts declared: this transformer does not limit its own "
                "secondary current, so its load must")
        return notes

    def graph_attributes(self) -> dict:
        return {
            "part_role": "transformer",
            "core_steel": self.core,
            "core_mass_kg": float(self.core_mass_kg),
            "core_area_m2": float(self.core_area_m2),
            "peak_flux_density_tesla": float(self.peak_flux_density_tesla),
            "saturation_margin": float(self.saturation_margin),
            "turns_ratio": float(self.turns_ratio),
            "open_circuit_secondary_v": float(self.open_circuit_secondary_v),
            "short_circuit_secondary_a": (
                None if self.short_circuit_secondary_a is None
                else float(self.short_circuit_secondary_a)),
            "leakage_inductance_h": (
                None if self.leakage_inductance_h is None
                else float(self.leakage_inductance_h)),
            "core_loss_w": float(self.core_loss_w),
            "current_limiting": self.leakage_inductance_h is not None,
            **self.service.graph_attributes(),
        }


__all__ = [
    "CoreLamination", "CORE_STEELS", "Winding", "IronCoreTransformer",
]
