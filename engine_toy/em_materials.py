"""What a material does to a field, keyed on what a part is already made of.

Every part in this tree already declares a ``material``.  Nothing has yet
asked what that material does electromagnetically, because nothing could:
there was no field.  This adds the three numbers that answer it --
conductivity, permittivity, permeability -- on the SAME keys the rest of
the tree uses, so a part that says it is stainless steel acquires a skin
depth without being told anything new.

THE THREE NUMBERS, AND WHY EACH MATTERS SEPARATELY

  sigma   conductivity.  Sets skin depth, and therefore whether a wall
          is a shield or a suggestion at a given frequency.  Spans
          twenty orders of magnitude across this table, which is why a
          single "conductive / not conductive" flag would be useless.
  eps_r   relative permittivity.  For an insulator this and the loss
          tangent say how much field gets in and how much of it becomes
          heat; dielectric heating is ``omega eps0 eps_r tan_d |E|^2``.
  mu_r    relative permeability.  Near one for almost everything, and
          emphatically not for the ferromagnetics -- which is the whole
          reason a low-frequency magnetic shield is mu-metal and not
          copper, however much better copper conducts.

WHAT IS DELIBERATELY NOT HERE.  These are room-temperature,
low-field, mid-frequency figures.  Permeability falls with frequency and
collapses near saturation; conductivity falls with temperature; a loss
tangent is a curve, not a constant.  Where a part knows better it should
override on its own node, exactly as ``thermal_domains`` lets a node
override its thermal properties.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

#: Vacuum, in SI.  Everything below is relative to these.
VACUUM_PERMEABILITY_H_M = 4.0e-7 * math.pi
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
COPPER_CONDUCTIVITY_S_M = 5.96e7


@dataclass(frozen=True)
class EMMaterial:
    """One material's electromagnetic properties.

    ``conductivity_s_m`` of zero means an insulator, for which the
    permittivity and loss tangent are the interesting numbers instead.
    """

    key: str
    label: str
    conductivity_s_m: float
    relative_permittivity: float = 1.0
    relative_permeability: float = 1.0
    loss_tangent: float = 0.0
    spec: str = ""
    note: str = ""

    @property
    def is_conductor(self) -> bool:
        return self.conductivity_s_m > 1.0

    @property
    def conductivity_relative_to_copper(self) -> float:
        return self.conductivity_s_m / COPPER_CONDUCTIVITY_S_M

    def skin_depth_m(self, frequency_hz: float) -> float:
        """``1 / sqrt(pi f mu sigma)``.

        The depth at which a field entering a conductor has fallen to
        ``1/e``.  It is the number that decides whether a given wall
        thickness is a shield: copper at 1 MHz is 66 microns, so a
        millimetre sheet is fifteen skin depths and effectively opaque,
        while the same sheet at 50 Hz is a third of one skin depth and
        barely notices the field passing through.

        An insulator has no skin depth; the answer is infinite rather
        than an error, because "the field goes straight through" is the
        correct statement about an insulator.
        """
        if not self.is_conductor or frequency_hz <= 0.0:
            return math.inf
        permeability = self.relative_permeability * VACUUM_PERMEABILITY_H_M
        return 1.0 / math.sqrt(
            math.pi * float(frequency_hz) * permeability * self.conductivity_s_m)

    def surface_resistance_ohm(self, frequency_hz: float) -> float:
        """``1 / (sigma delta)``: the ohms per square a wave sees.

        This is what sets a cavity's Q and what turns a shield's
        absorbed field into heat.
        """
        depth = self.skin_depth_m(frequency_hz)
        if not math.isfinite(depth):
            return 0.0
        return 1.0 / (self.conductivity_s_m * depth)

    def dielectric_loss_w_m3(self, frequency_hz: float, field_v_m: float) -> float:
        """``omega eps0 eps_r tan_d |E|^2`` -- volumetric heating.

        The term that makes a lossy dielectric warm in a field, and the
        one a thermal system wants as a source density.
        """
        omega = 2.0 * math.pi * float(frequency_hz)
        return (omega * VACUUM_PERMITTIVITY_F_M * self.relative_permittivity
                * self.loss_tangent * float(field_v_m) ** 2)

    def graph_attributes(self) -> dict:
        return {
            "em_material": self.key,
            "conductivity_s_m": float(self.conductivity_s_m),
            "relative_permittivity": float(self.relative_permittivity),
            "relative_permeability": float(self.relative_permeability),
            "loss_tangent": float(self.loss_tangent),
            "em_conductor": self.is_conductor,
        }


#: Keyed on the material names parts in this tree already declare, so a
#: node that says ``material="stainless-steel"`` gains these for free.
EM_MATERIALS: dict[str, EMMaterial] = {
    "copper": EMMaterial(
        "copper", "annealed copper", 5.96e7, spec="IACS 100%",
        note="the reference every other conductor is quoted against"),
    "aluminium": EMMaterial(
        "aluminium", "aluminium alloy", 3.5e7,
        note="three fifths of copper's conductivity at a third of its "
             "density, which is why an enclosure is aluminium and a "
             "busbar is copper"),
    "aluminium-casting": EMMaterial(
        "aluminium-casting", "cast aluminium", 2.4e7,
        note="the silicon that makes it castable also scatters electrons; "
             "a die-cast box shields well but conducts worse than sheet"),
    "steel-plate": EMMaterial(
        "steel-plate", "mild steel plate", 1.0e7, relative_permeability=300.0,
        note="poor conductor, strongly magnetic: the permeability more "
             "than makes up for the conductivity in a shield, because "
             "skin depth falls with the product of the two"),
    "stainless-steel": EMMaterial(
        "stainless-steel", "304 stainless", 1.45e6, relative_permeability=1.008,
        note="forty times worse than copper and essentially non-magnetic; "
             "chosen for corrosion and strength, never for shielding"),
    "mu-metal": EMMaterial(
        "mu-metal", "80% nickel-iron, annealed", 1.6e6,
        relative_permeability=50000.0, spec="ASTM A753 Type 4",
        note="the only practical answer to a LOW-FREQUENCY magnetic field. "
             "Copper conducts thirty times better and shields it far "
             "worse, because at mains frequency the field is magnetic and "
             "what stops it is permeability, not conductivity"),
    "brass": EMMaterial("brass", "cartridge brass", 1.5e7),
    "battery-module": EMMaterial(
        "battery-module", "cell stack, effective", 1.0e5,
        note="effective bulk figure for a pack of cells, cans and busbars; "
             "not a material property, an assembly one"),
    # --- insulators, where permittivity and loss are the story
    "air": EMMaterial("air", "dry air", 0.0, 1.00059, 1.0, 0.0),
    "ptfe": EMMaterial(
        "ptfe", "PTFE", 0.0, 2.1, 1.0, 2.0e-4,
        note="the low-loss reference for RF insulation"),
    "polyethylene": EMMaterial("polyethylene", "polyethylene", 0.0, 2.25, 1.0, 3.0e-4),
    "fr4": EMMaterial(
        "fr4", "FR-4 laminate", 0.0, 4.35, 1.0, 0.02,
        note="what a board is; the loss tangent is why FR-4 stops being "
             "the right substrate somewhere above a gigahertz"),
    "glass": EMMaterial("glass", "soda-lime glass", 0.0, 7.0, 1.0, 0.01),
    "water": EMMaterial(
        "water", "liquid water, 20 C", 5.5e-6, 80.1, 1.0, 0.157,
        note="high permittivity and a large loss tangent: it couples to a "
             "field strongly and turns what it absorbs into heat, which is "
             "why wet things warm in a field and dry ones do not"),
    "ceramic-alumina": EMMaterial("ceramic-alumina", "96% alumina", 0.0, 9.4, 1.0, 1.0e-4),
    "water-ice": EMMaterial(
        "water-ice", "water ice, -12 C", 0.0, 3.2, 1.0, 9.0e-4,
        note="the same molecule as liquid water and four thousand times "
             "less absorbing: locked in the lattice it cannot rotate to "
             "follow the field. This single fact is why defrosting runs "
             "away -- the first film to melt absorbs far harder than the "
             "ice around it"),
    "water-vapour": EMMaterial(
        "water-vapour", "water vapour, 1 bar", 0.0, 1.0006, 1.0, 1.0e-5,
        note="too dilute and too free to couple; steam is effectively "
             "transparent, so a field passes through what has already "
             "boiled and keeps working on what has not"),
    "water-hot": EMMaterial(
        "water-hot", "liquid water, 80 C", 2.0e-4, 60.0, 1.0, 0.055,
        note="hot water absorbs LESS than cold at 2.45 GHz: warming speeds "
             "the dipole's relaxation past the drive, so the coupling is "
             "mildly self-limiting rather than a runaway"),
    "brine": EMMaterial(
        "brine", "saline water, 20 C, ~5 S/m", 5.0, 78.0, 1.0, 0.63,
        note="dissolved ions add sigma/(omega eps0) to the loss, which at "
             "2.45 GHz roughly triples what pure water does; a drain's "
             "contents are never pure water and it matters"),
}


#: Measured anchors at 2.45 GHz, not a model.  There is no single closed
#: form covering both liquid water and ice here: liquid follows Debye
#: relaxation, while ice's relaxation sits near a kilohertz and its
#: microwave loss comes from a different mechanism entirely -- a Debye
#: tail under-predicts ice by about four orders of magnitude.  So the
#: states are declared separately, each from measurement, and the phase
#: is chosen by the caller rather than inferred from temperature.
WATER_EM_STATES = {
    "ice": ("water-ice", 261.15),
    "liquid": ("water", 293.15),
    "liquid-hot": ("water-hot", 353.15),
    "vapour": ("water-vapour", 373.15),
    "brine": ("brine", 293.15),
}

#: The range the liquid anchors were measured over.  Outside it the
#: table is refused rather than extrapolated.
WATER_LIQUID_RANGE_K = (273.15, 373.15)


def water_em(state: str, temperature_k: float | None = None) -> EMMaterial:
    """The electromagnetic properties of water in a DECLARED state.

    ``state`` comes from the chemistry side -- it is the phase the
    species is actually in -- and is never guessed from temperature,
    because supercooled water, superheated ice and metastable states are
    all real and all have the properties of the phase they are in rather
    than the one their temperature suggests.

    For liquid, two measured anchors bracket the range and the result is
    interpolated linearly between them in permittivity and in loss.
    That is an interpolation between measurements, stated as such; it is
    not a model, and outside the bracket it is refused.
    """
    if state not in WATER_EM_STATES:
        raise KeyError(
            f"{state!r} is not a declared water state; the table holds "
            f"{sorted(WATER_EM_STATES)}")
    key, anchor_k = WATER_EM_STATES[state]
    material = EM_MATERIALS[key]
    if temperature_k is None or state not in ("liquid", "liquid-hot"):
        return material

    low, high = WATER_LIQUID_RANGE_K
    if not low <= float(temperature_k) <= high:
        raise ValueError(
            f"liquid water is tabulated here only between {low} K and "
            f"{high} K; {temperature_k} K is outside the measured range "
            "and this table will not extrapolate into it")
    cold, hot = EM_MATERIALS["water"], EM_MATERIALS["water-hot"]
    span = 353.15 - 293.15
    fraction = min(1.0, max(0.0, (float(temperature_k) - 293.15) / span))
    permittivity = (cold.relative_permittivity * (1.0 - fraction)
                    + hot.relative_permittivity * fraction)
    loss = (cold.relative_permittivity * cold.loss_tangent * (1.0 - fraction)
            + hot.relative_permittivity * hot.loss_tangent * fraction)
    return EMMaterial(
        f"water@{float(temperature_k):.0f}K",
        f"liquid water, {float(temperature_k) - 273.15:.0f} C",
        cold.conductivity_s_m, permittivity, 1.0, loss / permittivity,
        note="interpolated between the 20 C and 80 C measured anchors")


def material_for(node: dict) -> EMMaterial | None:
    """The EM material a graph node implies, or ``None`` if it says nothing.

    Reads ``em_material`` first, because a part may declare its
    electromagnetic identity separately from its structural one -- a
    plated plastic housing is polymer to the stress solver and copper to
    the field.  Otherwise it falls back to ``material``, which is how an
    existing part gains field behaviour without being edited.
    """
    for field_name in ("em_material", "material"):
        key = node.get(field_name)
        if key in EM_MATERIALS:
            return EM_MATERIALS[key]
    return None


__all__ = [
    "VACUUM_PERMEABILITY_H_M", "VACUUM_PERMITTIVITY_F_M",
    "COPPER_CONDUCTIVITY_S_M", "EMMaterial", "EM_MATERIALS", "material_for",
    "WATER_EM_STATES", "WATER_LIQUID_RANGE_K", "water_em",
]
