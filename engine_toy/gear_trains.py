"""Real gear trains: every rotating moment inside a case, specified.

A gearbox, a transfer case, a differential and a torque converter are
all the same class of object as an alternator -- a casing that does not
turn, around groups that do -- but with the one feature that makes them
worth their own module: their groups turn at DIFFERENT speeds from each
other, and some of those speeds change with the engaged gear. That is
the whole reason `RotatingGroup` carries a `speed_ratio` and reflects
itself through the square of it.

WHAT THIS MODULE IS FOR. Until now `powertrain.transmission` and
`powertrain.transfer_case` carried `mass_kg=0.0` and no inertia at all,
so the drivetrain solver saw a gearbox as one-millionth of a kg*m^2 --
less than a spark plug. The honest fix is not to invent a lumped number
for it but to say what is actually in there, because a gear case's
inertia is not a property of the case: it is a property of the case AND
the gear currently engaged, and no single number can be both.

THE FACT THAT MATTERS MOST. In a real constant-mesh gearbox, the input
shaft, the layshaft and every gear riding on the mainshaft turn
whenever the clutch is engaged -- INCLUDING IN NEUTRAL. A gearbox in
neutral is not disconnected from the engine; it is a real rotating
inertia the engine must accelerate through every idle fluctuation, and
only the mainshaft and everything downstream of it is free. An engine
that cannot hold an idle in neutral is being asked to swing that
inertia, which is exactly the thing a model with `mass_kg=0.0` cannot
tell you.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rotating_inertia import HousedAssembly, RotatingGroup


# --- manual gearbox --------------------------------------------------
#
# Real mass split of a manual transmission. The case, covers, shift
# mechanism and bellhousing flange are well over half the unit and never
# turn; the layshaft cluster is the single heaviest rotating piece,
# because it is a solid forged cluster of four or five gears.
_MANUAL_CASE_FRAC = 0.58
_MANUAL_INPUT_FRAC = 0.07
_MANUAL_LAYSHAFT_FRAC = 0.20
_MANUAL_MAINSHAFT_FRAC = 0.15
# Gear pitch radius as a fraction of the case's own outer radius -- real
# centre distances put the meshes well inside the casting.
_GEAR_PITCH_RADIUS_FRAC = 0.30
# Real constant-mesh (input-to-layshaft) ratios sit between about 1.2
# and 2.5. In a box with a direct-drive top gear the lowest ratio is the
# product of the constant mesh and the low gear pair, and a real design
# splits that reduction roughly evenly between the two meshes -- so the
# square root of the lowest ratio is the real derivation, not a guess.
_CONSTANT_MESH_MIN = 1.2
_CONSTANT_MESH_MAX = 2.5


def constant_mesh_ratio(gear_ratios) -> float:
    """Layshaft speed / input speed, from the box's own declared ratios."""
    ratios = [float(r) for r in (gear_ratios or ()) if float(r) > 0.0]
    if not ratios:
        return 1.5
    lowest = max(ratios)
    return min(_CONSTANT_MESH_MAX, max(_CONSTANT_MESH_MIN, math.sqrt(lowest)))


@dataclass(frozen=True)
class Gearbox:
    """A constant-mesh manual gearbox, with every moment named."""
    unit_mass_kg: float
    case_radius_m: float
    gear_ratios: tuple[float, ...]
    reverse_ratio: float

    @property
    def pitch_radius_m(self) -> float:
        return max(1e-3, self.case_radius_m * _GEAR_PITCH_RADIUS_FRAC)

    @property
    def constant_mesh(self) -> float:
        return constant_mesh_ratio(self.gear_ratios)

    def groups(self, engaged_ratio: float | None) -> tuple[RotatingGroup, ...]:
        """Every spinning mass in the case, at its real speed.

        `engaged_ratio` is None in neutral. The input shaft and the
        layshaft turn either way -- that is what constant mesh means --
        and only the mainshaft's own speed depends on the gear."""
        m = self.unit_mass_kg
        r = self.pitch_radius_m
        cm = self.constant_mesh
        out = [
            RotatingGroup("input_shaft", "geared-shaft-train", m * _MANUAL_INPUT_FRAC, r, 1.0),
            # the layshaft cluster runs FASTER than the input shaft and
            # is the heaviest rotating piece, so the ratio-squared
            # reflection makes it the dominant moment in most boxes
            RotatingGroup("layshaft_cluster", "geared-shaft-train",
                          m * _MANUAL_LAYSHAFT_FRAC, r, cm),
        ]
        if engaged_ratio is None:
            # NEUTRAL: the mainshaft gears still ride in mesh with the
            # layshaft and still spin -- they are simply not locked to
            # the mainshaft. They are the reason a gearbox in neutral is
            # not free, and they turn at layshaft speed scaled by their
            # own pair, which averages back to roughly input speed.
            out.append(RotatingGroup("mainshaft_free_gears", "geared-shaft-train",
                                     m * _MANUAL_MAINSHAFT_FRAC, r, 1.0))
        else:
            # IN GEAR: the mainshaft and everything splined to it turn
            # at output speed, which is input speed divided by the
            # engaged ratio.
            er = float(engaged_ratio) or 1.0
            out.append(RotatingGroup("mainshaft", "geared-shaft-train",
                                     m * _MANUAL_MAINSHAFT_FRAC, r, 1.0 / er))
        return tuple(out)

    def assembly(self, engaged_ratio: float | None = None) -> HousedAssembly:
        return HousedAssembly(kind="manual-transmission", unit_mass_kg=self.unit_mass_kg,
                              groups=self.groups(engaged_ratio))

    def inertia_at_input(self, engaged_ratio: float | None = None) -> float:
        return self.assembly(engaged_ratio).inertia_kg_m2


# --- differential / final drive --------------------------------------
#
# A differential's two speeds are genuinely different objects: the
# pinion turns at propshaft speed, the crown wheel and carrier turn at
# wheel speed, and the final drive ratio between them is large, so the
# small fast pinion reflects far more inertia than its mass suggests.
_DIFF_CASE_FRAC = 0.52
_DIFF_CROWN_CARRIER_FRAC = 0.33
_DIFF_PINION_FRAC = 0.15


@dataclass(frozen=True)
class Differential:
    unit_mass_kg: float
    case_radius_m: float
    final_drive_ratio: float

    def groups(self) -> tuple[RotatingGroup, ...]:
        """Referred to the PROPSHAFT, which is this unit's input.

        The side gears and spiders are deliberately not their own group:
        running straight ahead they do not turn relative to the carrier
        at all, and only spin on their own pins when the two wheels
        differ -- a real effect, but not one that stores energy along
        the driveline axis."""
        m = self.unit_mass_kg
        r_crown = max(1e-3, self.case_radius_m * 0.62)   # a crown wheel nearly fills the case
        r_pinion = max(1e-3, self.case_radius_m * 0.18)  # the pinion head is small
        fd = float(self.final_drive_ratio) or 1.0
        return (
            RotatingGroup("pinion_and_flange", "geared-shaft-train",
                          m * _DIFF_PINION_FRAC, r_pinion, 1.0),
            RotatingGroup("crown_wheel_and_carrier", "solid-disc",
                          m * _DIFF_CROWN_CARRIER_FRAC, r_crown, 1.0 / fd),
        )

    def assembly(self) -> HousedAssembly:
        return HousedAssembly(kind="differential", unit_mass_kg=self.unit_mass_kg,
                              groups=self.groups())

    def inertia_at_input(self) -> float:
        return self.assembly().inertia_kg_m2


# --- transfer case ---------------------------------------------------
_TCASE_CASE_FRAC = 0.60
_TCASE_INPUT_FRAC = 0.14
_TCASE_OUTPUT_FRAC = 0.16
_TCASE_CHAIN_FRAC = 0.10


@dataclass(frozen=True)
class TransferCase:
    unit_mass_kg: float
    case_radius_m: float
    low_range_ratio: float = 2.72   # real, typical part-time 4WD low range

    def groups(self, low_range: bool = False) -> tuple[RotatingGroup, ...]:
        m = self.unit_mass_kg
        r = max(1e-3, self.case_radius_m * _GEAR_PITCH_RADIUS_FRAC)
        ratio = (1.0 / float(self.low_range_ratio)) if low_range else 1.0
        return (
            RotatingGroup("input_shaft", "geared-shaft-train", m * _TCASE_INPUT_FRAC, r, 1.0),
            RotatingGroup("output_shafts", "geared-shaft-train", m * _TCASE_OUTPUT_FRAC, r, ratio),
            # the drive chain and its sprockets turn with the front
            # output: real mass, and real inertia the engine feels in
            # four-wheel drive that simply is not there in two
            RotatingGroup("chain_and_sprockets", "toothed-sprocket",
                          m * _TCASE_CHAIN_FRAC, r * 1.6, ratio),
        )

    def assembly(self, low_range: bool = False) -> HousedAssembly:
        return HousedAssembly(kind="transfer-case", unit_mass_kg=self.unit_mass_kg,
                              groups=self.groups(low_range))

    def inertia_at_input(self, low_range: bool = False) -> float:
        return self.assembly(low_range).inertia_kg_m2


# --- torque converter ------------------------------------------------
#
# An automatic has no clutch pack on the crank at all: it has a
# converter, and the converter's impeller is BOLTED TO THE CRANK. That
# impeller, its shell, and the oil inside it are permanent crank
# inertia, present at idle in park -- which is the real reason an
# automatic idles differently from a manual and why the flexplate it
# bolts to is lighter than a manual's flywheel.
_CONVERTER_SHELL_IMPELLER_FRAC = 0.55   # shell + impeller: bolted to the crank
_CONVERTER_TURBINE_FRAC = 0.30          # turbine: turns at gearbox input speed
_CONVERTER_STATOR_FRAC = 0.15           # stator: on a one-way clutch


@dataclass(frozen=True)
class TorqueConverter:
    unit_mass_kg: float          # dry mass of the converter
    outer_radius_m: float
    oil_mass_kg: float = 0.0     # the charge it runs full of: real rotating mass

    def groups(self, speed_ratio: float = 0.0) -> tuple[RotatingGroup, ...]:
        """`speed_ratio` is turbine speed / impeller speed: 0 at stall,
        approaching 1 at the coupling point (exactly 1 when locked)."""
        m = self.unit_mass_kg
        r = max(1e-3, float(self.outer_radius_m))
        sr = min(1.0, max(0.0, float(speed_ratio)))
        # the oil rides with the impeller shell that slings it
        impeller = m * _CONVERTER_SHELL_IMPELLER_FRAC + max(0.0, float(self.oil_mass_kg))
        return (
            RotatingGroup("shell_and_impeller", "rim-weighted-flywheel", impeller, r, 1.0),
            RotatingGroup("turbine", "rim-weighted-flywheel",
                          m * _CONVERTER_TURBINE_FRAC, r * 0.94, sr),
            # the stator freewheels below the coupling point and locks
            # to the case above it: either way it is not turning at
            # impeller speed, and at stall it is not turning at all
            RotatingGroup("stator", "solid-disc", m * _CONVERTER_STATOR_FRAC, r * 0.45,
                          0.0 if sr < 0.86 else sr),
        )

    def assembly(self, speed_ratio: float = 0.0) -> HousedAssembly:
        return HousedAssembly(kind="torque-converter",
                              unit_mass_kg=self.unit_mass_kg + max(0.0, self.oil_mass_kg),
                              groups=self.groups(speed_ratio))

    def inertia_at_crank(self, speed_ratio: float = 0.0) -> float:
        return self.assembly(speed_ratio).inertia_kg_m2
