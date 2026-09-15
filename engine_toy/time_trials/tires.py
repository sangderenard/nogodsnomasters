"""The cheap tire, used as a tire and not as a spring constant.

`vehicle_tire_reduced_contact_law` is the bottom rung of turing's tire
fidelity ladder: contact patch area integrated from the tire's OWN four
ring stations against a flat ground plane, times its own inflation
pressure. That is the standard pneumatic load relation -- the same one
load-capacity tables are built from -- rather than a stiffness someone
picked. It is the mode the ladder's own docstring says every ordinary
rig run should use, which is exactly what a racer is.

We take it symbolically and lambdify it here. That is deliberate and
temporary: `compile_reduced_contact_law_ssa()` is the same equations
routed into repository SSA, and swapping this call for that one is the
only change needed to put the tire on the native lane. Keeping both
reachable from one place is why this module exists.

WHAT IT GIVES A RACER. Normal load per corner, from how far that corner
is pressed into the ground -- so weight transfer under braking and
cornering is an OUTPUT of the geometry rather than a fudge, and grip
follows the load the tire actually carries. A corner that unloads
carries no patch, so it has no grip, and it says so by returning zero
area rather than by a special case.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

#: engine_toy itself, so `graph_physics` (which puts turing on the path)
#: is importable however this package was entered.
_ENGINE_TOY = str(Path(__file__).resolve().parent.parent)
if _ENGINE_TOY not in sys.path:
    sys.path.insert(0, _ENGINE_TOY)

import sympy


def _law():
    """Lambdify the reduced contact law once, in its declared argument order."""
    import graph_physics  # noqa: F401  -- puts turing on sys.path
    from src.compiler.vehicle_tire_reduced_contact_law import (
        symbolic_reduced_contact_law_equations)

    equations, symbols_by_name = symbolic_reduced_contact_law_equations()
    area = next(e for e in equations
                if str(e.lhs) == "reduced_contact_patch_area_m2").rhs
    force = next(e for e in equations
                 if str(e.lhs) == "reduced_contact_force_n").rhs
    order = tuple(sorted(symbols_by_name))
    args = [symbols_by_name[name] for name in order]
    return order, sympy.lambdify(args, [area, force], "math")


_ORDER, _EVALUATE = _law()


@dataclass(frozen=True)
class TireSection:
    """One tire's rest cross-section, as the four ring stations.

    Radii and axial offsets in metres. The defaults are the light-truck
    section turing's own verification fixture uses, scaled by
    `scaled_to`, so nothing here is a number invented for a game.
    """

    bead_r: float = 0.28
    bead_z: float = 0.11
    shoulder_r: float = 0.38
    shoulder_z: float = 0.09
    pressure_pa: float = 2.4e5

    def scaled_to(self, radius_m: float, width_m: float, pressure_pa: float) -> "TireSection":
        """The same SHAPE at another size -- proportions preserved."""
        r_scale = radius_m / self.shoulder_r
        z_scale = (width_m * 0.5) / self.bead_z
        return TireSection(
            bead_r=self.bead_r * r_scale, bead_z=self.bead_z * z_scale,
            shoulder_r=self.shoulder_r * r_scale, shoulder_z=self.shoulder_z * z_scale,
            pressure_pa=float(pressure_pa),
        )

    @property
    def stations(self) -> dict[str, float]:
        return {
            "bead_inboard_r": self.bead_r, "bead_inboard_z": -self.bead_z,
            "shoulder_inboard_r": self.shoulder_r, "shoulder_inboard_z": -self.shoulder_z,
            "shoulder_outboard_r": self.shoulder_r, "shoulder_outboard_z": self.shoulder_z,
            "bead_outboard_r": self.bead_r, "bead_outboard_z": self.bead_z,
        }

    @property
    def max_compression_m(self) -> float:
        """Travel available before the ground reaches the bead -- the rim.

        The law's own note draws the line here: it is accurate until the
        ground plane approaches a BEAD radius, which is the tire crushed
        onto its rim and not an operating point. So this is the tire's
        real usable compression, read off its own section.
        """
        return max(0.0, self.shoulder_r - self.bead_r)

    def contact(self, compression_m: float) -> tuple[float, float]:
        """(patch area m2, normal load N) at this compression.

        NOTE ON THE LAW'S ARGUMENT. `compression_depth_m` in
        `vehicle_tire_reduced_contact_law` is the height of the spin AXIS
        above the ground plane, not the penetration: the chord width
        ``2*sqrt(r(z)**2 - depth**2)`` is the width of a disk of radius
        r(z) cut by a plane that far from its centre. So the tire first
        touches at ``depth == shoulder_r`` and reaches its rim at
        ``depth == bead_r``. Passing a penetration straight in reads as a
        nearly flat tire and returns most of the carcass as patch, which
        is how this was caught. We convert here, once.
        """
        travel = self.max_compression_m
        compression = max(0.0, min(float(compression_m), travel))
        if compression <= 0.0:
            return 0.0, 0.0
        axle_height = self.shoulder_r - compression
        values = dict(self.stations)
        values["compression_depth_m"] = axle_height
        values["gas_pressure_pa"] = self.pressure_pa
        area, force = _EVALUATE(*(values[name] for name in _ORDER))
        return float(area), float(force)


#: A go-kart-sized tire: 0.23 m rolling radius, 0.16 m section, 1.9 bar.
KART = TireSection().scaled_to(radius_m=0.23, width_m=0.16, pressure_pa=1.9e5)
#: A light aircraft tailwheel-class tire: taller, narrower, harder.
AIRCRAFT = TireSection().scaled_to(radius_m=0.26, width_m=0.12, pressure_pa=3.1e5)


if __name__ == "__main__":
    print(f"argument order: {', '.join(_ORDER)}")
    for name, section in (("kart", KART), ("aircraft", AIRCRAFT)):
        print(f"\n{name}: shoulder r {section.shoulder_r:.3f} m, "
              f"half-width {section.bead_z:.3f} m, {section.pressure_pa/1e5:.2f} bar, "
              f"travel {section.max_compression_m*1000:.0f} mm")
        for depth_mm in (0, 2, 5, 10, 20, 40, 60):
            area, force = section.contact(depth_mm / 1000.0)
            print(f"   {depth_mm:3d} mm -> patch {area*1e4:8.1f} cm2   load {force:9.1f} N")
