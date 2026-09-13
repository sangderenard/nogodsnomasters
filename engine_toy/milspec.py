"""Structural materials and tube schedules, to real specification.

WHAT THIS REPLACES. Every member in a production graph was sized as
`area = pi * radius**2` -- a SOLID BAR -- against one hardcoded steel
yield, and its damage record declared no modulus, no yield and no
section at all. So the game's solver fell back to its own car defaults,
350 MPa and 205 GPa, and a gun mount was being analysed as solid mild
steel rod. Two errors pulling opposite ways: a solid section is far
stronger than the tube actually drawn, and mild steel is far weaker
than anything a mount is built from.

A TUBE IS NOT A ROD. Area goes with the difference of the squares, so a
60 mm tube with a 4 mm wall has a quarter of the area of a 60 mm bar --
but it keeps almost all of the bending stiffness, because the second
moment goes with the FOURTH power of radius and the material that
matters is the material furthest from the axis. That is the whole
reason structures are tubes, and modelling them as bars gets the axial
answer wrong by a factor of four while getting the bending answer
roughly right by accident.

THE NUMBERS ARE SPECIFICATION VALUES, not invented ones: ASME B36.10
pipe schedules, MIL-T-6736 for 4130 tube, MIL-S-16216 for HY-80,
MIL-DTL-12560 for rolled homogeneous armour, and the standard aerospace
and gun-steel alloys. Elongation is carried as the fracture strain the
member law asks for, which is what elongation-at-break physically is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralMaterial:
    """What a member is made of, to spec."""
    key: str
    label: str
    youngs_pa: float
    shear_pa: float
    yield_pa: float
    ultimate_pa: float
    elongation: float          # at break: the member law's fracture strain
    density_kg_m3: float
    spec: str = ""
    note: str = ""


MATERIALS: tuple[StructuralMaterial, ...] = (
    StructuralMaterial("a36", "A36 structural steel", 200e9, 79.3e9, 250e6, 400e6,
                       0.23, 7850.0, "ASTM A36",
                       note="what a building is made of, and the weakest thing "
                            "here; it is the game's own default, which is why a "
                            "turret analysed without declaring anything came out "
                            "looking like scaffolding"),
    StructuralMaterial("4130n", "4130 chromoly, normalised", 205e9, 80e9, 460e6, 670e6,
                       0.255, 7850.0, "MIL-T-6736",
                       note="the standard tube for space frames, roll cages and "
                            "light mounts: nearly twice A36's yield, weldable, and "
                            "it keeps its ductility"),
    StructuralMaterial("4340qt", "4340 alloy steel, quenched and tempered",
                       205e9, 80e9, 1080e6, 1280e6, 0.12, 7850.0, "AMS 6415",
                       note="where the load is high and the section cannot grow: "
                            "trunnion pins, ram rods, highly loaded links"),
    StructuralMaterial("hy80", "HY-80 steel", 207e9, 79e9, 550e6, 690e6,
                       0.20, 7860.0, "MIL-S-16216",
                       note="submarine hull plate: high yield that stays tough and "
                            "weldable, which is the combination that makes it "
                            "worth its cost"),
    StructuralMaterial("hy100", "HY-100 steel", 207e9, 79e9, 690e6, 790e6,
                       0.18, 7860.0, "MIL-S-16216",
                       note="HY-80's harder sibling; more yield, less forgiveness"),
    StructuralMaterial("rha", "rolled homogeneous armour", 207e9, 79e9, 1000e6, 1200e6,
                       0.12, 7850.0, "MIL-DTL-12560",
                       note="the armour standard everything else is quoted against"),
    StructuralMaterial("300m", "300M gun and landing-gear steel", 205e9, 80e9,
                       1520e6, 1930e6, 0.10, 7870.0, "AMS 6417",
                       note="the strongest steel here and the least ductile: gun "
                            "tubes, landing gear, recoil rods"),
    StructuralMaterial("6061t6", "6061-T6 aluminium", 68.9e9, 26e9, 276e6, 310e6,
                       0.12, 2700.0, "ASTM B221",
                       note="a third the density and a third the stiffness; good "
                            "where mass matters more than deflection"),
    StructuralMaterial("7075t6", "7075-T6 aluminium", 71.7e9, 26.9e9, 503e6, 572e6,
                       0.11, 2810.0, "AMS 4045",
                       note="aircraft aluminium: steel-grade yield at a third the "
                            "weight, but it does not weld and it does not forgive"),
    StructuralMaterial("ti64", "Ti-6Al-4V", 113.8e9, 44e9, 880e6, 950e6,
                       0.14, 4430.0, "MIL-T-9046",
                       note="half the density of steel at most of the strength, "
                            "for when the answer to weight is money"),
)
MATERIAL_BY_KEY = {m.key: m for m in MATERIALS}


# ---------------------------------------------------------------------
#  TUBE SCHEDULES
# ---------------------------------------------------------------------
#: ASME B36.10 welded and seamless wrought steel pipe: nominal size ->
#: (outside diameter m, {schedule: wall thickness m}). Real table.
PIPE_B36_10 = {
    "1/2":   (0.0213, {"40": 0.00277, "80": 0.00373, "160": 0.00478}),
    "3/4":   (0.0267, {"40": 0.00287, "80": 0.00391, "160": 0.00556}),
    "1":     (0.0334, {"40": 0.00338, "80": 0.00455, "160": 0.00635}),
    "1-1/4": (0.0422, {"40": 0.00356, "80": 0.00485, "160": 0.00635}),
    "1-1/2": (0.0483, {"40": 0.00368, "80": 0.00508, "160": 0.00714}),
    "2":     (0.0603, {"40": 0.00391, "80": 0.00554, "160": 0.00874}),
    "2-1/2": (0.0730, {"40": 0.00516, "80": 0.00701, "160": 0.00953}),
    "3":     (0.0889, {"40": 0.00549, "80": 0.00762, "160": 0.01113}),
    "4":     (0.1143, {"40": 0.00602, "80": 0.00856, "160": 0.01349}),
    "6":     (0.1683, {"40": 0.00711, "80": 0.01097, "160": 0.01826}),
    "8":     (0.2191, {"40": 0.00818, "80": 0.01270, "160": 0.02301}),
}


@dataclass(frozen=True)
class TubeSection:
    """A real section: an outside diameter and a wall, not a radius."""
    outer_diameter_m: float
    wall_m: float
    material: str = "4130n"
    designation: str = ""

    @property
    def inner_diameter_m(self) -> float:
        return max(0.0, self.outer_diameter_m - 2.0 * self.wall_m)

    @property
    def area_m2(self) -> float:
        """THE DIFFERENCE OF THE SQUARES. This is the number that was
        wrong: a solid bar of the same diameter has several times this."""
        ro, ri = self.outer_diameter_m / 2.0, self.inner_diameter_m / 2.0
        return math.pi * (ro * ro - ri * ri)

    @property
    def second_moment_m4(self) -> float:
        """Bending stiffness, which goes with the FOURTH power -- and is
        why removing the middle of a bar costs so little of it."""
        ro, ri = self.outer_diameter_m / 2.0, self.inner_diameter_m / 2.0
        return math.pi * (ro ** 4 - ri ** 4) / 4.0

    @property
    def radius_of_gyration_m(self) -> float:
        return math.sqrt(self.second_moment_m4 / max(self.area_m2, 1e-12))

    @property
    def mat(self) -> StructuralMaterial:
        return MATERIAL_BY_KEY[self.material]

    @property
    def mass_per_m_kg(self) -> float:
        return self.area_m2 * self.mat.density_kg_m3

    def axial_yield_n(self) -> float:
        return self.area_m2 * self.mat.yield_pa

    def buckling_load_n(self, length_m: float, end_fixity: float = 1.0) -> float:
        """Euler buckling. A COLUMN DOES NOT FAIL BY YIELDING, it fails by
        going sideways, and a long thin tube goes sideways at a small
        fraction of its yield load. Ignoring this is how a structure of
        slender members passes a strength check and folds anyway."""
        if length_m <= 0.0:
            return float("inf")
        effective = length_m / max(end_fixity, 1e-6)
        return (math.pi ** 2 * self.mat.youngs_pa * self.second_moment_m4
                / (effective * effective))

    def capacity_n(self, length_m: float, end_fixity: float = 1.0) -> float:
        """What it will actually take in compression: the lesser of
        squashing and buckling."""
        return min(self.axial_yield_n(), self.buckling_load_n(length_m, end_fixity))

    def describe(self) -> list[str]:
        m = self.mat
        return [f"  {self.designation or 'tube'}: {self.outer_diameter_m * 1000:.1f} mm OD "
                f"x {self.wall_m * 1000:.2f} mm wall, {m.label} ({m.spec})",
                f"    area {self.area_m2 * 1e6:7.1f} mm2  I {self.second_moment_m4 * 1e12:8.1f} mm4  "
                f"{self.mass_per_m_kg:5.2f} kg/m  yield {self.axial_yield_n() / 1000:7.1f} kN"]


def pipe(nominal: str, schedule: str = "40", material: str = "4130n") -> TubeSection:
    od, walls = PIPE_B36_10[nominal]
    return TubeSection(od, walls[schedule], material,
                       designation=f"NPS {nominal} Sch {schedule}")


def select_tube(required_n: float, length_m: float, *, material: str = "4130n",
                factor: float = 1.5, end_fixity: float = 1.0) -> TubeSection:
    """The lightest standard pipe that carries this load with margin.

    Checked against BOTH yielding and buckling, because a member chosen
    on yield alone is a member chosen on the wrong failure mode whenever
    it is slender -- which, in a space frame, is most of them."""
    need = abs(required_n) * factor
    best = None
    for nominal in PIPE_B36_10:
        for schedule in ("40", "80", "160"):
            t = pipe(nominal, schedule, material)
            if t.capacity_n(length_m, end_fixity) >= need:
                if best is None or t.mass_per_m_kg < best.mass_per_m_kg:
                    best = t
    return best or pipe("8", "160", material)


def damage_record(section: TubeSection, rest_length_m: float, *,
                  spring_like: bool = False, travels: bool = False) -> dict:
    """The member's damage record, to spec, for the game's solver.

    Every field the game reads is declared here rather than left to its
    defaults: section, modulus, yield, ultimate and fracture strain. A
    member that declares none of these is solved as the game's default
    car steel, which is what was happening to every member this project
    has ever authored.
    """
    m = section.mat
    record = {
        "model": "elastic-plastic-member-with-shear-fracture",
        "natural_rest_length": rest_length_m,
        "section_area_m2": section.area_m2,
        "second_moment_m4": section.second_moment_m4,
        "youngs_modulus_pa": m.youngs_pa,
        "shear_modulus_pa": m.shear_pa,
        "yield_strength_pa": m.yield_pa,
        "ultimate_strength_pa": m.ultimate_pa,
        "fracture_strain": m.elongation,
        "plastic_strain_limit": 0.035 if spring_like else 0.0025,
        "axial_yield_force_n": section.axial_yield_n(),
        "buckling_load_n": section.buckling_load_n(rest_length_m),
        "shear_force_limit_n": section.area_m2 * m.yield_pa * 0.577,  # von Mises
        "material": m.key,
        "material_spec": m.spec,
        "section_designation": section.designation,
        "failure_response": "constraint-opens-and-load-path-is-removed",
        "respawn_response": "restore-authored-natural-length-and-health",
    }
    if travels:
        # A MEMBER WHOSE LENGTH IS COMMANDED IS NOT BEING STRAINED. The
        # game already carries this idea -- `axial_kinematic` in
        # `vehicle_native_graph_program`, which it uses so a suspension
        # member's travel is not read as stretch. A recoil slide, a
        # prismatic guide and a hydraulic ram are the same case: without
        # it, the very motion the part exists to permit is reported as
        # fracture, and 320 mm of recoil came back as 178% strain.
        record["axial_travel_is_commanded"] = True
    return record
