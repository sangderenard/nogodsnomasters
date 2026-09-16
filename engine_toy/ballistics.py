"""Fast deterministic terminal-ballistics rules for engine impacts.

The model is intentionally compact rather than empirical ammunition-table
fitting. It preserves the useful physical relationships needed by the engine
simulation: kinetic energy and momentum, impact obliquity, yaw/tumble,
projected area, target strength/ductility, projectile deformation, and drag
through a contained fluid.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Sequence


ATMOSPHERIC_PRESSURE_PA = 101_325.0


class DamageMode(str, Enum):
    DENT = "dent"
    CRATER = "crater"
    PUNCTURE = "puncture"
    SPALLING_PUNCTURE = "spalling-puncture"
    RICOCHET = "ricochet"


@dataclass(frozen=True)
class MaterialProfile:
    toughness_j_m3: float
    yield_strength_pa: float
    hardness_pa: float
    density_kg_m3: float
    ductility: float
    ricochet_angle_deg: float
    shell_wall_m: float | None = None


@dataclass(frozen=True)
class FluidLayer:
    name: str
    density_kg_m3: float
    pressure_pa: float = ATMOSPHERIC_PRESSURE_PA
    drag_coefficient: float = 0.8


@dataclass(frozen=True)
class ProjectileState:
    mass_kg: float
    diameter_m: float
    speed_m_s: float
    direction: tuple[float, float, float]
    length_m: float
    yaw_rad: float = 0.0
    tumble_rad_s: float = 0.0
    hardness_pa: float = 1.2e9
    integrity: float = 1.0

    @classmethod
    def from_energy(
        cls,
        energy_j: float,
        diameter_m: float,
        *,
        mass_kg: float = 0.0095,
        direction: Sequence[float] = (1.0, 0.0, 0.0),
        yaw_rad: float = 0.0,
        tumble_rad_s: float = 0.0,
    ) -> "ProjectileState":
        mass = max(float(mass_kg), 1e-9)
        speed = math.sqrt(max(0.0, 2.0 * float(energy_j) / mass))
        return cls(
            mass_kg=mass,
            diameter_m=max(float(diameter_m), 1e-6),
            speed_m_s=speed,
            direction=_unit(direction),
            length_m=max(float(diameter_m) * 3.5, 1e-6),
            yaw_rad=float(yaw_rad),
            tumble_rad_s=float(tumble_rad_s),
        )

    @property
    def energy_j(self) -> float:
        return 0.5 * self.mass_kg * self.speed_m_s * self.speed_m_s

    @property
    def momentum_kg_m_s(self) -> float:
        return self.mass_kg * self.speed_m_s

    @property
    def presented_area_m2(self) -> float:
        base = math.pi * (self.diameter_m * 0.5) ** 2
        slenderness = max(1.0, self.length_m / self.diameter_m)
        yaw_factor = abs(math.cos(self.yaw_rad)) + slenderness * abs(math.sin(self.yaw_rad))
        return base * min(4.0, max(1.0, yaw_factor))


@dataclass(frozen=True)
class ImpactResult:
    mode: DamageMode
    perforated: bool
    ricocheted: bool
    incidence_angle_deg: float
    effective_thickness_m: float
    penetration_depth_m: float
    energy_before_j: float
    energy_spent_j: float
    fluid_energy_spent_j: float
    speed_before_m_s: float
    speed_after_m_s: float
    diameter_after_m: float
    yaw_after_rad: float
    integrity_after: float
    projectile_disrupted: bool
    projectile_after: ProjectileState


MATERIAL_PROFILES: dict[str, MaterialProfile] = {
    # CANVAS IS NOT A WALL. A proofed cotton or polyester shelter keeps
    # weather off equipment and stops nothing else: a round crosses it
    # having lost effectively none of its energy, and what it leaves is
    # a TEAR that runs, not a neat hole that stays the size of the
    # projectile. Toughness three orders below sheet steel is the point
    # of the material, not an approximation of one -- an enclosure like
    # this is cover from rain and from being seen, and from nothing at
    # all that is being shot at it.
    "canvas": MaterialProfile(2.0e6, 30e6, 4.0e6, 380.0, 0.95, 88.0, 0.0006),
    "flame": MaterialProfile(1.0e3, 1.0e3, 1.0e3, 0.8, 1.0, 90.0),
    "smoke": MaterialProfile(1.0e3, 1.0e3, 1.0e3, 1.2, 1.0, 90.0),
    "cover": MaterialProfile(0.4e9, 180e6, 0.7e9, 2700.0, 0.75, 68.0, 0.0015),
    "intake": MaterialProfile(0.3e9, 160e6, 0.6e9, 2700.0, 0.70, 67.0, 0.0030),
    "fuel": MaterialProfile(0.8e9, 220e6, 0.9e9, 7800.0, 0.70, 66.0, 0.0015),
    "lube": MaterialProfile(0.8e9, 220e6, 0.9e9, 7800.0, 0.70, 66.0, 0.0020),
    "ignition": MaterialProfile(0.2e9, 80e6, 0.3e9, 1600.0, 0.65, 70.0, 0.0010),
    "exhaust": MaterialProfile(0.9e9, 260e6, 1.0e9, 7800.0, 0.55, 64.0, 0.0025),
    "head": MaterialProfile(2.2e9, 240e6, 0.9e9, 2700.0, 0.35, 62.0, 0.0100),
    "piston": MaterialProfile(1.8e9, 260e6, 1.0e9, 2700.0, 0.45, 62.0),
    "case": MaterialProfile(2.5e9, 300e6, 1.4e9, 7100.0, 0.25, 58.0, 0.0060),
    "cylinder": MaterialProfile(2.5e9, 300e6, 1.4e9, 7100.0, 0.25, 58.0, 0.0080),
    "casting_port": MaterialProfile(2.0e9, 280e6, 1.2e9, 6500.0, 0.30, 60.0),
    "intake_port": MaterialProfile(2.0e9, 240e6, 0.9e9, 2700.0, 0.35, 62.0),
    "exhaust_port": MaterialProfile(2.0e9, 280e6, 1.2e9, 6500.0, 0.30, 60.0),
    "rod": MaterialProfile(5.0e9, 700e6, 2.2e9, 7850.0, 0.45, 55.0),
    "crank": MaterialProfile(6.0e9, 850e6, 2.6e9, 7850.0, 0.38, 53.0),
    "valve": MaterialProfile(6.0e9, 900e6, 2.8e9, 7900.0, 0.30, 52.0),
    "cam": MaterialProfile(5.0e9, 750e6, 2.5e9, 7800.0, 0.30, 54.0),
    "spring": MaterialProfile(3.0e9, 1000e6, 3.0e9, 7800.0, 0.55, 50.0),
    "follower": MaterialProfile(4.0e9, 650e6, 2.2e9, 7800.0, 0.40, 55.0),
    "rotor": MaterialProfile(4.0e9, 550e6, 1.8e9, 7600.0, 0.35, 56.0),
    "other": MaterialProfile(1.0e9, 250e6, 1.0e9, 5000.0, 0.40, 62.0, 0.0030),
}

_PROFILE_ALIASES = {
    "igniter": ("other", 1.0e9),
    "injector": ("case", 2.0e9),
    "expander_port": ("case", 2.0e9),
    "air_filter": ("cover", 0.08e9),
    "throttle_plate": ("follower", 1.5e9),
    "throttle_body": ("intake", 0.7e9),
    "magneto": ("cover", 0.5e9),
    "compression_brake": ("case", 2.5e9),
    "pneumatic": ("fuel", 1.2e9),
    "mount": ("other", 0.12e9),
    "pulley": ("follower", 3.0e9),
    "damper": ("follower", 2.0e9),
    "starter": ("case", 1.8e9),
    "coolant_gear": ("cover", 0.35e9),
    "egr": ("exhaust", 1.1e9),
    "pcv": ("cover", 0.15e9),
    "housing": ("case", 1.8e9),
    "blower": ("head", 1.4e9),
    "burst_panel": ("cover", 0.25e9),
    "blower_pulley": ("follower", 3.0e9),
    "charge_side": ("head", 1.2e9),
    "hot_side": ("case", 2.0e9),
    "barrel_valve": ("follower", 2.0e9),
    "prop_drive": ("crank", 5.0e9),
    "governor": ("follower", 2.0e9),
    "injection_pump": ("case", 2.5e9),
    "oil_cooler": ("cover", 0.5e9),
    "cooling_tin": ("cover", 0.25e9),
    "gas_path": ("head", 2.0e9),
    "turbine_accessory": ("case", 1.8e9),
    "drive_unit": ("case", 2.5e9),
    "power_electronics": ("cover", 0.45e9),
    "boiler": ("case", 3.0e9),
    "plant_fittings": ("follower", 2.0e9),
    "otto_mechanism": ("follower", 2.0e9),
    "small_engine_kit": ("case", 1.2e9),
    "transaxle": ("case", 2.5e9),
    "halfshaft": ("crank", 5.0e9),
}
for _name, (_base, _toughness) in _PROFILE_ALIASES.items():
    MATERIAL_PROFILES[_name] = replace(MATERIAL_PROFILES[_base], toughness_j_m3=_toughness)


# ---------------------------------------------------------------------
# MACHINE MATERIALS. Everything above is keyed by the render row an
# ENGINE part draws with, which is why a turret's armour reported as
# "other" and was stopped by a default: the profiles simply had no row
# for it. These are the same kind of entry for the materials machines
# (machines.py, turret_production.py) actually declare.
#
# The numbers are ordinary published ones for the alloys concerned, and
# the relationships between them are the point:
#
#   RHA (rolled homogeneous armour) is not remarkable steel. Its yield
#   is a little over twice mild plate and its toughness about three
#   times -- armour works by being THICK and SLOPED far more than by
#   being exotic, which is why the ricochet angle matters as much as
#   the toughness.
#
#   Gun steel is harder and stronger than armour but LESS tough: it is
#   built to contain pressure and resist erosion, not to absorb a hit.
#   A barrel is a bad place to be shot.
#
#   Cast iron is stiff, cheap and brittle -- high hardness, low
#   ductility -- so it cracks rather than dents, which is exactly how a
#   holed pump housing behaves.
#
#   Aluminium tanks and covers are a third the density and a fraction
#   of the toughness: they stop very little, and their value is that
#   they are light.
MATERIAL_PROFILES.update({
    "armour": MaterialProfile(
        toughness_j_m3=3_400e6, yield_strength_pa=1_100e6, hardness_pa=3.6e9,
        density_kg_m3=7850.0, ductility=0.12, ricochet_angle_deg=42.0,
        shell_wall_m=0.020),
    "gun_steel": MaterialProfile(
        toughness_j_m3=2_600e6, yield_strength_pa=1_250e6, hardness_pa=4.1e9,
        density_kg_m3=7850.0, ductility=0.08, ricochet_angle_deg=50.0),
    "hardened": MaterialProfile(
        toughness_j_m3=3_000e6, yield_strength_pa=950e6, hardness_pa=3.2e9,
        density_kg_m3=7850.0, ductility=0.10, ricochet_angle_deg=52.0),
    "plate": MaterialProfile(
        toughness_j_m3=1_600e6, yield_strength_pa=350e6, hardness_pa=1.4e9,
        density_kg_m3=7850.0, ductility=0.22, ricochet_angle_deg=58.0,
        shell_wall_m=0.008),
    "pressed": MaterialProfile(
        toughness_j_m3=1_200e6, yield_strength_pa=300e6, hardness_pa=1.2e9,
        density_kg_m3=7850.0, ductility=0.26, ricochet_angle_deg=62.0,
        shell_wall_m=0.004),
    "cast_iron": MaterialProfile(
        toughness_j_m3=700e6, yield_strength_pa=250e6, hardness_pa=2.0e9,
        density_kg_m3=7100.0, ductility=0.02, ricochet_angle_deg=58.0),
    "pipe": MaterialProfile(
        toughness_j_m3=1_400e6, yield_strength_pa=310e6, hardness_pa=1.3e9,
        density_kg_m3=7850.0, ductility=0.24, ricochet_angle_deg=60.0,
        shell_wall_m=0.003),
    "shaft": MaterialProfile(
        toughness_j_m3=4_200e6, yield_strength_pa=850e6, hardness_pa=2.8e9,
        density_kg_m3=7850.0, ductility=0.16, ricochet_angle_deg=53.0),
    "airline": MaterialProfile(
        toughness_j_m3=160e6, yield_strength_pa=60e6, hardness_pa=0.12e9,
        density_kg_m3=1150.0, ductility=0.60, ricochet_angle_deg=74.0,
        shell_wall_m=0.0015),
    "leak_gear_oil": MaterialProfile(
        toughness_j_m3=1e6, yield_strength_pa=1e5, hardness_pa=1e6,
        density_kg_m3=900.0, ductility=1.0, ricochet_angle_deg=88.0),
})

# What a DECLARED material is called, versus the render row it draws
# with. A part says "armour-plate"; the mesh draws it as "armour"; the
# ballistics looks up "armour". One mapping, so the three can never
# drift apart.
DECLARED_MATERIAL_ROWS = {
    "armour-plate": "armour",
    "gun-steel": "gun_steel",
    "hardened-steel": "hardened",
    "steel-plate": "plate",
    "pressed-steel": "pressed",
    "cast-iron": "cast_iron",
    "steel-pipe": "pipe",
    "steel-shaft": "shaft",
    "nylon-airline": "airline",
}


def profile_for_declared(declared: str) -> MaterialProfile:
    """The physical profile behind a declared material name."""
    return material_profile(DECLARED_MATERIAL_ROWS.get(str(declared), str(declared)))


def mass_of(declared: str, volume_m3: float, *, hollow_fraction: float = 0.0) -> float:
    """Mass from geometry and material, rather than a number somebody
    typed.

    `hollow_fraction` is how much of the envelope is air: a tank, a
    tube or a shell is mostly empty, and treating it as solid steel
    gives a mount that weighs more than a car. Declaring the fraction
    keeps the estimate honest without needing a wall-by-wall model."""
    profile = profile_for_declared(declared)
    solid = max(0.0, float(volume_m3)) * max(0.0, 1.0 - float(hollow_fraction))
    return solid * profile.density_kg_m3


def material_profile(name: str, toughness_override_j_m3: float | None = None) -> MaterialProfile:
    profile = MATERIAL_PROFILES.get(name)
    if profile is None:
        profile = replace(MATERIAL_PROFILES["other"], shell_wall_m=None)
    if toughness_override_j_m3 is None:
        return profile
    return replace(profile, toughness_j_m3=max(float(toughness_override_j_m3), 1e-9))


def resisting_thickness(profile: MaterialProfile, crossed_m: float) -> float:
    crossed = max(0.0, float(crossed_m))
    if profile.shell_wall_m is None:
        return crossed
    return min(crossed, 2.0 * profile.shell_wall_m)


def resolve_impact(
    projectile: ProjectileState,
    target: MaterialProfile,
    crossed_thickness_m: float,
    surface_normal: Sequence[float],
    *,
    fluid: FluidLayer | None = None,
    fluid_path_m: float = 0.0,
) -> ImpactResult:
    """Resolve one wall traversal and return the next projectile state."""
    direction = _unit(projectile.direction)
    normal = _unit(surface_normal)
    incidence_cos = min(1.0, max(0.0, abs(_dot(direction, normal))))
    incidence_angle = math.degrees(math.acos(incidence_cos))
    effective_thickness = max(0.0, crossed_thickness_m) / max(incidence_cos, 0.15)
    energy_before = projectile.energy_j
    speed_before = projectile.speed_m_s
    area = projectile.presented_area_m2

    normal_speed = speed_before * incidence_cos
    hardness_ratio = target.hardness_pa / max(projectile.hardness_pa, 1e-9)
    deformation_drive = hardness_ratio * (normal_speed / 800.0) ** 2
    diameter_factor = 1.0 + min(0.8, max(0.0, deformation_drive - 0.15) * 0.28)
    diameter_after = projectile.diameter_m * diameter_factor
    deformation_factor = 1.0 + (diameter_factor - 1.0) * 1.5

    work_required = target.toughness_j_m3 * area * effective_thickness * deformation_factor
    critical_angle = (
        target.ricochet_angle_deg
        + 8.0 * math.log2(max(projectile.hardness_pa / target.hardness_pa, 0.25))
        - min(10.0, abs(projectile.yaw_rad) * 20.0)
    )
    normal_energy = energy_before * incidence_cos * incidence_cos
    ricocheted = incidence_angle >= critical_angle and normal_energy < work_required * 1.5

    if ricocheted:
        energy_spent = min(energy_before, max(work_required * 0.08, energy_before * 0.18))
        remaining = max(0.0, energy_before - energy_spent)
        reflected = _reflect(direction, normal)
        yaw_after = min(math.pi / 2.0, abs(projectile.yaw_rad) + math.radians(20.0))
        integrity = max(0.0, projectile.integrity - 0.08 - 0.12 * min(1.0, deformation_drive))
        after = replace(
            projectile,
            speed_m_s=_speed_from_energy(remaining, projectile.mass_kg),
            direction=reflected,
            diameter_m=diameter_after,
            yaw_rad=yaw_after,
            integrity=integrity,
        )
        return ImpactResult(
            mode=DamageMode.RICOCHET,
            perforated=False,
            ricocheted=True,
            incidence_angle_deg=incidence_angle,
            effective_thickness_m=effective_thickness,
            penetration_depth_m=0.0,
            energy_before_j=energy_before,
            energy_spent_j=energy_spent,
            fluid_energy_spent_j=0.0,
            speed_before_m_s=speed_before,
            speed_after_m_s=after.speed_m_s,
            diameter_after_m=diameter_after,
            yaw_after_rad=yaw_after,
            integrity_after=integrity,
            projectile_disrupted=integrity <= 0.1,
            projectile_after=after,
        )

    perforated = energy_before >= work_required
    solid_energy_spent = min(energy_before, work_required)
    penetration_depth = (
        effective_thickness
        if perforated
        else energy_before / max(target.toughness_j_m3 * area * deformation_factor, 1e-12)
    )
    remaining = max(0.0, energy_before - solid_energy_spent)

    fluid_energy = 0.0
    if perforated and fluid is not None and fluid_path_m > 0.0 and remaining > 0.0:
        speed_after_solid = _speed_from_energy(remaining, projectile.mass_kg)
        drag_force = 0.5 * fluid.density_kg_m3 * fluid.drag_coefficient * area * speed_after_solid * speed_after_solid
        pressure_work = max(0.0, fluid.pressure_pa - ATMOSPHERIC_PRESSURE_PA) * area * fluid_path_m
        fluid_energy = min(remaining, drag_force * fluid_path_m + pressure_work)
        remaining -= fluid_energy

    transit_s = max(effective_thickness, 0.0) / max(speed_before, 1e-9)
    integrity_loss = min(
        0.85,
        (solid_energy_spent / max(energy_before, 1e-9))
        * (0.35 + 0.45 * (1.0 - target.ductility))
        * max(0.5, hardness_ratio),
    )
    integrity = max(0.0, projectile.integrity - integrity_loss)
    yaw_after = min(
        math.pi / 2.0,
        abs(projectile.yaw_rad + projectile.tumble_rad_s * transit_s)
        + math.radians(incidence_angle) * integrity_loss * 0.15,
    )
    tumble_after = projectile.tumble_rad_s + math.radians(incidence_angle) * integrity_loss * 50.0
    after = replace(
        projectile,
        speed_m_s=_speed_from_energy(remaining, projectile.mass_kg),
        diameter_m=diameter_after,
        yaw_rad=yaw_after,
        tumble_rad_s=tumble_after,
        integrity=integrity,
    )

    if perforated:
        mode = DamageMode.SPALLING_PUNCTURE if target.ductility < 0.35 and energy_before > work_required * 1.4 else DamageMode.PUNCTURE
    else:
        mode = DamageMode.CRATER if penetration_depth >= effective_thickness * 0.2 else DamageMode.DENT
    return ImpactResult(
        mode=mode,
        perforated=perforated,
        ricocheted=False,
        incidence_angle_deg=incidence_angle,
        effective_thickness_m=effective_thickness,
        penetration_depth_m=penetration_depth,
        energy_before_j=energy_before,
        energy_spent_j=solid_energy_spent + fluid_energy,
        fluid_energy_spent_j=fluid_energy,
        speed_before_m_s=speed_before,
        speed_after_m_s=after.speed_m_s,
        diameter_after_m=diameter_after,
        yaw_after_rad=yaw_after,
        integrity_after=integrity,
        projectile_disrupted=integrity <= 0.1,
        projectile_after=after,
    )


def inferred_fluid(part: str, material: str, pressure_pa: float = ATMOSPHERIC_PRESSURE_PA) -> FluidLayer | None:
    lowered = part.lower()
    if "water_jacket" in lowered or "coolant" in lowered:
        return FluidLayer("coolant", 1050.0, pressure_pa, 0.85)
    if material == "lube" or "oil" in lowered:
        return FluidLayer("engine-oil", 850.0, pressure_pa, 1.2)
    if material == "fuel" or "fuel" in lowered:
        return FluidLayer("fuel", 760.0, pressure_pa, 0.9)
    if material == "pneumatic" or "air_reservoir" in lowered:
        return FluidLayer("compressed-air", 1.2 * pressure_pa / ATMOSPHERIC_PRESSURE_PA, pressure_pa, 0.6)
    if material in ("intake", "intake_port"):
        return FluidLayer("intake-gas", 1.2 * pressure_pa / ATMOSPHERIC_PRESSURE_PA, pressure_pa, 0.5)
    if material in ("exhaust", "exhaust_port"):
        return FluidLayer("exhaust-gas", 0.7 * pressure_pa / ATMOSPHERIC_PRESSURE_PA, pressure_pa, 0.6)
    return None


def advance_projectile(projectile: ProjectileState, dt: float, medium: FluidLayer | None = None) -> ProjectileState:
    """Advance yaw and quadratic fluid drag for one inexpensive time step."""
    elapsed = max(0.0, float(dt))
    yaw = (projectile.yaw_rad + projectile.tumble_rad_s * elapsed) % math.pi
    speed = projectile.speed_m_s
    if medium is not None and speed > 0.0 and elapsed > 0.0:
        drag_k = (
            0.5
            * medium.density_kg_m3
            * medium.drag_coefficient
            * projectile.presented_area_m2
            / max(projectile.mass_kg, 1e-12)
        )
        speed = speed / (1.0 + drag_k * speed * elapsed)
    return replace(projectile, speed_m_s=speed, yaw_rad=yaw)


def _speed_from_energy(energy_j: float, mass_kg: float) -> float:
    return math.sqrt(max(0.0, 2.0 * energy_j / max(mass_kg, 1e-12)))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _unit(value: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(component) for component in value)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0)
    return (x / norm, y / norm, z / norm)


def _reflect(direction: Sequence[float], normal: Sequence[float]) -> tuple[float, float, float]:
    scale = 2.0 * _dot(direction, normal)
    return _unit((
        float(direction[0]) - scale * float(normal[0]),
        float(direction[1]) - scale * float(normal[1]),
        float(direction[2]) - scale * float(normal[2]),
    ))
