"""Rough 3D cylinder + mount-point geometry derived from an engine's architecture.

Not a CAD model — a plausible placement good enough to give mechanical
vibration sources distinct positions relative to a handful of named mount
points, so different mounts pick up different sources with different
weight (valve cover hears valvetrain loudest, front mount hears the
accessory drive and primary imbalance loudest, etc).

Cylinder numbering convention: physical cylinder k (1-indexed) is placed
at bank = (k-1) % banks, position-along-crank = (k-1) // banks. Real
manufacturers vary this; it's fixed here just to be internally consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from engines import Engine

Vec3 = tuple[float, float, float]

NOMINAL_BORE_SPACING_M = 0.105
BANK_RADIUS_M = 0.16
# NOMINAL_BORE_SPACING_M is calibrated AT this real per-cylinder
# displacement (a real 2.0L I4's own 0.5L/cylinder, real bore spacing
# ~0.094m -- 0.105m is a disclosed, reasonable approximation of that).
# Every size formula below scales off this reference via a cube root
# (bore scales with volume^(1/3)) -- a previous version divided by 0.4
# instead (implying a ~0.064L/cylinder reference), which inflated
# spacing ~2.3x for any ordinary engine: a real AMC 258 I6 (0.7L/
# cylinder) came out 0.234m instead of the real ~0.111m, stretching the
# whole crank to 1.29m instead of ~0.68m and putting a full 0.69m of
# empty space between the crank center and the accessory-drive face.
NOMINAL_DISPLACEMENT_PER_CYLINDER_L = 0.5


def _engine_bank_radius_m(engine: Engine) -> float:
    """BANK_RADIUS_M, scaled the SAME real cube-root way spacing already
    is -- this was still a flat constant everywhere it was used outside
    _radial_cylinder_sites (which already had its own local per-engine
    scale), so a 25cc trimmer and a 21,000L 14-cylinder marine diesel
    got the exact same cylinder-bore offset, valve-cover height, and
    block cross-section: real crank LENGTH scaled correctly while real
    crank GIRTH stayed fixed, an increasingly "exploded" mismatch the
    bigger or smaller an engine got from the 0.5L/cylinder reference."""
    n = max(engine.architecture.cylinders, 1)
    displacement_per_cylinder = max(engine.displacement_l / n, 1e-4)
    return BANK_RADIUS_M * (displacement_per_cylinder / NOMINAL_DISPLACEMENT_PER_CYLINDER_L) ** (1 / 3)


def block_half_yz_m(engine: Engine) -> float:
    """The crankcase's own real half-height/half-width (kept square) --
    the ONE place this is computed. Everything that needs to clear the
    block's own surface (the accessory-drive arc, port standoff, ...)
    must derive its own clearance from THIS value, not a separate
    bank_radius-scaled constant of its own that happens to agree or not
    -- that per-symptom patchwork (each individually reasonable-looking)
    is exactly what put the accessory arc's own radius at the same
    value as the block's half-extent, landing accessories inside the
    block's own volume instead of outside it."""
    return _engine_bank_radius_m(engine) * 0.5


def block_clearance_radius_m(engine: Engine) -> float:
    """The minimum real radius from the crank centerline that clears the
    block's own CORNER in every direction, not just axis-aligned ones --
    the block is a square box (block_half_yz_m on each side), so its own
    corner sits at hypot(half, half) from center, plus a small real
    standoff so accessories clear the surface rather than just touching
    it."""
    half = block_half_yz_m(engine)
    return math.hypot(half, half) + 0.02


@dataclass
class CylinderSite:
    number: int
    bank: int
    position: Vec3


RADIAL_ROW_SPACING_M = 0.30   # axial gap between rows on a multi-row radial


def _radial_cylinder_sites(engine: Engine) -> list[CylinderSite]:
    """Polar layout: every cylinder in a row shares ONE crank throw, so
    they differ in angle around the crank axis, not in position along
    it -- the opposite of an inline/V bank. A row's radius comes from
    displacement (bigger swept volume per cylinder needs a longer
    connecting rod / bigger circle); multiple rows (rare, e.g. a
    twin-row radial) stack along the crank axis, each rotated so its
    cylinders sit in the gaps of the row ahead of it.
    """
    arch = engine.architecture
    n = arch.cylinders
    rows = max(1, arch.rows)
    per_row = max(1, arch.banks)   # banks is repurposed as cylinders-per-row for radials
    radius = max(BANK_RADIUS_M * 0.5,
                 BANK_RADIUS_M * (engine.displacement_l / max(n, 1) / NOMINAL_DISPLACEMENT_PER_CYLINDER_L) ** (1 / 3))
    sites = []
    k = 1
    for row in range(rows):
        row_x = (row - (rows - 1) / 2.0) * RADIAL_ROW_SPACING_M
        row_offset_deg = (360.0 / per_row / 2.0) * row   # stagger alternate rows into the gaps
        for i in range(per_row):
            if k > n:
                break
            angle = math.radians(i * 360.0 / per_row + row_offset_deg)
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)
            sites.append(CylinderSite(number=k, bank=row, position=(row_x, y, z)))
            k += 1
    return sites


def cylinder_sites(engine: Engine) -> list[CylinderSite]:
    arch = engine.architecture
    n = arch.cylinders
    if n == 0:
        return []
    if arch.radial:
        return _radial_cylinder_sites(engine)
    banks = max(1, arch.banks)
    per_bank = math.ceil(n / banks)
    spacing = max(0.05,
                 NOMINAL_BORE_SPACING_M * (engine.displacement_l / max(n, 1) / NOMINAL_DISPLACEMENT_PER_CYLINDER_L) ** (1 / 3))
    half_angle = math.radians(arch.bank_angle_degrees / 2.0)
    bank_radius = _engine_bank_radius_m(engine)
    sites = []
    for k in range(1, n + 1):
        bank = (k - 1) % banks
        pos_index = (k - 1) // banks
        x = (pos_index - (per_bank - 1) / 2.0) * spacing
        if banks == 1:
            # +Y = away from the crank centerline (up), matching
            # mount_points()'s exhaust_manifold and _radial_cylinder_sites'
            # own y=cos/z=sin convention -- this and the V-bank case below
            # were both using +Z instead, backwards from both of those.
            y, z = bank_radius * 0.4, 0.0
        else:
            angle = half_angle if bank == 0 else -half_angle
            y = bank_radius * math.cos(angle)
            z = bank_radius * math.sin(angle)
        sites.append(CylinderSite(number=k, bank=bank, position=(x, y, z)))
    return sites


def crank_extent(sites: list[CylinderSite]) -> tuple[float, float]:
    if not sites:
        return (-0.12, 0.12)
    xs = [s.position[0] for s in sites]
    span = max(xs) - min(xs)
    # a single-row radial has every cylinder at the same x (they differ in
    # angle, not axial position) -- pad out to a plausible crankcase/
    # accessory-case depth instead of collapsing front/rear mounts together
    padding = NOMINAL_BORE_SPACING_M * 0.6 if span > 1e-6 else RADIAL_ROW_SPACING_M * 0.9
    return (min(xs) - padding, max(xs) + padding)


def accessory_front_point(engine: Engine) -> Vec3:
    """Where the front-of-block cluster (fuel delivery, throttle body)
    sits -- just ahead of the frontmost cylinder. NOT used for the
    accessory-drive arc itself any more (see accessory_mount_points):
    the belt-driven accessories surround the crank's own real axis
    (x=0), not this offset point."""
    sites = cylinder_sites(engine)
    x_min, _x_max = crank_extent(sites)
    return (x_min - 0.04, 0.0, _engine_bank_radius_m(engine) * 0.3)


def accessory_mount_points(engine: Engine) -> dict[str, Vec3]:
    """Every belt-driven accessory at the crank's own real FRONT FACE x
    (crank_extent's x_min -- the real timing-cover end a serpentine belt
    actually bolts to), not the crank's geometric center (x=0): once the
    engine block was sized to its own real length, x=0 put every
    accessory at the block's MIDPOINT, sprouting out its side instead of
    mounting at the front the way a real accessory drive does. Never
    varying per accessory: one single x, arranged around a real arc in
    that one slice's own Y-Z plane. One slot per accessory this engine's
    Accessories/pneumatics flags actually declare present, not a fixed
    set every engine is forced through regardless of its own build."""
    x_min, _x_max = crank_extent(cylinder_sites(engine))
    front = (x_min, 0.0, 0.0)
    acc = engine.accessories
    names: list[str] = []
    if acc.alternator:
        names.append("alternator")
    if acc.water_pump:
        names.append("water_pump")
    if acc.mechanical_fan:
        names.append("fan")
    # coolant_pump is deliberately excluded: it's electric/servo-only
    # (Accessories' own docstring) and drivetrain_graph.py's real subunit
    # never builds a mesh node for it at all -- an arc slot with nothing
    # real behind it.
    if acc.air_conditioning:
        names.append("ac_compressor")
    if engine.pneumatics.compressor_fitted and engine.pneumatics.compressor_drive == "crank-belt":
        # only a crank-belt compressor takes an arc slot; a motor-driven
        # one lives wherever its motor does (drivetrain_graph places it)
        names.append("pneumatic_compressor")
    radius = block_clearance_radius_m(engine)
    points: dict[str, Vec3] = {}
    for i, name in enumerate(names):
        angle = 2.0 * math.pi * i / len(names)
        points[name] = (front[0], front[1] + radius * math.cos(angle), front[2] + radius * math.sin(angle))
    return points


def acoustic_points(engine: Engine) -> dict[str, Vec3]:
    """Three listening positions: inside the header/primary-drive housing
    (the sharp exhaust/combustion note, plus turbo/backfire events), out
    in the engine bay (softer valvetrain/accessory noise), and at the
    intake plenum (induction roar, supercharger/turbo compressor whine --
    the "front" of the air-delivery contract instead of the back).
    """
    sites = cylinder_sites(engine)
    x_min, x_max = crank_extent(sites)
    if sites:
        header = (x_max + 0.22, BANK_RADIUS_M * 0.65, -0.08)
        intake = (x_min - 0.02, -BANK_RADIUS_M * 0.55, BANK_RADIUS_M * 0.9)
    else:
        # electric/servo: no exhaust, but keep the slot as the primary
        # drive-unit/inverter emission point.
        header = (x_max + 0.12, 0.0, -0.04)
        intake = (x_min - 0.05, 0.0, BANK_RADIUS_M * 0.5)
    engine_bay = ((x_min + x_max) / 2.0, 0.0, BANK_RADIUS_M * 2.2)
    return {"header": header, "engine_bay": engine_bay, "intake": intake}


def mount_points(engine: Engine) -> dict[str, Vec3]:
    sites = cylinder_sites(engine)
    x_min, x_max = crank_extent(sites)
    arch = engine.architecture
    points: dict[str, Vec3] = {
        "front_mount": (x_min, 0.0, -0.05),
        "rear_mount": (x_max, 0.0, -0.05),
        # below the crank (-Y), matching exhaust_manifold's own +Y=up
        # below -- was -Z, disagreeing with exhaust_manifold in the same
        # dict about which axis "away from the crank" even is.
        "oil_pan": ((x_min + x_max) / 2.0, -BANK_RADIUS_M * 0.9, 0.0),
        "exhaust_manifold": ((x_min + x_max) / 2.0, BANK_RADIUS_M * 0.55, -0.02),
    }
    if arch.banks >= 2 and sites and not arch.radial:
        half_angle = math.radians(arch.bank_angle_degrees / 2.0)
        top_r = BANK_RADIUS_M * 1.35
        # same sin/cos swap as cylinder_sites' V-bank case: angle 0 is
        # +Y (up), banks lean outward laterally (+-Z) as bank_angle_degrees
        # opens from vertical.
        points["valve_cover_left"] = (
            (x_min + x_max) / 2.0,
            top_r * math.cos(half_angle),
            top_r * math.sin(half_angle),
        )
        points["valve_cover_right"] = (
            (x_min + x_max) / 2.0,
            top_r * math.cos(-half_angle),
            top_r * math.sin(-half_angle),
        )
    else:
        points["valve_cover"] = ((x_min + x_max) / 2.0, BANK_RADIUS_M * 1.35, 0.0)
    return points


def distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
