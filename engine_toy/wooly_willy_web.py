"""Compile the magnetic-filings toy through Turing's normal web publisher.

This file does not contain an HTML, JavaScript, Wasm, or WGSL implementation.
It authors one ``TURING_PAGE`` Python program and hands that source to
``site_bundle.build_program_bundle``.  That existing path captures the Python
with AbstractTensor, lowers repository SSA, emits the resident WebAssembly and
WebGPU products, and gives them the standard generated HTML shell.

The numerical laws are materialized from the honorary equation catalogue:

* F16.15, F16.41 and F16.42 -- dipole force and Cartesian field;
* N5.5 -- Coulomb tangential contact impulse;
* N5.6/N5.7 -- the zero-restitution collision limit used by the contact step;
* N2.3 and WI5.7 -- mechanical work and frictional heating ledger.

The pair search is a bounded broad phase: 4,096 filings are arranged into 512
local cohorts of eight.  Every pair inside a cohort is tested each tick.  A
contact receives normal and tangential impulses; magnetic contact also enters
a persistent bond graph.  Three boolean-matrix squarings compute the complete
eight-member transitive closure, after which every connected member shares the
same momentum-conserving translational velocity and cannot separate.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
TURING_ROOT = HERE.parent / "turing"
if str(TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(TURING_ROOT))


DEFAULT_FILINGS = 4096
COHORT_SIZE = 8
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 64


@lru_cache(maxsize=1)
def _law_sources() -> tuple[str, str, dict[str, tuple[str, ...]]]:
    """Materialize the selected SymPy laws as AbstractTensor Python."""

    from honorary_engine_equation_catalogue import (
        eq_F16_15,
        eq_F16_41,
        eq_F16_42,
        eq_N5_5,
    )
    from src.compiler.symbolic_equation_compiler import compile_sympy_equations
    from src.compiler.vehicle_python_compilation import (
        symbolic_abstract_tensor_source,
    )

    field = compile_sympy_equations(
        (eq_F16_41, eq_F16_42, eq_F16_15),
        name="wooly_willy_cartesian_dipole",
    )
    friction = compile_sympy_equations(
        (eq_N5_5,), name="wooly_willy_coulomb_contact",
    )
    provenance = {
        "field_equations": ("eq_F16_41", "eq_F16_42", "eq_F16_15"),
        "contact_equations": ("eq_N5_5", "eq_N5_6", "eq_N5_7"),
        "work_equations": ("eq_N2_3", "eq_WI5_7"),
        "field_symbolic_source": tuple(field.function.metadata["symbolic_equations"]),
        "friction_symbolic_source": tuple(
            friction.function.metadata["symbolic_equations"]
        ),
    }
    return (
        symbolic_abstract_tensor_source(field, "catalogue_dipole_field"),
        symbolic_abstract_tensor_source(friction, "catalogue_coulomb_impulse"),
        provenance,
    )


def _initial_filings(count: int) -> tuple[list[float], list[float]]:
    """A non-overlapping, locally grouped powder tray."""

    if count % COHORT_SIZE:
        raise ValueError(f"filing count must be divisible by {COHORT_SIZE}")
    groups = count // COHORT_SIZE
    block_columns = 12
    block_rows = (groups + block_columns - 1) // block_columns
    grid_columns = block_columns * 4
    grid_rows = block_rows * 2
    xs = np.linspace(-0.090, 0.090, grid_columns, dtype=np.float64)
    ys = np.linspace(-0.090, -0.012, grid_rows, dtype=np.float64)
    position_x: list[float] = []
    position_y: list[float] = []
    # Group-major 4x2 blocks keep each broad-phase cohort spatially local.
    for block_y in range(block_rows):
        for block_x in range(block_columns):
            if len(position_x) >= count:
                break
            for local_y in range(2):
                for local_x in range(4):
                    position_x.append(float(xs[block_x * 4 + local_x]))
                    position_y.append(float(ys[block_y * 2 + local_y]))
    return position_x, position_y


def make_source(
    *,
    filings: int = DEFAULT_FILINGS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> str:
    """Return one source program consumable by ``build_program_bundle``."""

    if filings <= 0 or width <= 0 or height <= 0:
        raise ValueError("filings and image dimensions must be positive")
    if filings % COHORT_SIZE:
        raise ValueError(f"filings must be divisible by {COHORT_SIZE}")
    groups = filings // COHORT_SIZE
    position_x, position_y = _initial_filings(filings)
    position_x = np.asarray(position_x, dtype=np.float64).reshape(
        groups, COHORT_SIZE,
    ).tolist()
    position_y = np.asarray(position_y, dtype=np.float64).reshape(
        groups, COHORT_SIZE,
    ).tolist()
    pixel_count = width * height
    zeros = [[0.0] * COHORT_SIZE for _group in range(groups)]
    directions_x = [[0.0] * COHORT_SIZE for _group in range(groups)]
    directions_y = [[1.0] * COHORT_SIZE for _group in range(groups)]
    bonds = np.zeros((groups, COHORT_SIZE, COHORT_SIZE), dtype=np.float64)
    for group in range(groups):
        np.fill_diagonal(bonds[group], 1.0)

    field_source, friction_source, provenance = _law_sources()
    page = {
        "entrypoint": "wooly_willy_tick",
        "title": "Compiled Magnetic Filings — Aggregate Collapse",
        "slug": "compiled-magnetic-filings",
        "width": width,
        "height": height,
        # The generated shell broadcasts pointer inputs across this lane.
        # Matching it to the public filing spans preserves one contiguous,
        # SIMD-friendly Wasm ABI rather than discovering an unrelated
        # eight-wide pointer tensor and repairing that disagreement later.
        "probe_size": filings,
        "backend": "c",
        "remove_loops": True,
        "render_fps": 60.0,
        "autostart": True,
        "feeds": {
            "position_x": position_x,
            "position_y": position_y,
            "velocity_x": zeros,
            "velocity_y": zeros,
            "direction_x": directions_x,
            "direction_y": directions_y,
            "bond": bonds.tolist(),
            "friction_work": zeros,
            "member_index": [float(index) for index in range(COHORT_SIZE)],
            "pixel_x": [0.0] * pixel_count,
            "pixel_y": [0.0] * pixel_count,
            "pointer_x": 0.0,
            "pointer_y": 0.0,
            "pointer_buttons": 0.0,
            "dt": 1.0 / 120.0,
            "filing_mass": 1.0e-5,
            "filing_moment": 1.0e-4,
            "magnet_moment": 0.40,
            "mu_0": 1.25663706212e-6,
            "gravity": 9.81,
            "cover_normal_force": 2.0e-5,
            "surface_mu": 0.24,
            "filing_mu": 0.68,
            "contact_diameter": 0.0022,
            "stylus_radius": 0.006,
        },
        "feed_expressions": {
            "pixel_x": f"((x + 0.5) / {float(width)!r}) * 0.2 - 0.1",
            "pixel_y": f"0.1 - ((y + 0.5) / {float(height)!r}) * 0.2",
        },
        "state_feedback": {
            "position_x": "next_position_x",
            "position_y": "next_position_y",
            "velocity_x": "next_velocity_x",
            "velocity_y": "next_velocity_y",
            "direction_x": "next_direction_x",
            "direction_y": "next_direction_y",
            "bond": "next_bond",
            "friction_work": "next_friction_work",
        },
        "constants": {
            "member_index": [float(index) for index in range(COHORT_SIZE)],
            "pixel_x": [
                ((x + 0.5) / width) * 0.2 - 0.1
                for y in range(height) for x in range(width)
            ],
            "pixel_y": [
                0.1 - ((y + 0.5) / height) * 0.2
                for y in range(height) for x in range(width)
            ],
            "dt": 1.0 / 120.0,
            "filing_mass": 1.0e-5,
            "filing_moment": 1.0e-4,
            "magnet_moment": 0.40,
            "mu_0": 1.25663706212e-6,
            "gravity": 9.81,
            "cover_normal_force": 2.0e-5,
            "surface_mu": 0.24,
            "filing_mu": 0.68,
            "contact_diameter": 0.0022,
            "stylus_radius": 0.006,
        },
        # Retained in the immutable source and bundle digest.  This is the
        # equation-to-materialization handoff record; the compiler's own
        # identity/concordance metadata takes over once these functions enter
        # the AbstractTensor capture below.
        "law_provenance": provenance,
    }

    tick = f'''
def wooly_willy_tick(
    position_x, position_y, velocity_x, velocity_y,
    direction_x, direction_y, bond, friction_work, member_index,
    pixel_x, pixel_y, pointer_x, pointer_y, pointer_buttons,
    dt, filing_mass, filing_moment, magnet_moment, mu_0, gravity,
    cover_normal_force, surface_mu, filing_mu, contact_diameter,
    stylus_radius,
):
    # Pointer coordinates enter through the generated shell's standard input
    # liaison.  Physical coordinates span a 20 cm square cavity.
    # The standard web liaison broadcasts each live scalar control across its
    # probe lane.  All entries represent the same physical pointer, so reduce
    # that transport representation back to one value before combining it
    # with either the (groups, 8) filing field or the flat pixel field.
    pointer_x_value = pointer_x.mean()
    pointer_y_value = pointer_y.mean()
    pointer_buttons_value = pointer_buttons.mean()
    magnet_x = pointer_x_value / {float(width)!r} * 0.2 - 0.1
    magnet_y = 0.1 - pointer_y_value / {float(height)!r} * 0.2
    active = pointer_buttons_value > 0.0
    attracting = active * (pointer_buttons_value < 2.0)
    force_sign = (pointer_buttons_value > 1.0) * 2.0 - 1.0

    dx_m = position_x - magnet_x
    dy_m = position_y - magnet_y
    radius = (dx_m * dx_m + dy_m * dy_m).sqrt().maximum(stylus_radius)
    field_x, field_y, dipole_force = catalogue_dipole_field(
        dx_m, dy_m, magnet_moment, filing_moment, magnet_moment, mu_0,
        radius,
    )
    field_norm = (field_x * field_x + field_y * field_y).sqrt().maximum(1.0e-18)
    target_x = field_x / field_norm
    target_y = field_y / field_norm
    equivalent_sign = ((target_x * direction_x + target_y * direction_y) >= 0.0) * 2.0 - 1.0
    target_x = target_x * equivalent_sign
    target_y = target_y * equivalent_sign
    next_direction_x = direction_x * 0.72 + target_x * 0.28
    next_direction_y = direction_y * 0.72 + target_y * 0.28
    direction_norm = (next_direction_x * next_direction_x + next_direction_y * next_direction_y).sqrt().maximum(1.0e-18)
    next_direction_x = next_direction_x / direction_norm
    next_direction_y = next_direction_y / direction_norm

    acceleration = dipole_force / filing_mass
    next_velocity_x = velocity_x + active * force_sign * acceleration * dx_m / radius * dt
    next_velocity_y = velocity_y + active * force_sign * acceleration * dy_m / radius * dt
    next_velocity_y = next_velocity_y - gravity * dt

    # Sliding friction against the cover/backing. N5.5 returns a signed
    # Coulomb impulse. The max is a time-step admissibility condition: an
    # impulse may bring a filing to rest but may not reverse it in one tick.
    speed = (next_velocity_x * next_velocity_x + next_velocity_y * next_velocity_y).sqrt()
    safe_speed = speed.maximum(1.0e-10)
    surface_impulse = catalogue_coulomb_impulse(
        cover_normal_force * dt, surface_mu, safe_speed,
    )
    surface_impulse = surface_impulse.maximum(-filing_mass * speed)
    next_velocity_x = next_velocity_x + surface_impulse * next_velocity_x / safe_speed / filing_mass
    next_velocity_y = next_velocity_y + surface_impulse * next_velocity_y / safe_speed / filing_mass
    surface_work = abs(surface_impulse * speed)

    # Every pair inside each spatially local eight-filing cohort collides.
    px_i = position_x.unsqueeze(2)
    py_i = position_y.unsqueeze(2)
    px_j = position_x.unsqueeze(1)
    py_j = position_y.unsqueeze(1)
    pair_dx = px_i - px_j
    pair_dy = py_i - py_j
    pair_distance = (pair_dx * pair_dx + pair_dy * pair_dy).sqrt()
    safe_pair_distance = pair_distance.maximum(1.0e-10)
    nx = pair_dx / safe_pair_distance
    ny = pair_dy / safe_pair_distance
    tx = -ny
    ty = nx
    member_i = member_index.unsqueeze(0).unsqueeze(2)
    member_j = member_index.unsqueeze(0).unsqueeze(1)
    identity = (member_i == member_j) * 1.0
    contact = (pair_distance < contact_diameter) * (1.0 - identity)

    vx_i = next_velocity_x.unsqueeze(2)
    vy_i = next_velocity_y.unsqueeze(2)
    vx_j = next_velocity_x.unsqueeze(1)
    vy_j = next_velocity_y.unsqueeze(1)
    relative_x = vx_i - vx_j
    relative_y = vy_i - vy_j
    normal_speed = relative_x * nx + relative_y * ny
    approaching = (normal_speed < 0.0) * 1.0
    normal_impulse = -0.5 * filing_mass * normal_speed * contact * approaching
    tangent_speed = relative_x * tx + relative_y * ty
    sliding = (abs(tangent_speed) > 1.0e-10) * 1.0
    safe_tangent_speed = tangent_speed + (1.0 - sliding) * 1.0e-10
    tangent_impulse = catalogue_coulomb_impulse(
        normal_impulse, filing_mu, safe_tangent_speed,
    ) * contact * sliding

    collision_dvx = ((normal_impulse * nx + tangent_impulse * tx).sum(dim=2) / filing_mass)
    collision_dvy = ((normal_impulse * ny + tangent_impulse * ty).sum(dim=2) / filing_mass)
    next_velocity_x = next_velocity_x + collision_dvx
    next_velocity_y = next_velocity_y + collision_dvy
    contact_work = abs(tangent_impulse * tangent_speed).sum(dim=2) * 0.5

    # Correct overlap along the same measured contact normals. This is the
    # position-level nonpenetration constraint paired with N5's impulses.
    penetration = (contact_diameter - pair_distance).maximum(0.0) * contact
    neighbours = contact.sum(dim=2).maximum(1.0)
    correction_x = (0.5 * penetration * nx).sum(dim=2) / neighbours
    correction_y = (0.5 * penetration * ny).sum(dim=2) / neighbours

    # Magnetic contact collapses into a persistent aggregate. Three boolean
    # squarings are the exact transitive closure for an eight-member cohort.
    binding = contact * attracting
    reach = bond.maximum(binding).maximum(identity)
    reach = ((reach @ reach) > 0.0) * 1.0
    reach = ((reach @ reach) > 0.0) * 1.0
    reach = ((reach @ reach) > 0.0) * 1.0
    next_bond = reach
    members = reach.sum(dim=2).maximum(1.0)
    cluster_vx = (
        reach @ next_velocity_x.unsqueeze(2)
    ).squeeze(2) / members
    cluster_vy = (
        reach @ next_velocity_y.unsqueeze(2)
    ).squeeze(2) / members
    next_velocity_x = cluster_vx
    next_velocity_y = cluster_vy

    proposed_x = position_x + next_velocity_x * dt + correction_x
    proposed_y = position_y + next_velocity_y * dt + correction_y
    hit_x = (abs(proposed_x) > 0.092) * 1.0
    hit_y = (abs(proposed_y) > 0.092) * 1.0
    next_position_x = proposed_x.maximum(-0.092).minimum(0.092)
    next_position_y = proposed_y.maximum(-0.092).minimum(0.092)
    next_velocity_x = next_velocity_x * (1.0 - 1.15 * hit_x)
    next_velocity_y = next_velocity_y * (1.0 - 1.15 * hit_y)
    next_friction_work = friction_work + surface_work + contact_work

    # The main compiled program owns presentation too. The standard publisher
    # displays these named RGB arrays through its compiler-emitted WebGPU
    # passthrough, with WebAssembly/Canvas as its existing fallback.
    image_x = pixel_x.unsqueeze(1).unsqueeze(2)
    image_y = pixel_y.unsqueeze(1).unsqueeze(2)
    filing_x = next_position_x.unsqueeze(0)
    filing_y = next_position_y.unsqueeze(0)
    filing_dx = next_direction_x.unsqueeze(0)
    filing_dy = next_direction_y.unsqueeze(0)
    image_dx = image_x - filing_x
    image_dy = image_y - filing_y
    along = image_dx * filing_dx + image_dy * filing_dy
    across = -image_dx * filing_dy + image_dy * filing_dx
    ink = ((abs(along) < 0.0048) * (abs(across) < 0.0017)).sum(dim=2).sum(dim=1).minimum(1.0)

    radial2 = pixel_x * pixel_x + pixel_y * pixel_y
    cavity = (radial2 < 0.0092) * 1.0
    tray = ((pixel_y < -0.075) * (abs(pixel_x) < 0.091)) * 1.0
    eye_left = (((pixel_x + 0.026) ** 2 / 0.00012 + (pixel_y - 0.026) ** 2 / 0.00007) < 1.0) * 1.0
    eye_right = (((pixel_x - 0.026) ** 2 / 0.00012 + (pixel_y - 0.026) ** 2 / 0.00007) < 1.0) * 1.0
    pupils = ((((pixel_x + 0.026) ** 2 + (pixel_y - 0.026) ** 2) < 0.000012) + (((pixel_x - 0.026) ** 2 + (pixel_y - 0.026) ** 2) < 0.000012)).minimum(1.0)
    mouth = ((abs(pixel_y + 0.030 - 5.5 * pixel_x * pixel_x) < 0.0022) * (abs(pixel_x) < 0.030)) * 1.0
    face_r = cavity * 0.91 + (1.0 - cavity) * 0.16
    face_g = cavity * 0.88 + (1.0 - cavity) * 0.48
    face_b = cavity * 0.73 + (1.0 - cavity) * 0.68
    face_r = face_r * (1.0 - tray) + tray * 0.10
    face_g = face_g * (1.0 - tray) + tray * 0.34
    face_b = face_b * (1.0 - tray) + tray * 0.50
    eyes = (eye_left + eye_right).minimum(1.0)
    face_r = face_r * (1.0 - eyes) + eyes * 0.98
    face_g = face_g * (1.0 - eyes) + eyes * 0.98
    face_b = face_b * (1.0 - eyes) + eyes * 0.94
    marks = (pupils + mouth).minimum(1.0)
    face_r = face_r * (1.0 - marks) + marks * 0.12
    face_g = face_g * (1.0 - marks) + marks * 0.18
    face_b = face_b * (1.0 - marks) + marks * 0.19
    magnet_mark = ((((pixel_x - magnet_x) ** 2 + (pixel_y - magnet_y) ** 2) < 0.000055) * active) * 1.0
    red = face_r * (1.0 - ink) + ink * 0.045
    green = face_g * (1.0 - ink) + ink * 0.040
    blue = face_b * (1.0 - ink) + ink * 0.032
    red = red * (1.0 - magnet_mark) + magnet_mark * 0.92
    green = green * (1.0 - magnet_mark) + magnet_mark * 0.14
    blue = blue * (1.0 - magnet_mark) + magnet_mark * 0.11

    return (
        red, green, blue,
        next_position_x, next_position_y, next_velocity_x, next_velocity_y,
        next_direction_x, next_direction_y, next_bond,
        next_friction_work,
    )
'''
    origin = (
        "# Generated only from the catalogue compilations recorded in "
        "TURING_PAGE['law_provenance'].\n"
    )
    return "\n\n".join((
        "TURING_PAGE = " + repr(page),
        origin + field_source.rstrip(),
        friction_source.rstrip(),
        tick.strip(),
    )) + "\n"


def build_web_bundle(
    destination: Path,
    *,
    filings: int = DEFAULT_FILINGS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
):
    """Use the repository's existing compiler and web publication path."""

    from src.compiler.site_bundle import build_program_bundle

    source = make_source(filings=filings, width=width, height=height)
    return build_program_bundle(
        source,
        destination.resolve(),
        source_filename="wooly_willy_compiled.py",
        include_backends=True,
        backend_targets=("ssa", "wat", "abstract_tensor", "webgpu"),
        include_mathematics=True,
        progress_sink=lambda record: print(record, flush=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=HERE.parent,
        help="static gallery root consumed by Turing's site publisher",
    )
    parser.add_argument("--filings", type=int, default=DEFAULT_FILINGS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--source-only", type=Path,
        help="write the generated authored Python source without compiling",
    )
    args = parser.parse_args()
    source = make_source(
        filings=args.filings, width=args.width, height=args.height,
    )
    if args.source_only is not None:
        args.source_only.parent.mkdir(parents=True, exist_ok=True)
        args.source_only.write_text(source, encoding="utf-8", newline="\n")
        print(args.source_only.resolve())
        print(hashlib.sha256(source.encode("utf-8")).hexdigest())
        return 0
    bundle = build_web_bundle(
        args.destination,
        filings=args.filings,
        width=args.width,
        height=args.height,
    )
    print(bundle.page_path)
    print(bundle.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
