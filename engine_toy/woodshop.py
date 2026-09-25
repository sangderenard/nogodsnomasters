"""First playable woodshop slice: real Machines under the managed dt API.

The object model in this module is deliberately the existing ``Machine``
model.  A stud is one solid-part Machine.  A hand saw is a two-part Machine
with one declared working edge and one held-click action.  Inventory and two
hands only hold their identities; they do not replace them with UI objects.

Coordinates use the ordinary shop drawing convention for this small top-down
slice: X/Y are the work plane and the stock rests on Z=0.  The rest of engine
toy is free to view the same graph in 3-D because every part retains a full
position and box extent.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import pickle
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp

import honorary_engine_equation_catalogue as honorary
from honorary_engine_equation_catalogue import equation_piece, law_pieces
from material_topology import (
    Disincorporation,
    SdfMaterialBody,
    WorldObjectGraph,
    chip_batch,
    disincorporate_sdf_body,
    update_chip_batch,
)
from sdf_geometry import CapsuleSdf
from machines import (
    Machine,
    MachineAction,
    MachinePart,
    MachinePose,
    MachineSim,
    MachineSystem,
    MachineWorkEdge,
)
from src.common.dt_system.engine_api import DtCompatibleEngine, IdentityAssembly
from src.common.dt_system.error_channels import empty_channels
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.time_contracts import HOLD
from src.common.dt_system.dt_scaler import Metrics
from src.common.tensors import AbstractTensor
from woodworking_joints import (
    CompressionSleeveInterface,
    CompressionSleeveJoint,
    WoodworkingJoint,
    WoodworkingJointInterface,
    bar_clamp,
    clamp_spec,
    pipe_clamp,
    resin_sawhorse_bracket,
)

_TURING_EXAMPLES = Path(__file__).resolve().parents[1] / "turing" / "examples"
if str(_TURING_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_TURING_EXAMPLES))

from llvm_dt_system import dt_system_from_graph, lowered_system, piece_leaf
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import ControllerNode, RoundNode


INCH_M = 0.0254
FOOT_M = 0.3048
NOMINAL_STUD_LENGTH_M = 8.0 * FOOT_M
STARTING_STUD_COUNT = 8
STARTING_SAWHORSE_BRACKET_COUNT = 4
STUD_WIDTH_M = 3.5 * INCH_M
STUD_THICKNESS_M = 1.5 * INCH_M
EYE_HEIGHT_M = 1.62
GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11
EARTH_MASS_KG = 5.9722e24
EARTH_RADIUS_M = 6_371_000.0


def newton_world_dt_pieces(batch: int):
    """Manifest N4.1 -> N1.2 -> N1.1 as three batched LLVM pieces.

    The expressions below are substitutions into the catalogue equations,
    followed only by symplectic-Euler time discretization.  Each piece sees
    every world-object lane in one call; no law is invoked once per Machine.
    """
    active, dt = sp.symbols("active dt")
    mass, earth_mass = sp.symbols("mass earth_mass")
    earth_radius, position_z = sp.symbols("earth_radius position_z")
    gravity_constant = sp.Symbol("gravity_constant")
    force_x, force_y, force_z = sp.symbols("force_x force_y force_z")
    momentum_x, momentum_y, momentum_z = sp.symbols(
        "momentum_x momentum_y momentum_z")
    position_x, position_y = sp.symbols("position_x position_y")

    gravity_rhs = honorary.eq_N4_1.rhs.xreplace({
        honorary.G: gravity_constant,
        honorary.m_i(honorary.t): mass,
        honorary.m_j: earth_mass,
        honorary.x_i(honorary.t): earth_radius + position_z,
        honorary.x_j: sp.Integer(0),
    })
    gravity = equation_piece("woodshop_newton_gravity", (
        sp.Eq(sp.Symbol("force_z_next"), active * gravity_rhs,
              evaluate=False),
    ), batch=batch)

    def momentum_next(momentum, force):
        derivative = honorary.eq_N1_2.rhs.xreplace({
            honorary.F_i(honorary.t): force,
        })
        return momentum + dt * derivative

    momentum = equation_piece("woodshop_newton_momentum", (
        sp.Eq(sp.Symbol("momentum_x_next"),
              momentum_next(momentum_x, force_x), evaluate=False),
        sp.Eq(sp.Symbol("momentum_y_next"),
              momentum_next(momentum_y, force_y), evaluate=False),
        sp.Eq(sp.Symbol("momentum_z_next"),
              momentum_next(momentum_z, force_z), evaluate=False),
    ), batch=batch)

    def position_next(position, momentum):
        velocity = honorary.eq_N1_1.rhs.xreplace({
            honorary.m_i(honorary.t): mass,
            honorary.p_i(honorary.t): momentum,
        })
        return position + active * dt * velocity

    speed = active * sp.sqrt(
        momentum_x**2 + momentum_y**2 + momentum_z**2) / mass
    position = equation_piece("woodshop_newton_position", (
        sp.Eq(sp.Symbol("position_x_next"),
              position_next(position_x, momentum_x), evaluate=False),
        sp.Eq(sp.Symbol("position_y_next"),
              position_next(position_y, momentum_y), evaluate=False),
        sp.Eq(sp.Symbol("position_z_next"),
              position_next(position_z, momentum_z), evaluate=False),
        sp.Eq(sp.Symbol("max_vel"), speed, evaluate=False),
    ), batch=batch)
    return gravity, momentum, position


def euler_matrix_deg(rotation_deg_xyz: tuple[float, float, float]) -> np.ndarray:
    """Right-handed intrinsic XYZ object rotation."""
    rx, ry, rz = np.radians(np.asarray(rotation_deg_xyz, dtype=float))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=float)
    my = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=float)
    mz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=float)
    return mz @ my @ mx


class ToothPattern(str, Enum):
    RIP_CHISEL = "rip-chisel"
    CROSSCUT_ALTERNATING_BEVEL = "crosscut-alternating-bevel"
    UNIVERSAL = "universal"
    HACKSAW_WAVY_SET = "hacksaw-wavy-set"


@dataclass(frozen=True)
class OrthotropicWood:
    """Material data in the wood's local longitudinal/radial/tangential axes."""

    identity: str
    label: str
    moisture_content: float
    density_kg_m3: float
    youngs_l_pa: float
    youngs_r_pa: float
    youngs_t_pa: float
    shear_lr_pa: float
    shear_lt_pa: float
    shear_rt_pa: float
    poisson_lr: float
    poisson_lt: float
    poisson_rt: float
    cutting_energy_j_m3: float
    cutting_coefficient_pa: float
    edge_coefficient_n_m: float
    provenance: str

    def compliance_matrix(self) -> np.ndarray:
        """Symmetric 3-D engineering compliance in L/R/T order."""

        e_l, e_r, e_t = self.youngs_l_pa, self.youngs_r_pa, self.youngs_t_pa
        nu_rl = self.poisson_lr * e_r / e_l
        nu_tl = self.poisson_lt * e_t / e_l
        nu_tr = self.poisson_rt * e_t / e_r
        return np.asarray([
            [1 / e_l, -nu_rl / e_r, -nu_tl / e_t, 0, 0, 0],
            [-self.poisson_lr / e_l, 1 / e_r, -nu_tr / e_t, 0, 0, 0],
            [-self.poisson_lt / e_l, -self.poisson_rt / e_r, 1 / e_t, 0, 0, 0],
            [0, 0, 0, 1 / self.shear_rt_pa, 0, 0],
            [0, 0, 0, 0, 1 / self.shear_lt_pa, 0],
            [0, 0, 0, 0, 0, 1 / self.shear_lr_pa],
        ], dtype=float)


# A runnable first calibration, intentionally identified as such rather than
# presented as the design value for every species sold as "pine".  Elastic
# ratios follow the USDA Wood Handbook's orthotropic clear-wood convention;
# the cutting coefficients are the authored hand-saw calibration to replace
# when a selected stock/tool pair has measurements.
KILN_DRY_PINE_STUD = OrthotropicWood(
    identity="kiln-dry-pine-stud-v0",
    label="kiln-dried pine framing stud, 12% MC",
    moisture_content=0.12,
    density_kg_m3=450.0,
    youngs_l_pa=9.5e9,
    youngs_r_pa=0.78e9,
    youngs_t_pa=0.43e9,
    shear_lr_pa=0.72e9,
    shear_lt_pa=0.58e9,
    shear_rt_pa=0.075e9,
    poisson_lr=0.34,
    poisson_lt=0.36,
    poisson_rt=0.43,
    cutting_energy_j_m3=8.0e6,
    cutting_coefficient_pa=8.0e6,
    edge_coefficient_n_m=40.0,
    provenance=(
        "USDA Wood Handbook orthotropic form; initial authored pine/saw "
        "cutting calibration, not a structural design value"
    ),
)


def kiln_dried_pine_2x4x8(identity: str = "stock.pine-stud.001") -> Machine:
    volume = NOMINAL_STUD_LENGTH_M * STUD_WIDTH_M * STUD_THICKNESS_M
    stock = Machine(
        identity=identity,
        label="kiln-dried nominal 2x4x8 pine stud",
        power="human-carried",
        medium="solid",
        note="actual dressed section 1.5 x 3.5 inches; grain axis +X",
        interaction_poses=[
            MachinePose("selected", (0.0, -0.42, 0.78), (0.0, 0.0, 0.0)),
            MachinePose("used-1", (0.0, -0.39, 0.82), (0.0, 0.0, 0.0)),
            MachinePose("used-2", (0.0, -0.44, 0.74), (0.0, 0.0, 0.0)),
        ],
    )
    stock.parts = [MachinePart(
        f"{identity}.wood",
        "solid-wood-stock",
        (0.0, 0.0, STUD_THICKNESS_M / 2.0),
        (NOMINAL_STUD_LENGTH_M / 2.0, STUD_WIDTH_M / 2.0,
         STUD_THICKNESS_M / 2.0),
        mass_kg=volume * KILN_DRY_PINE_STUD.density_kg_m3,
        material=KILN_DRY_PINE_STUD.identity,
        part_role="workpiece",
        attributes={
            "nominal_dimensions_in": [2.0, 4.0, 96.0],
            "dressed_dimensions_m": [STUD_THICKNESS_M, STUD_WIDTH_M,
                                      NOMINAL_STUD_LENGTH_M],
            "grain_axis_local": [1.0, 0.0, 0.0],
            "radial_axis_local": [0.0, 0.0, 1.0],
            "tangential_axis_local": [0.0, 1.0, 0.0],
            "moisture_content": KILN_DRY_PINE_STUD.moisture_content,
            "material_data": KILN_DRY_PINE_STUD.identity,
        },
    )]
    return stock


def hand_saw(identity: str = "tool.hand-saw.001") -> Machine:
    saw = Machine(
        identity=identity,
        label="550 mm crosscut hand saw",
        power="human-held",
        medium="manual",
        note="two bodies and one finite kerf-generating working edge",
        interaction_poses=[
            MachinePose("selected", (0.0, -0.34, 0.48), (0.0, 0.0, 0.0)),
            MachinePose("used-1", (0.0, -0.29, 0.78), (0.0, -10.0, 0.0)),
            MachinePose("used-2", (0.0, -0.31, 0.53), (0.0, 7.0, 0.0)),
        ],
    )
    saw.parts = [
        MachinePart(
            f"{identity}.blade", "saw-blade", (0.0, -0.48, 0.0115),
            (0.30, 0.035, 0.0005), mass_kg=0.33,
            material="hardened-steel", part_role="blade",
            attributes={"youngs_modulus_pa": 205e9, "grain_axis": None},
        ),
        MachinePart(
            f"{identity}.handle", "saw-handle", (-0.375, -0.48, 0.025),
            (0.075, 0.055, 0.025), mass_kg=0.24,
            material="kiln-dry-hardwood-handle", part_role="handle",
        ),
    ]
    edge_identity = f"{identity}.work-edge"
    saw.work_edges = [MachineWorkEdge(
        edge_identity,
        owner=f"{identity}.blade",
        a_local=(-0.29, -0.035, 0.0),
        b_local=(0.29, -0.035, 0.0),
        thickness_m=0.0009,
        wave_amplitude_m=0.00045,
        wave_length_m=0.008,
        tooth_pattern=ToothPattern.CROSSCUT_ALTERNATING_BEVEL.value,
        attributes={
            "stroke_length_m": 0.45,
            "stroke_frequency_hz": 1.2,
            "engaged_teeth": 10,
            "feed_per_stroke_m": 0.00030,
            "cutting_efficiency": 0.12,
            "honorary_laws": [
                "eq_WO4_1", "eq_WO4_2", "eq_WO4_3", "eq_WO4_4",
                "eq_WO4_5", "eq_N2_3", "eq_T1_1",
            ],
        },
    )]
    saw.actions = [MachineAction(
        f"{identity}.actions.saw",
        gesture="held-primary",
        operation="advance-kerf",
        destination=edge_identity,
        attributes={"release_operation": "stop-kerf-work"},
    )]
    return saw


@dataclass(frozen=True)
class BoxMesh:
    vertices: np.ndarray
    triangles: np.ndarray


def part_box_mesh(part: MachinePart) -> BoxMesh:
    """Triangulated box derived directly from a MachinePart's real extent."""

    c = np.asarray(part.position, dtype=float)
    h = np.asarray(part.half_extent_m, dtype=float)
    vertices = np.asarray([
        c + (sx * h[0], sy * h[1], sz * h[2])
        for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)
    ], dtype=float)
    triangles = np.asarray([
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
    ], dtype=np.int32)
    return BoxMesh(vertices, triangles)


@dataclass(frozen=True)
class CutProjection:
    target: str
    intended_a: tuple[float, float, float]
    intended_b: tuple[float, float, float]
    actual_a: tuple[float, float, float]
    actual_b: tuple[float, float, float]
    support_quality: float
    longitudinal_error_m: float
    angle_error_rad: float


def _stable_phase(*tokens: str) -> float:
    digest = hashlib.sha256("\0".join(tokens).encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer / float(2**64 - 1)) * math.tau


def project_crosscut(
    stock: Machine,
    point_xy: tuple[float, float],
    *,
    support_quality: float,
    attempt_identity: str,
) -> CutProjection:
    """Project a crosscut onto the top triangles of a rectangular wood part."""

    part = next(p for p in stock.parts if p.part_role == "workpiece")
    mesh = part_box_mesh(part)
    lo = mesh.vertices.min(axis=0)
    hi = mesh.vertices.max(axis=0)
    x = min(max(float(point_xy[0]), lo[0]), hi[0])
    intended_a = np.asarray((x, lo[1], hi[2]), dtype=float)
    intended_b = np.asarray((x, hi[1], hi[2]), dtype=float)

    quality = min(max(float(support_quality), 0.0), 1.0)
    phase = _stable_phase(stock.identity, attempt_identity)
    # A hand-held eight-foot stud is a poor fixture.  These are explicit
    # operator/support parameters, not a material-strength multiplier.
    longitudinal_error = (1.0 - quality) * 0.025 * math.sin(phase)
    angle_error = (1.0 - quality) * math.radians(14.0) * math.cos(phase)
    center = (intended_a + intended_b) * 0.5
    center[0] += longitudinal_error
    half_width = (hi[1] - lo[1]) * 0.5
    across = np.asarray((-math.sin(angle_error), math.cos(angle_error), 0.0))
    actual_a = center - across * half_width
    actual_b = center + across * half_width
    actual_a[2] = actual_b[2] = hi[2]
    return CutProjection(
        stock.identity, tuple(intended_a), tuple(intended_b),
        tuple(actual_a), tuple(actual_b), quality,
        longitudinal_error, angle_error,
    )


@dataclass
class Kerf:
    identity: str
    target: str
    tool: str
    work_edge: str
    intended_a: tuple[float, float, float]
    intended_b: tuple[float, float, float]
    actual_a: tuple[float, float, float]
    actual_b: tuple[float, float, float]
    width_m: float
    depth_m: float = 0.0
    through_depth_m: float = STUD_THICKNESS_M
    support_quality: float = 1.0
    removed_volume_m3: float = 0.0
    removed_mass_kg: float = 0.0
    work_j: float = 0.0
    complete: bool = False


@dataclass
class CuttingState:
    kerfs: list[Kerf] = field(default_factory=list)
    active_kerf: str | None = None
    held: bool = False
    elapsed_s: float = 0.0


def _run_law(piece, **values) -> float:
    columns = tuple(
        np.asarray([values[name]], dtype=np.float64)
        for name in piece.argument_names
    )
    return float(np.asarray(piece(*columns)[0], dtype=np.float64).reshape(-1)[0])


class WoodCuttingSystem(DtCompatibleEngine):
    """Managed-dt cutting engine operating on Machine work edges and stock."""

    def __init__(self, machines: dict[str, Machine], state: CuttingState | None = None):
        self.machines = machines
        self.state = state or CuttingState()
        self.world_time = 0.0
        self.observer_time = 0.0
        self.last_metrics = None
        self._laws = law_pieces(
            ("Woodshop",),
            law_ids=tuple(f"eq_WO4_{number}" for number in range(1, 6)),
        )
        self.material_bodies: dict[str, SdfMaterialBody] = {}

    def begin(self, tool: str, projection: CutProjection) -> Kerf:
        saw = self.machines[tool]
        edge = saw.work_edges[0]
        identity = f"{projection.target}.kerf.{len(self.state.kerfs) + 1:03d}"
        kerf = Kerf(
            identity, projection.target, tool, edge.identity,
            projection.intended_a, projection.intended_b,
            projection.actual_a, projection.actual_b,
            width_m=edge.thickness_m + 2.0 * edge.wave_amplitude_m,
            support_quality=projection.support_quality,
        )
        target_part = next(
            p for p in self.machines[projection.target].parts
            if p.part_role == "workpiece"
        )
        kerf.through_depth_m = 2.0 * float(target_part.half_extent_m[2])
        if target_part.identity not in self.material_bodies:
            body = SdfMaterialBody.from_part(
                target_part, KILN_DRY_PINE_STUD.density_kg_m3,
                kerf.width_m,
            )
            self.material_bodies[target_part.identity] = body
            target_part.mass_kg = body.mass_kg
        self.state.kerfs.append(kerf)
        self.state.active_kerf = kerf.identity
        return kerf

    def set_held(self, held: bool) -> None:
        self.state.held = bool(held)
        if not held:
            self.state.active_kerf = None

    def active(self) -> Kerf | None:
        return next((k for k in self.state.kerfs
                     if k.identity == self.state.active_kerf), None)

    def step(self, dt: float, state=None, state_table=None):
        state = self.state if state is None else state
        state.elapsed_s += float(dt)
        kerf = self.active()
        kerf_velocity = 0.0
        if self.state.held and kerf is not None and not kerf.complete:
            saw = self.machines[kerf.tool]
            edge = next(e for e in saw.work_edges if e.identity == kerf.work_edge)
            stock = self.machines[kerf.target]
            part = next(p for p in stock.parts if p.part_role == "workpiece")
            attr = edge.attributes
            h_chip = float(attr["feed_per_stroke_m"]) / float(attr["engaged_teeth"])
            cut_width = 2.0 * float(part.half_extent_m[1])
            force = _run_law(
                self._laws["eq_WO4_1"], K_tc=KILN_DRY_PINE_STUD.cutting_coefficient_pa,
                h_chip=h_chip, K_te=KILN_DRY_PINE_STUD.edge_coefficient_n_m,
                b_cut=cut_width,
            )
            edge_speed = _run_law(
                self._laws["eq_WO4_2"], L_stroke=float(attr["stroke_length_m"]),
                f_stroke=float(attr["stroke_frequency_hz"]),
            )
            power = _run_law(self._laws["eq_WO4_3"], F_cut=force, v_edge=edge_speed)
            # Fixture quality affects how much hand work arrives at the tooth;
            # the missing work remains rigid-body motion/rebound, not vanished
            # material strength.
            efficiency = float(attr["cutting_efficiency"]) * (
                0.2 + 0.8 * kerf.support_quality
            )
            removed_rate = _run_law(
                self._laws["eq_WO4_4"], eta_cut=efficiency, P_cut=power,
                u_specific=KILN_DRY_PINE_STUD.cutting_energy_j_m3,
            )
            kerf_velocity = _run_law(
                self._laws["eq_WO4_5"], Vdot_cut=removed_rate,
                w_kerf=kerf.width_m, b_work=cut_width,
            )
            old_depth = kerf.depth_m
            advance = min(kerf_velocity * float(dt),
                          kerf.through_depth_m - kerf.depth_m)
            kerf.depth_m += advance
            body = self.material_bodies[part.identity]
            radius = kerf.width_m * 0.5
            # A tooth stroke is a finite capsule SDF.  Sample a long advance
            # no farther apart than one radius so a large dt cannot tunnel
            # through the material field.
            samples = max(1, int(math.ceil(max(advance, 1.0e-15)
                                           / max(radius, 1.0e-12))))
            top_z = float(part.position[2] + part.half_extent_m[2])
            sweeps = []
            for sample in range(1, samples + 1):
                depth = old_depth + advance * sample / samples
                z = top_z - depth
                sweeps.append(CapsuleSdf(
                    (kerf.actual_a[0], kerf.actual_a[1], z),
                    (kerf.actual_b[0], kerf.actual_b[1], z),
                    radius,
                ))
            body.subtract(sweeps)
            kerf.removed_volume_m3 = body.removed_volume_m3
            kerf.removed_mass_kg = body.removed_mass_kg
            part.mass_kg = body.mass_kg
            kerf.work_j += power * float(dt)
            # Depth alone never creates a second object. The final stroke has
            # to remove the last material bridge and leave separate islands.
            if kerf.depth_m >= kerf.through_depth_m - 2.0 * radius:
                _labels, island_count = body.material_islands()
                kerf.complete = island_count > 1
            if kerf.complete:
                self.state.held = False
                self.state.active_kerf = None

        if state_table is not None:
            state_table.set("woodshop", "cutting", "kerfs", copy.deepcopy(state.kerfs))
            state_table.set("woodshop", "cutting", "active_kerf", state.active_kerf)
            if kerf is not None:
                target_part = next(
                    p for p in self.machines[kerf.target].parts
                    if p.part_role == "workpiece"
                )
                for uuid_str, identity in state_table.identity_registry.items():
                    if identity.get("semantic_identity") == target_part.identity:
                        state_table.update_identity(uuid_str, mass=target_part.mass_kg)

        channels = empty_channels()
        metrics = Metrics(
            max_vel=kerf_velocity, max_flux=0.0, div_inf=0.0, mass_err=0.0,
            pub_exchange_time=AbstractTensor.tensor([0.0]),
            pub_exchange_time_present=AbstractTensor.tensor([0.0]),
            pub_contract=AbstractTensor.tensor([HOLD]),
            pub_dt_limit=AbstractTensor.tensor([0.0]),
            pub_dt_limit_present=AbstractTensor.tensor([0.0]),
            pub_values=channels.copy(),
            pub_present=AbstractTensor.zeros_like(channels),
            pub_limits=AbstractTensor.zeros_like(channels),
            pub_limits_present=AbstractTensor.zeros_like(channels),
            advanced_dt=float(dt),
        )
        self.last_metrics = metrics
        return True, metrics, state

    def get_state(self, state=None):
        return self.state if state is None else state

    def snapshot(self):
        masses = {
            part.identity: float(part.mass_kg)
            for machine in self.machines.values() for part in machine.parts
        }
        return (copy.deepcopy(self.state), masses,
                copy.deepcopy(self.material_bodies), float(self.world_time),
                float(self.observer_time), self.last_metrics)

    def restore(self, snapshot) -> None:
        state, masses, bodies, world_time, observer_time, metrics = copy.deepcopy(snapshot)
        self.state = state
        self.material_bodies = bodies
        for machine in self.machines.values():
            for part in machine.parts:
                if part.identity in masses:
                    part.mass_kg = masses[part.identity]
        self.world_time = world_time
        self.observer_time = observer_time
        self.last_metrics = metrics

    def translate_target(self, target: str, delta_xyz) -> None:
        """Move retained cut geometry with the Machine that owns it."""

        supplied = tuple(float(value) for value in delta_xyz)
        delta = supplied if len(supplied) == 3 else (supplied[0], supplied[1], 0.0)
        for kerf in self.state.kerfs:
            if kerf.target != target:
                continue
            for name in ("intended_a", "intended_b", "actual_a", "actual_b"):
                point = getattr(kerf, name)
                setattr(kerf, name, tuple(point[i] + delta[i] for i in range(3)))
        machine = self.machines.get(target)
        if machine is None:
            return
        delta3 = np.asarray(delta, dtype=float)
        for part in machine.parts:
            body = self.material_bodies.get(part.identity)
            if body is None:
                continue
            body.kernel.bounds_min += delta3
            body.kernel.bounds_max += delta3
            body.kernel.triangles += delta3
            body.kernel.cutouts = [CapsuleSdf(
                tuple(np.asarray(c.start) + delta3),
                tuple(np.asarray(c.end) + delta3), c.radius_m,
            ) for c in body.kernel.cutouts]


@dataclass
class WorldMachine:
    sim: MachineSim
    custody: str = "world"  # world | inventory
    slot: int | None = None
    pose_state: str = "placed"
    orientation_deg_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_momentum_kg_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def identity(self) -> str:
        return self.sim.machine.identity

    def bounds_xy(self) -> tuple[float, float, float, float]:
        lo, hi = self.bounds_xyz()
        return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])

    def bounds_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        """World-axis bounds of the Machine's oriented real part boxes."""
        lo = np.asarray((math.inf, math.inf, math.inf), dtype=float)
        hi = np.asarray((-math.inf, -math.inf, -math.inf), dtype=float)
        for _part, part_lo, part_hi in self.part_bounds_xyz():
            lo = np.minimum(lo, part_lo)
            hi = np.maximum(hi, part_hi)
        return lo, hi

    def part_bounds_xyz(self):
        """World-axis bounds for each oriented real MachinePart box."""
        pivot = self.center_xyz()
        rotation = self.rotation_matrix()
        for part in self.sim.machine.parts:
            lo = np.asarray((math.inf, math.inf, math.inf), dtype=float)
            hi = np.asarray((-math.inf, -math.inf, -math.inf), dtype=float)
            center = pivot + rotation @ (np.asarray(part.position, dtype=float) - pivot)
            half = np.asarray(part.half_extent_m, dtype=float)
            for signs in ((sx, sy, sz) for sx in (-1, 1)
                          for sy in (-1, 1) for sz in (-1, 1)):
                point = center + rotation @ (half * np.asarray(signs))
                lo = np.minimum(lo, point)
                hi = np.maximum(hi, point)
            yield part, lo, hi

    def center_xy(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds_xy()
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def center_xyz(self) -> np.ndarray:
        return np.asarray([part.position for part in self.sim.machine.parts],
                          dtype=float).mean(axis=0)

    def rotation_matrix(self) -> np.ndarray:
        return euler_matrix_deg(self.orientation_deg_xyz)

    @property
    def mass_kg(self) -> float:
        return sum(max(0.0, float(part.mass_kg))
                   for part in self.sim.machine.parts)


@dataclass(frozen=True)
class WorldContact:
    """One resolved N5 contact between Machine identities or the floor."""

    a: str
    b: str
    normal: tuple[float, float, float]
    penetration_m: float
    normal_impulse_n_s: float
    tangent_impulse_n_s: float


class WoodshopWorldRules(DtCompatibleEngine):
    """Apply the existing honorary Newton laws to live world Machines.

    Geometry supplies contact candidates; it does not supply another physics
    model. N4.1 supplies gravity, N1.2/N1.1 advance momentum and position,
    and N5 supplies unilateral contact, impact restitution and friction.
    """

    HONORARY_LAWS = (
        "eq_N1_1", "eq_N1_2", "eq_N4_1",
        "eq_N5_1", "eq_N5_2", "eq_N5_3", "eq_N5_4", "eq_N5_5",
        "eq_N5_6", "eq_N5_7",
    )

    def __init__(self, items: dict[str, WorldMachine], *, object_graph=None,
                 restitution: float = 0.12, friction: float = 0.48):
        self.items = items
        self.object_graph = object_graph
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.contacts: list[WorldContact] = []
        self.world_time = 0.0
        self.observer_time = 0.0
        self.last_metrics = None
        self.newton_dt_graph = None
        self.newton_dt_state = None
        self._newton_dt_pieces = ()
        self._newton_batch = 0
        self._newton_dt_next = None
        self._laws = law_pieces(
            ("Newton",),
            law_ids=("eq_N5_5", "eq_N5_6", "eq_N5_7"),
        )
        constraints = {"eq_N5_1", "eq_N5_2", "eq_N5_3", "eq_N5_4"}
        manifested = {"eq_N1_1", "eq_N1_2", "eq_N4_1"}
        missing = tuple(name for name in self.HONORARY_LAWS
                        if name not in self._laws
                        and name not in constraints
                        and name not in manifested)
        if missing:
            raise RuntimeError(f"Newton honorary laws are not executable: {missing}")

    @staticmethod
    def _world_items(items):
        return [(identity, item) for identity, item in items.items()
                if item.custody == "world" and item.mass_kg > 0.0]

    @staticmethod
    def _momentum(item: WorldMachine) -> np.ndarray:
        return np.asarray(item.linear_momentum_kg_m_s, dtype=float)

    @staticmethod
    def _set_momentum(item: WorldMachine, momentum) -> None:
        item.linear_momentum_kg_m_s = tuple(float(value) for value in momentum)

    def _ensure_newton_dt_system(self) -> None:
        batch = len(self.items)
        if batch == self._newton_batch:
            return
        self._newton_dt_pieces = newton_world_dt_pieces(batch)
        self._newton_batch = batch
        self.newton_dt_state = None

    def _advance_newton_dt_system(self, dt: float):
        """Advance all object lanes through one real LLVM dt-system round."""
        self._ensure_newton_dt_system()
        identities = tuple(self.items)
        active = np.asarray([
            float(self.items[identity].custody == "world"
                  and self.items[identity].mass_kg > 0.0)
            for identity in identities
        ], dtype=np.float64)
        centers = np.asarray([
            self.items[identity].center_xyz() for identity in identities
        ], dtype=np.float64)
        momenta = np.asarray([
            self.items[identity].linear_momentum_kg_m_s
            for identity in identities
        ], dtype=np.float64)
        masses = np.asarray([
            self.items[identity].mass_kg for identity in identities
        ], dtype=np.float64)
        columns = {
            "active": active,
            "earth_mass": np.full(self._newton_batch, EARTH_MASS_KG),
            "earth_radius": np.full(self._newton_batch, EARTH_RADIUS_M),
            "force_x": np.zeros(self._newton_batch),
            "force_y": np.zeros(self._newton_batch),
            "force_z": np.zeros(self._newton_batch),
            "gravity_constant": np.full(
                self._newton_batch, GRAVITATIONAL_CONSTANT_M3_KG_S2),
            "mass": masses,
            "momentum_x": momenta[:, 0].copy(),
            "momentum_y": momenta[:, 1].copy(),
            "momentum_z": momenta[:, 2].copy(),
            "position_x": centers[:, 0].copy(),
            "position_y": centers[:, 1].copy(),
            "position_z": centers[:, 2].copy(),
        }
        requested = float(dt)
        initial = requested if self._newton_dt_next is None else min(
            requested, float(self._newton_dt_next))
        controller = STController(dt_min=requested * 1.0e-6)
        self.newton_dt_graph = RoundNode(
            plan=SuperstepPlan(round_max=requested, dt_init=initial),
            controller=ControllerNode(
                ctrl=controller,
                targets=Targets(cfl=0.5, div_max=1.0e9, mass_max=1.0e-3,
                                energy_exchange_fraction=0.2),
                dx=1.0,
            ),
            children=[
                piece_leaf(piece, label=label)
                for piece, label in zip(
                    self._newton_dt_pieces,
                    ("N4.1 gravity", "N1.2 momentum", "N1.1 position"),
                )
            ],
            schedule="sequential",
            label="woodshop-newton",
        )
        state, _controller, results = dt_system_from_graph(
            self.newton_dt_graph, columns, rounds=1)
        self.newton_dt_state = state
        advanced, self._newton_dt_next, _telemetry = results[0]
        if not math.isclose(float(advanced), requested, rel_tol=0.0,
                            abs_tol=1.0e-12 * max(1.0, requested)):
            raise RuntimeError(
                f"Newton dt system advanced {float(advanced)} of {requested}")

        for lane, identity in enumerate(identities):
            if not active[lane]:
                continue
            item = self.items[identity]
            self._set_momentum(item, (
                columns["momentum_x"][lane],
                columns["momentum_y"][lane],
                columns["momentum_z"][lane],
            ))
            translate_machine_3d(item.sim.machine, (
                float(columns["position_x"][lane]),
                float(columns["position_y"][lane]),
                float(columns["position_z"][lane]),
            ))
        return columns

    def lower_newton_dt_system(self, directory, *, backend="c",
                               optimization="O2", piece_mode="link"):
        """Lower the complete batched Newton dt loop as one native artifact."""
        self._ensure_newton_dt_system()
        build = Path(directory)
        piece_dir = build / "pieces"
        piece_dir.mkdir(parents=True, exist_ok=True)
        piece_files = []
        for index, piece in enumerate(self._newton_dt_pieces):
            path = piece_dir / f"{index:02d}-{piece.entry}.piece"
            piece.save(path)
            piece_files.append(path)
        return lowered_system(
            piece_files,
            backend=backend,
            directory=build,
            optimization=optimization,
            piece_mode=piece_mode,
        )

    def _normal_impulse(self, relative_normal_velocity: float,
                        mass_a: float, mass_b: float,
                        restitution: float) -> float:
        return _run_law(
            self._laws["eq_N5_7"],
            **{
                "I_a": 1.0, "I_b": 1.0,
                "e_r": restitution,
                "m_a": mass_a, "m_b": mass_b,
                "n": 1.0, "r_a": 0.0, "r_b": 0.0,
                "v_n_minus": relative_normal_velocity,
            },
        )

    def _friction_impulse(self, normal_impulse: float,
                          tangent_speed: float, cap: float) -> float:
        if tangent_speed <= 1.0e-12 or normal_impulse <= 0.0:
            return 0.0
        requested = abs(_run_law(
            self._laws["eq_N5_5"],
            **{"lambda_n": normal_impulse, "mu_f": self.friction,
               "v_t": tangent_speed},
        ))
        return min(requested, max(0.0, cap))

    def _translate(self, item: WorldMachine, delta) -> None:
        center = item.center_xyz() + np.asarray(delta, dtype=float)
        translate_machine_3d(item.sim.machine, tuple(center))

    def _resolve_floor(self, identity: str, item: WorldMachine) -> float:
        lo, _hi = item.bounds_xyz()
        penetration = max(0.0, -float(lo[2]))
        if penetration <= 0.0:
            return 0.0
        self._translate(item, (0.0, 0.0, penetration))
        mass = item.mass_kg
        momentum = self._momentum(item)
        velocity = momentum / mass
        normal_impulse = 0.0
        tangent_impulse = 0.0
        if velocity[2] < 0.0:
            restitution = 0.0 if abs(float(velocity[2])) < 0.08 else self.restitution
            velocity_after = _run_law(
                self._laws["eq_N5_6"],
                **{"e_r": restitution, "v_n_minus": float(velocity[2])},
            )
            normal_impulse = self._normal_impulse(
                float(velocity[2]), mass, 1.0e30, restitution,
            )
            momentum[2] = mass * velocity_after
            tangent = momentum[:2].copy()
            tangent_speed = float(np.linalg.norm(tangent / mass))
            tangent_impulse = self._friction_impulse(
                normal_impulse, tangent_speed, float(np.linalg.norm(tangent)),
            )
            if tangent_impulse > 0.0:
                momentum[:2] -= tangent / np.linalg.norm(tangent) * tangent_impulse
            self._set_momentum(item, momentum)
        self.contacts.append(WorldContact(
            identity, "world.floor", (0.0, 0.0, 1.0), penetration,
            normal_impulse, tangent_impulse,
        ))
        return penetration

    def _resolve_pair(self, identity_a: str, item_a: WorldMachine,
                      identity_b: str, item_b: WorldMachine) -> float:
        candidates = []
        for _part_a, lo_a, hi_a in item_a.part_bounds_xyz():
            for _part_b, lo_b, hi_b in item_b.part_bounds_xyz():
                overlap = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
                if np.any(overlap <= 0.0):
                    continue
                axis = int(np.argmin(overlap))
                candidates.append((float(overlap[axis]), axis,
                                   (lo_a + hi_a) * 0.5,
                                   (lo_b + hi_b) * 0.5))
        if not candidates:
            return 0.0
        penetration, axis, center_a, center_b = min(
            candidates, key=lambda candidate: candidate[0])
        normal = np.zeros(3, dtype=float)
        normal[axis] = 1.0 if center_b[axis] >= center_a[axis] else -1.0
        mass_a, mass_b = item_a.mass_kg, item_b.mass_kg
        inv_a, inv_b = 1.0 / mass_a, 1.0 / mass_b
        inv_sum = inv_a + inv_b
        self._translate(item_a, -normal * penetration * inv_a / inv_sum)
        self._translate(item_b, normal * penetration * inv_b / inv_sum)

        momentum_a = self._momentum(item_a)
        momentum_b = self._momentum(item_b)
        velocity_a = momentum_a / mass_a
        velocity_b = momentum_b / mass_b
        relative = velocity_b - velocity_a
        normal_speed = float(np.dot(relative, normal))
        normal_impulse = 0.0
        tangent_impulse = 0.0
        if normal_speed < 0.0:
            restitution = (0.0 if abs(normal_speed) < 0.08
                           else self.restitution)
            normal_impulse = self._normal_impulse(
                normal_speed, mass_a, mass_b, restitution,
            )
            impulse = normal * normal_impulse
            momentum_a -= impulse
            momentum_b += impulse
            tangent_velocity = relative - normal * normal_speed
            tangent_speed = float(np.linalg.norm(tangent_velocity))
            reduced_mass = 1.0 / inv_sum
            tangent_impulse = self._friction_impulse(
                normal_impulse, tangent_speed, reduced_mass * tangent_speed,
            )
            if tangent_impulse > 0.0:
                tangent_direction = tangent_velocity / tangent_speed
                tangent = tangent_direction * tangent_impulse
                momentum_a += tangent
                momentum_b -= tangent
            self._set_momentum(item_a, momentum_a)
            self._set_momentum(item_b, momentum_b)
        self.contacts.append(WorldContact(
            identity_a, identity_b, tuple(float(v) for v in normal),
            penetration, normal_impulse, tangent_impulse,
        ))
        return penetration

    def step(self, dt: float, state=None, state_table=None):
        self.contacts = []
        self._advance_newton_dt_system(dt)
        world_items = self._world_items(self.items)

        max_penetration = 0.0
        for identity, item in world_items:
            max_penetration = max(max_penetration,
                                  self._resolve_floor(identity, item))
        for index, (identity_a, item_a) in enumerate(world_items):
            for identity_b, item_b in world_items[index + 1:]:
                if (self.object_graph is not None
                        and any(edge.attributes.get("rigid", False)
                                for edge in self.object_graph.edges_between(
                                    identity_a, identity_b))):
                    continue
                max_penetration = max(
                    max_penetration,
                    self._resolve_pair(identity_a, item_a, identity_b, item_b),
                )

        max_velocity = max((
            float(np.linalg.norm(self._momentum(item) / item.mass_kg))
            for _identity, item in world_items
        ), default=0.0)
        if state_table is not None:
            state_table.set("woodshop", "world_rules", "honorary_laws",
                            self.HONORARY_LAWS)
            state_table.set("woodshop", "world_rules", "contacts",
                            copy.deepcopy(self.contacts))
            for identity, item in world_items:
                state_table.set("woodshop", "momentum", identity,
                                item.linear_momentum_kg_m_s)
                part_positions = {
                    part.identity: tuple(part.position)
                    for part in item.sim.machine.parts
                }
                for uuid_str, registered in state_table.identity_registry.items():
                    semantic = registered.get("semantic_identity")
                    if semantic in part_positions:
                        state_table.update_identity(
                            uuid_str, pos=part_positions[semantic])
        channels = empty_channels()
        metrics = Metrics(
            max_vel=max_velocity, max_flux=0.0,
            div_inf=max_penetration, mass_err=0.0,
            pub_exchange_time=AbstractTensor.tensor([0.0]),
            pub_exchange_time_present=AbstractTensor.tensor([0.0]),
            pub_contract=AbstractTensor.tensor([HOLD]),
            pub_dt_limit=AbstractTensor.tensor([0.0]),
            pub_dt_limit_present=AbstractTensor.tensor([0.0]),
            pub_values=channels.copy(),
            pub_present=AbstractTensor.zeros_like(channels),
            pub_limits=AbstractTensor.zeros_like(channels),
            pub_limits_present=AbstractTensor.zeros_like(channels),
            advanced_dt=float(dt),
        )
        self.last_metrics = metrics
        return True, metrics, self.get_state(state)

    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out["momentum"] = {
            identity: item.linear_momentum_kg_m_s
            for identity, item in self.items.items()
        }
        out["contacts"] = copy.deepcopy(self.contacts)
        return out

    def snapshot(self):
        return {
            "positions": {
                identity: [
                    (tuple(part.position),
                     None if part.base_position is None
                     else tuple(part.base_position),
                     tuple(tuple(map(float, port.position))
                           for port in part.ports))
                    for part in item.sim.machine.parts
                ]
                for identity, item in self.items.items()
            },
            "momentum": {
                identity: item.linear_momentum_kg_m_s
                for identity, item in self.items.items()
            },
            "contacts": copy.deepcopy(self.contacts),
            "world_time": self.world_time,
            "observer_time": self.observer_time,
            "newton_dt_next": self._newton_dt_next,
        }

    def restore(self, snapshot) -> None:
        for identity, transforms in snapshot["positions"].items():
            item = self.items.get(identity)
            if item is None:
                continue
            for part, (position, base_position, port_positions) in zip(
                    item.sim.machine.parts, transforms):
                part.position = tuple(position)
                part.base_position = (None if base_position is None
                                      else tuple(base_position))
                for port, port_position in zip(part.ports, port_positions):
                    port.position = np.asarray(port_position, dtype=float)
            item.sim.graph = item.sim.machine.build_graph()
        for identity, momentum in snapshot["momentum"].items():
            if identity in self.items:
                self.items[identity].linear_momentum_kg_m_s = tuple(momentum)
        self.contacts = copy.deepcopy(snapshot["contacts"])
        self.world_time = float(snapshot["world_time"])
        self.observer_time = float(snapshot["observer_time"])
        self._newton_dt_next = snapshot.get("newton_dt_next")


@dataclass
class DropPlacement:
    """Live, not-yet-released object placement controlled by the gimbal."""
    hand: str
    identity: str
    point_xy: tuple[float, float]
    orientation_deg_xyz: tuple[float, float, float]


@dataclass
class TwoHandHotbar:
    """Ten inventory slots with independent left- and right-hand views."""

    slots: dict[int, str] = field(default_factory=dict)
    left_slot: int | None = None
    right_slot: int | None = None

    def put(self, identity: str) -> int:
        existing = next((n for n, item in self.slots.items() if item == identity), None)
        if existing is not None:
            return existing
        slot = next((n for n in range(1, 11) if n not in self.slots), None)
        if slot is None:
            raise RuntimeError("hotbar is full")
        self.slots[slot] = identity
        return slot

    def equip(self, hand: str, slot: int) -> None:
        if hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        if slot not in self.slots:
            raise KeyError(slot)
        other = self.right_slot if hand == "left" else self.left_slot
        if other == slot:
            if hand == "left":
                self.right_slot = None
            else:
                self.left_slot = None
        if hand == "left":
            self.left_slot = slot
        else:
            self.right_slot = slot

    def held(self, hand: str) -> str | None:
        slot = self.left_slot if hand == "left" else self.right_slot
        return self.slots.get(slot) if slot is not None else None

    def remove(self, identity: str) -> None:
        slot = next((n for n, item in self.slots.items() if item == identity), None)
        if slot is None:
            return
        self.slots.pop(slot)
        if self.left_slot == slot:
            self.left_slot = None
        if self.right_slot == slot:
            self.right_slot = None

    def replace(self, predecessor: str, successor: str) -> int:
        """Transfer one occupied slot/hand binding across object fission."""
        slot = next((n for n, item in self.slots.items()
                     if item == predecessor), None)
        if slot is None:
            return self.put(successor)
        self.slots[slot] = successor
        return slot


def translate_machine(machine: Machine, center_xy: tuple[float, float]) -> tuple[float, float]:
    points = np.asarray([p.position[:2] for p in machine.parts], dtype=float)
    current = points.mean(axis=0)
    delta = np.asarray(center_xy, dtype=float) - current
    for part in machine.parts:
        position = np.asarray(part.position, dtype=float)
        position[:2] += delta
        part.position = tuple(float(v) for v in position)
        for port in part.ports:
            port_position = np.asarray(port.position, dtype=float)
            port_position[:2] += delta
            port.position = port_position
        if part.base_position is not None:
            base = np.asarray(part.base_position, dtype=float)
            base[:2] += delta
            part.base_position = tuple(float(v) for v in base)
    return float(delta[0]), float(delta[1])


def translate_machine_3d(machine: Machine,
                         center_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    points = np.asarray([p.position for p in machine.parts], dtype=float)
    delta = np.asarray(center_xyz, dtype=float) - points.mean(axis=0)
    for part in machine.parts:
        part.position = tuple(float(v) for v in
                              (np.asarray(part.position, dtype=float) + delta))
        for port in part.ports:
            port.position = np.asarray(port.position, dtype=float) + delta
        if part.base_position is not None:
            part.base_position = tuple(float(v) for v in
                                       (np.asarray(part.base_position, dtype=float) + delta))
    return tuple(float(v) for v in delta)


@dataclass
class WoodshopWorldSnapshot:
    """Versioned checkpoint of the world and every participating dt engine."""

    schema_version: int
    state_table: StateTable
    object_graph: WorldObjectGraph
    hotbar: TwoHandHotbar
    cutting: object
    world_rules: object
    machine_clocks: dict[str, tuple[float, float, object]]
    woodworking_joints: dict[str, WoodworkingJoint]
    compression_sleeve_joints: dict[str, CompressionSleeveJoint]
    elapsed_s: float
    action_counter: int
    retired_objects: dict[str, Disincorporation]
    separated_kerfs: set[str]
    chip_products: dict[str, str]
    severed_object_edges: dict[str, tuple]
    active_drop: DropPlacement | None
    hand_use_elapsed: dict[str, float]


class WoodshopSimulation:
    """Python gameplay state over real Machines and their dt engines."""

    def __init__(self) -> None:
        stocks = [
            kiln_dried_pine_2x4x8(f"stock.pine-stud.{number:03d}")
            for number in range(1, STARTING_STUD_COUNT + 1)
        ]
        # Keep one loose stud in the original work position and put the other
        # seven into a real, initially stable pile. Their dimensions are
        # unchanged; the first-person camera does not compensate or rescale.
        for index, stock in enumerate(stocks[1:]):
            stack = index // 4
            layer = index % 4
            translate_machine_3d(
                stock,
                (0.0,
                 1.45 + stack * (STUD_WIDTH_M + 0.012),
                 (layer + 0.5) * STUD_THICKNESS_M),
            )
        saw = hand_saw()
        bracket_positions = (
            (-0.75, -1.05), (0.75, -1.05),
            (-0.75, -1.42), (0.75, -1.42),
        )
        brackets = [
            resin_sawhorse_bracket(
                f"jig.sawhorse-bracket.{number:03d}",
                (*bracket_positions[number - 1], 0.145),
            )
            for number in range(1, STARTING_SAWHORSE_BRACKET_COUNT + 1)
        ]
        bracket_items = [
            WorldMachine(
                MachineSim(bracket),
                orientation_deg_xyz=(0.0, 0.0, 180.0 if index % 2 else 0.0),
            )
            for index, bracket in enumerate(brackets)
        ]
        clamps = [
            bar_clamp("tool.bar-clamp.001"),
            bar_clamp("tool.bar-clamp.002"),
            pipe_clamp("tool.pipe-clamp.001"),
            pipe_clamp("tool.pipe-clamp.002"),
        ]
        for machine, position in zip(
            clamps, ((-0.6, 0.75), (0.1, 0.75), (-0.5, 1.0), (0.7, 1.0))
        ):
            translate_machine(machine, position)
        self.object_graph = WorldObjectGraph()
        for stock in stocks:
            self.object_graph.add_node(stock.identity,
                                       WorldMachine(MachineSim(stock)))
        self.object_graph.add_node(saw.identity, WorldMachine(MachineSim(saw)))
        for item in bracket_items:
            self.object_graph.add_node(item.identity, item)
        for clamp in clamps:
            self.object_graph.add_node(clamp.identity,
                                       WorldMachine(MachineSim(clamp)))
        # Compatibility name for the interaction/render code. This is the
        # graph's live node table, not a second object collection.
        self.items = self.object_graph.nodes
        self.hotbar = TwoHandHotbar()
        self.state_table = StateTable()
        self.machine_systems: dict[str, MachineSystem] = {}
        shared_uuids: list[str] = []
        for identity, item in self.items.items():
            system = MachineSystem(item.sim)
            assembly = system.register(
                self.state_table,
                lambda part: {
                    "pos": tuple(part.position), "mass": float(part.mass_kg),
                    "semantic_identity": part.identity,
                    "machine_identity": identity,
                    "material_form": part.attributes.get("material_form", "solid"),
                },
                item.sim.machine.parts,
                group_label=identity,
            )
            shared_uuids.extend(assembly.uuids)
            self.machine_systems[identity] = system
        self.cutting = WoodCuttingSystem({
            identity: item.sim.machine for identity, item in self.items.items()
        })
        self.cutting.attach_assembly(IdentityAssembly(
            self.state_table, shared_uuids, "woodshop-machines"
        ))
        self.world_rules = WoodshopWorldRules(
            self.items, object_graph=self.object_graph)
        self.world_rules.attach_assembly(IdentityAssembly(
            self.state_table, shared_uuids, "woodshop-machines"
        ))
        self.woodworking_joints = WoodworkingJointInterface(self.object_graph)
        self.compression_sleeves = CompressionSleeveInterface(self.object_graph)
        self.elapsed_s = 0.0
        self.action_counter = 0
        self.retired_objects: dict[str, Disincorporation] = {}
        self.separated_kerfs: set[str] = set()
        self.chip_products: dict[str, str] = {}
        self.severed_object_edges: dict[str, tuple] = {}
        self.active_drop: DropPlacement | None = None
        self.hand_use_elapsed = {"left": 0.0, "right": 0.0}

    @staticmethod
    def default_save_path() -> Path:
        """Return the per-user save location used by the playable client."""
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "engine_toy" / "woodshop-world.pkl"

    def snapshot(self) -> WoodshopWorldSnapshot:
        """Checkpoint the live map through the engines' existing contracts.

        ``StateTable.snapshot`` remains the canonical dt-state operation.  The
        map graph is captured alongside it because custody, topology, meshes,
        ports, and Machine damage are real object state rather than table
        columns invented by this scene.
        """
        return WoodshopWorldSnapshot(
            schema_version=1,
            state_table=self.state_table.snapshot(),
            object_graph=copy.deepcopy(self.object_graph),
            hotbar=copy.deepcopy(self.hotbar),
            cutting=self.cutting.snapshot(),
            world_rules=self.world_rules.snapshot(),
            machine_clocks={
                identity: (
                    float(system.world_time),
                    float(system.observer_time),
                    copy.deepcopy(system.last_metrics),
                )
                for identity, system in self.machine_systems.items()
            },
            woodworking_joints=copy.deepcopy(self.woodworking_joints.joints),
            compression_sleeve_joints=copy.deepcopy(
                self.compression_sleeves.joints),
            elapsed_s=float(self.elapsed_s),
            action_counter=int(self.action_counter),
            retired_objects=copy.deepcopy(self.retired_objects),
            separated_kerfs=set(self.separated_kerfs),
            chip_products=dict(self.chip_products),
            severed_object_edges=copy.deepcopy(self.severed_object_edges),
            active_drop=copy.deepcopy(self.active_drop),
            hand_use_elapsed=dict(self.hand_use_elapsed),
        )

    def restore(self, snapshot: WoodshopWorldSnapshot) -> None:
        """Restore one complete checkpoint and rebind engines to its objects."""
        if not isinstance(snapshot, WoodshopWorldSnapshot):
            raise TypeError("not a woodshop world snapshot")
        if snapshot.schema_version != 1:
            raise ValueError(
                f"unsupported woodshop save version {snapshot.schema_version}")

        self.state_table.restore(snapshot.state_table)
        self.object_graph = copy.deepcopy(snapshot.object_graph)
        self.items = self.object_graph.nodes
        self.hotbar = copy.deepcopy(snapshot.hotbar)

        uuids_by_part: dict[str, list[str]] = {}
        for uuid_str, identity_state in self.state_table.identity_registry.items():
            semantic_identity = identity_state.get("semantic_identity")
            if semantic_identity is not None:
                uuids_by_part.setdefault(semantic_identity, []).append(uuid_str)

        self.machine_systems = {}
        live_uuids: list[str] = []
        for identity, item in self.items.items():
            system = MachineSystem(item.sim)
            machine_uuids = [
                uuid_str
                for part in item.sim.machine.parts
                for uuid_str in uuids_by_part.get(part.identity, ())
            ]
            system.attach_assembly(IdentityAssembly(
                self.state_table, machine_uuids, identity))
            world_time, observer_time, last_metrics = snapshot.machine_clocks.get(
                identity, (0.0, 0.0, None))
            system.world_time = float(world_time)
            system.observer_time = float(observer_time)
            system.last_metrics = copy.deepcopy(last_metrics)
            self.machine_systems[identity] = system
            live_uuids.extend(machine_uuids)

        shared = IdentityAssembly(
            self.state_table, live_uuids, "woodshop-machines")
        self.cutting = WoodCuttingSystem({
            identity: item.sim.machine for identity, item in self.items.items()
        })
        self.cutting.attach_assembly(shared)
        self.cutting.restore(snapshot.cutting)
        self.world_rules = WoodshopWorldRules(
            self.items, object_graph=self.object_graph)
        self.world_rules.attach_assembly(shared)
        self.world_rules.restore(snapshot.world_rules)

        self.woodworking_joints = WoodworkingJointInterface(self.object_graph)
        self.woodworking_joints.joints = copy.deepcopy(
            snapshot.woodworking_joints)
        self.compression_sleeves = CompressionSleeveInterface(self.object_graph)
        self.compression_sleeves.joints = copy.deepcopy(
            snapshot.compression_sleeve_joints)
        self.elapsed_s = float(snapshot.elapsed_s)
        self.action_counter = int(snapshot.action_counter)
        self.retired_objects = copy.deepcopy(snapshot.retired_objects)
        self.separated_kerfs = set(snapshot.separated_kerfs)
        self.chip_products = dict(snapshot.chip_products)
        self.severed_object_edges = copy.deepcopy(
            snapshot.severed_object_edges)
        self.active_drop = copy.deepcopy(snapshot.active_drop)
        self.hand_use_elapsed = dict(snapshot.hand_use_elapsed)

    def save_world(self, path: str | os.PathLike | None = None) -> Path:
        """Atomically persist a complete world checkpoint to disk."""
        destination = Path(path) if path is not None else self.default_save_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(self.snapshot(), stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return destination

    def load_world(self, path: str | os.PathLike | None = None) -> Path:
        """Load a world checkpoint made by :meth:`save_world`."""
        source = Path(path) if path is not None else self.default_save_path()
        with source.open("rb") as stream:
            snapshot = pickle.load(stream)
        self.restore(snapshot)
        return source

    def _publish_world_item(self, item: WorldMachine) -> None:
        """Attach one already-present object node to existing sim systems."""
        machine = item.sim.machine
        system = MachineSystem(item.sim)
        assembly = system.register(
            self.state_table,
            lambda part: {
                "pos": tuple(part.position), "mass": float(part.mass_kg),
                "semantic_identity": part.identity,
                "machine_identity": machine.identity,
                "material_form": part.attributes.get("material_form", "solid"),
            },
            machine.parts,
            group_label=machine.identity,
            dedup=False,
        )
        self.machine_systems[machine.identity] = system
        self.cutting.attach_assembly(assembly)
        self.cutting.machines[machine.identity] = machine
        self.world_rules.attach_assembly(assembly)

    def _publish_object_pose(self, item: WorldMachine) -> None:
        """Keep the existing unified state table current with map placement."""
        orientation = tuple(float(v) for v in item.orientation_deg_xyz)
        for part in item.sim.machine.parts:
            for uuid_str, state in self.state_table.identity_registry.items():
                if state.get("semantic_identity") == part.identity:
                    self.state_table.update_identity(uuid_str, pos=tuple(part.position))
        self.state_table.set("woodshop", "object_pose", item.identity, {
            "orientation_deg_xyz": orientation,
            "interaction_pose": item.pose_state,
        })

    def _register_world_machine(self, machine: Machine, *,
                                custody: str = "world",
                                slot: int | None = None) -> WorldMachine:
        item = WorldMachine(MachineSim(machine), custody=custody, slot=slot)
        self.object_graph.add_node(machine.identity, item)
        self._publish_world_item(item)
        return item

    def _retire_part_identities(self, transition: Disincorporation) -> None:
        retired = {record.identity: record for record in transition.retired_parts}
        for uuid_str, identity in self.state_table.identity_registry.items():
            record = retired.get(identity.get("semantic_identity"))
            if record is None:
                continue
            self.state_table.update_identity(uuid_str, mass=0.0)
            identity["retired"] = True
            identity["retirement_reason"] = record.reason
            identity["successors"] = record.successors

    def disincorporate_completed_kerf(self, kerf: Kerf) -> Disincorporation:
        """Turn one completed separation surface into new world objects."""
        if not kerf.complete:
            raise ValueError("a partial kerf has not separated the object")
        if kerf.identity in self.separated_kerfs:
            return self.retired_objects[kerf.target]
        predecessor_item = self.items[kerf.target]
        source = predecessor_item.sim.machine
        source_part = next(p for p in source.parts if p.part_role == "workpiece")
        body = self.cutting.material_bodies[source_part.identity]
        chip_identity = self.chip_products[kerf.identity]
        transition = disincorporate_sdf_body(
            source, body, transition_identity=kerf.identity,
            chips=self.items[chip_identity].sim.machine,
        )

        predecessor = kerf.target
        old_custody, old_slot = predecessor_item.custody, predecessor_item.slot
        self._retire_part_identities(transition)
        successor_items = {}
        for index, machine in enumerate(transition.solid_descendants):
            successor_items[machine.identity] = WorldMachine(
                MachineSim(machine),
                custody=old_custody if index == 0 else "world",
                slot=old_slot if index == 0 else None,
            )
        incident = self.object_graph.retire_and_add(
            transition.predecessor, successor_items,
        )
        self.severed_object_edges[predecessor] = incident
        self.machine_systems.pop(predecessor)
        self.cutting.machines.pop(predecessor, None)
        retained = transition.solid_descendants[0]
        if old_slot is not None:
            self.hotbar.replace(predecessor, retained.identity)
        for item in successor_items.values():
            self._publish_world_item(item)
        child_ids = tuple(machine.identity
                          for machine in transition.solid_descendants)
        if any(self.object_graph.edges_between(a, b)
               for index, a in enumerate(child_ids)
               for b in child_ids[index + 1:]):
            raise AssertionError("fission created an edge between new objects")
        self.retired_objects[predecessor] = transition
        self.separated_kerfs.add(kerf.identity)
        self.state_table.set(
            "woodshop", "topology", predecessor, transition,
        )
        return transition

    def _sync_chip_product(self, kerf: Kerf) -> None:
        if kerf.removed_mass_kg <= 0.0:
            return
        identity = self.chip_products.get(kerf.identity)
        if identity is None:
            source = self.items[kerf.target].sim.machine
            source_part = next(p for p in source.parts if p.part_role == "workpiece")
            identity = f"{kerf.target}.chips.{kerf.identity.replace('.', '-')}"
            midpoint = tuple((kerf.actual_a[i] + kerf.actual_b[i]) * 0.5
                             for i in range(3))
            chips = chip_batch(
                source, source_part, identity=identity, position=midpoint,
                mass_kg=kerf.removed_mass_kg,
                density_kg_m3=KILN_DRY_PINE_STUD.density_kg_m3,
                transition_identity=kerf.identity,
            )
            self._register_world_machine(chips)
            self.chip_products[kerf.identity] = identity
            return
        chips = self.items[identity].sim.machine
        update_chip_batch(chips, kerf.removed_mass_kg,
                          KILN_DRY_PINE_STUD.density_kg_m3)
        chip_part = chips.parts[0]
        for uuid_str, state in self.state_table.identity_registry.items():
            if state.get("semantic_identity") == chip_part.identity:
                self.state_table.update_identity(
                    uuid_str, pos=chip_part.position, mass=chip_part.mass_kg,
                )

    def auto_deploy_clamps(self, joint: WoodworkingJoint):
        """Deploy compatible live clamp Machines for a planned joint."""
        available = []
        for identity, item in self.items.items():
            try:
                clamp_spec(item.sim.machine)
            except ValueError:
                continue
            if not any(deployment.clamp == identity
                       for existing in self.woodworking_joints.joints.values()
                       for deployment in existing.deployments):
                available.append(identity)
        return self.woodworking_joints.auto_deploy(joint, available)

    def release_clamps(self, joint_identity: str) -> None:
        self.woodworking_joints.release(joint_identity)

    def begin_sleeve_fit_action(
            self, hand: str,
            cursor_xy: tuple[float, float]) -> CompressionSleeveJoint | None:
        """Insert held 2x4 stock into an other-hand or crosshair sleeve."""
        inserted_identity = self.hotbar.held(hand)
        if inserted_identity is None:
            return None
        inserted = self.items[inserted_identity].sim.machine
        if not any(part.part_role == "workpiece" for part in inserted.parts):
            return None
        other = "right" if hand == "left" else "left"
        sleeve_identity = self.hotbar.held(other)
        if sleeve_identity is None:
            sleeve_identity = self.object_at(cursor_xy, world_only=True)
        if sleeve_identity is None or sleeve_identity == inserted_identity:
            return None
        compatible = self.compression_sleeves.compatible_ports(
            sleeve_identity, inserted_identity)
        if not compatible:
            return None
        joint = self.compression_sleeves.initialize(
            sleeve_identity, inserted_identity,
            port_identity=compatible[0].identity,
        )
        sleeve_item = self.items[sleeve_identity]
        inserted_item = self.items[inserted_identity]
        self.hotbar.remove(inserted_identity)
        inserted_item.custody = sleeve_item.custody
        inserted_item.slot = None
        inserted_item.pose_state = "joined"
        inserted_item.linear_momentum_kg_m_s = (
            sleeve_item.linear_momentum_kg_m_s)
        self._publish_object_pose(inserted_item)
        self.state_table.set("woodshop", "compression_sleeve",
                             joint.identity, copy.deepcopy(joint))
        return joint

    def support_quality(self, identity: str) -> float:
        """Authored first calibration for progressively harder references."""
        clamp_edges = [edge for edge in self.object_graph.edges.values()
                       if identity in (edge.a, edge.b)
                       and edge.kind == "clamp-jaw-contact"]
        if clamp_edges:
            return 1.0
        sleeve_edges = [edge for edge in self.object_graph.edges.values()
                        if identity in (edge.a, edge.b)
                        and edge.kind == "compression-sleeve-fit"]
        if len(sleeve_edges) >= 2:
            return 0.90
        if len(sleeve_edges) == 1:
            return 0.70
        return 0.30

    def advance(self, dt: float) -> None:
        requested = float(dt)
        if not math.isfinite(requested) or requested <= 0.0:
            return
        for identity, system in self.machine_systems.items():
            sim = self.items[identity].sim
            ok, _metrics, _ = system.step_with_state(
                sim, requested, realtime=True, state_table=self.state_table
            )
            if not ok:
                raise RuntimeError(f"machine dt step rejected: {identity}")
        centers_before = {
            identity: item.center_xyz().copy()
            for identity, item in self.items.items()
        }
        ok, _metrics, _ = self.world_rules.step_with_state(
            self.world_rules.get_state(), requested, realtime=True,
            state_table=self.state_table,
        )
        if not ok:
            raise RuntimeError("world-rule dt step rejected")
        self.compression_sleeves.sync_geometry()
        for identity, item in self.items.items():
            delta = item.center_xyz() - centers_before.get(
                identity, item.center_xyz())
            if np.any(np.abs(delta) > 0.0):
                self.cutting.translate_target(identity, delta)
            item.sim.graph = item.sim.machine.build_graph()
            self._publish_object_pose(item)
        ok, _metrics, _ = self.cutting.step_with_state(
            self.cutting.state, requested, realtime=True,
            state_table=self.state_table,
        )
        if not ok:
            raise RuntimeError("cutting dt step rejected")
        for kerf in tuple(self.cutting.state.kerfs):
            self._sync_chip_product(kerf)
            if kerf.complete and kerf.identity not in self.separated_kerfs:
                self.disincorporate_completed_kerf(kerf)
        if self.cutting.state.held:
            for hand in ("left", "right"):
                identity = self.hotbar.held(hand)
                if identity is None:
                    continue
                machine = self.items[identity].sim.machine
                if not machine.actions:
                    continue
                self.hand_use_elapsed[hand] += requested
                stroke = int(self.hand_use_elapsed[hand] * 2.4) % 2
                self.items[identity].pose_state = "used-1" if stroke == 0 else "used-2"
        self.elapsed_s += requested

    def object_at(self, point_xy: tuple[float, float], *, world_only=False) -> str | None:
        candidates = []
        x, y = map(float, point_xy)
        for identity, item in self.items.items():
            if world_only and item.custody != "world":
                continue
            x0, y0, x1, y1 = item.bounds_xy()
            if x0 <= x <= x1 and y0 <= y <= y1:
                candidates.append(((x1 - x0) * (y1 - y0), identity))
        return min(candidates)[1] if candidates else None

    def pickup(self, identity: str, hand: str = "right") -> int:
        item = self.items[identity]
        slot = self.hotbar.put(identity)
        self.hotbar.equip(hand, slot)
        item.custody = "inventory"
        item.slot = slot
        item.pose_state = "selected"
        item.linear_momentum_kg_m_s = (0.0, 0.0, 0.0)
        return slot

    def try_put_in_hand(self, identity: str,
                        preferred_hand: str = "right") -> tuple[str, int] | None:
        """Default pickup rule used by any object without special handling."""
        if identity not in self.items or self.items[identity].custody != "world":
            return None
        other = "left" if preferred_hand == "right" else "right"
        hand = next((candidate for candidate in (preferred_hand, other)
                     if self.hotbar.held(candidate) is None), None)
        if hand is None:
            return None
        return hand, self.pickup(identity, hand)

    def _place_on_floor(self, item: WorldMachine,
                        point_xy: tuple[float, float]) -> tuple[float, float, float]:
        rotation = item.rotation_matrix()
        pivot = item.center_xyz()
        minimum_relative_z = math.inf
        for part in item.sim.machine.parts:
            relative_center = np.asarray(part.position, dtype=float) - pivot
            half = np.asarray(part.half_extent_m, dtype=float)
            for signs in ((sx, sy, sz) for sx in (-1, 1)
                          for sy in (-1, 1) for sz in (-1, 1)):
                relative = rotation @ (relative_center + half * np.asarray(signs))
                minimum_relative_z = min(minimum_relative_z, float(relative[2]))
        return float(point_xy[0]), float(point_xy[1]), -minimum_relative_z

    def begin_drop(self, hand: str, point_xy: tuple[float, float]) -> DropPlacement | None:
        identity = self.hotbar.held(hand)
        if identity is None:
            return None
        item = self.items[identity]
        item.pose_state = "drop-preview"
        self.active_drop = DropPlacement(
            hand, identity, tuple(map(float, point_xy)), item.orientation_deg_xyz)
        target = self._place_on_floor(item, self.active_drop.point_xy)
        delta = translate_machine_3d(item.sim.machine, target)
        self.cutting.translate_target(identity, delta)
        item.sim.graph = item.sim.machine.build_graph()
        self._publish_object_pose(item)
        return self.active_drop

    def adjust_drop(self, delta_deg_xyz: tuple[float, float, float]) -> DropPlacement:
        if self.active_drop is None:
            raise RuntimeError("no active drop placement")
        item = self.items[self.active_drop.identity]
        angles = tuple(float(a + d) for a, d in
                       zip(item.orientation_deg_xyz, delta_deg_xyz))
        item.orientation_deg_xyz = angles
        self.active_drop.orientation_deg_xyz = angles
        target = self._place_on_floor(item, self.active_drop.point_xy)
        delta = translate_machine_3d(item.sim.machine, target)
        self.cutting.translate_target(item.identity, delta)
        item.sim.graph = item.sim.machine.build_graph()
        self._publish_object_pose(item)
        return self.active_drop

    def cancel_drop(self) -> None:
        if self.active_drop is None:
            return
        item = self.items[self.active_drop.identity]
        item.pose_state = "selected"
        self.active_drop = None

    def drop(self, hand: str, point_xy: tuple[float, float],
             orientation_deg_xyz: tuple[float, float, float] | None = None) -> str | None:
        identity = self.hotbar.held(hand)
        if identity is None:
            return None
        self.hotbar.remove(identity)
        item = self.items[identity]
        item.custody = "world"
        item.slot = None
        item.pose_state = "placed"
        item.linear_momentum_kg_m_s = (0.0, 0.0, 0.0)
        if orientation_deg_xyz is not None:
            item.orientation_deg_xyz = tuple(map(float, orientation_deg_xyz))
        target = self._place_on_floor(item, point_xy)
        delta = translate_machine_3d(item.sim.machine, target)
        self.cutting.translate_target(identity, delta)
        item.sim.graph = item.sim.machine.build_graph()
        self._publish_object_pose(item)
        self.active_drop = None
        return identity

    def confirm_drop(self) -> str | None:
        if self.active_drop is None:
            return None
        placement = self.active_drop
        return self.drop(placement.hand, placement.point_xy,
                         placement.orientation_deg_xyz)

    def sync_hands(self, player_xy: tuple[float, float], aim_xy: tuple[float, float]) -> None:
        player = np.asarray(player_xy, dtype=float)
        aim = np.asarray(aim_xy, dtype=float) - player
        norm = float(np.linalg.norm(aim))
        forward = aim / norm if norm > 1e-9 else np.asarray((1.0, 0.0))
        side = np.asarray((-forward[1], forward[0]))
        for hand, sign in (("left", 1.0), ("right", -1.0)):
            identity = self.hotbar.held(hand)
            if identity is None:
                continue
            item = self.items[identity]
            if self.active_drop is not None and self.active_drop.identity == identity:
                continue
            pose = item.sim.machine.interaction_pose(item.pose_state)
            offset_right, offset_up, offset_forward = pose.offset_hand_m
            center_xy = (player + forward * offset_forward
                         + side * (sign * 0.16 + offset_right))
            center = (float(center_xy[0]), float(center_xy[1]),
                      EYE_HEIGHT_M + float(offset_up))
            heading_deg = math.degrees(math.atan2(forward[1], forward[0]))
            item.orientation_deg_xyz = (
                float(pose.rotation_deg_xyz[0]), float(pose.rotation_deg_xyz[1]),
                heading_deg + float(pose.rotation_deg_xyz[2]),
            )
            delta = translate_machine_3d(item.sim.machine, center)
            self.cutting.translate_target(identity, delta)
            item.sim.graph = item.sim.machine.build_graph()
            self._publish_object_pose(item)
        joined_before = {
            joint.inserted_object: self.items[joint.inserted_object].center_xyz().copy()
            for joint in self.compression_sleeves.joints.values()
            if not joint.released
        }
        self.compression_sleeves.sync_geometry()
        for identity, before in joined_before.items():
            item = self.items[identity]
            delta = item.center_xyz() - before
            if np.any(np.abs(delta) > 0.0):
                self.cutting.translate_target(identity, delta)
            self._publish_object_pose(item)

    def begin_saw_action(self, hand: str, cursor_xy: tuple[float, float]) -> Kerf | None:
        tool_identity = self.hotbar.held(hand)
        if tool_identity is None:
            return None
        tool = self.items[tool_identity].sim.machine
        if not any(action.gesture == "held-primary" and
                   action.operation == "advance-kerf" for action in tool.actions):
            return None
        other = "right" if hand == "left" else "left"
        target_identity = self.hotbar.held(other)
        support = 0.12 if target_identity is not None else None
        if target_identity is None:
            target_identity = self.object_at(cursor_xy, world_only=True)
        if target_identity is None:
            return None
        if support is None:
            support = self.support_quality(target_identity)
        target = self.items[target_identity].sim.machine
        if not any(part.part_role == "workpiece" for part in target.parts):
            return None
        self.action_counter += 1
        projection = project_crosscut(
            target, cursor_xy, support_quality=support,
            attempt_identity=f"attempt-{self.action_counter}",
        )
        kerf = self.cutting.begin(tool_identity, projection)
        self.cutting.set_held(True)
        self.items[tool_identity].pose_state = "used-1"
        self.hand_use_elapsed[hand] = 0.0
        return kerf

    def release_saw_action(self) -> None:
        self.cutting.set_held(False)
        for hand in ("left", "right"):
            identity = self.hotbar.held(hand)
            if identity is not None and self.items[identity].sim.machine.actions:
                self.items[identity].pose_state = "selected"


__all__ = [
    "BoxMesh", "CutProjection", "CuttingState", "Kerf",
    "KILN_DRY_PINE_STUD", "NOMINAL_STUD_LENGTH_M", "OrthotropicWood",
    "STUD_THICKNESS_M", "STUD_WIDTH_M", "ToothPattern", "TwoHandHotbar",
    "WoodCuttingSystem", "WoodshopSimulation", "WoodshopWorldRules",
    "WorldContact", "WorldMachine", "hand_saw",
    "WoodworkingJoint", "WoodworkingJointInterface",
    "kiln_dried_pine_2x4x8", "part_box_mesh", "project_crosscut",
    "translate_machine", "translate_machine_3d",
]
