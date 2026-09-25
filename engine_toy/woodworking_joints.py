"""Woodworking joint planning with real auto-deployed clamp Machines.

A clamp never turns two workpieces into one object and never creates a direct
workpiece-to-workpiece edge.  Its two jaw contacts are explicit world-object
edges, so the temporary force path is ``member A -> clamp -> member B``.
Releasing the clamp removes those edges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from honorary_engine_equation_catalogue import law_pieces
from assembly_ports import (
    CompressionSleeveSpec,
    FastenerArea,
    GlueSurface,
    PartPort,
)
from machines import Machine, MachineAction, MachineLine, MachinePart, MachinePose
from material_topology import ObjectEdge, WorldObjectGraph


@dataclass(frozen=True)
class ClampSpec:
    style: str
    opening_capacity_m: float
    throat_depth_m: float
    rated_force_n: float
    spindle_diameter_m: float
    thread_nut_factor: float
    handle_torque_nm: float
    pad_area_m2: float
    pad_friction: float
    frame_stiffness_n_m: float


@dataclass
class ClampDeployment:
    clamp: str
    contact_a: tuple[float, float, float]
    contact_b: tuple[float, float, float]
    force_n: float
    pad_pressure_pa: float
    slip_capacity_n: float
    frame_deflection_m: float
    edge_ids: tuple[str, str]


@dataclass
class WoodworkingJoint:
    identity: str
    member_a: str
    member_b: str
    seam_start: tuple[float, float, float]
    seam_end: tuple[float, float, float]
    clamp_normal: tuple[float, float, float]
    required_opening_m: float
    required_throat_m: float
    target_total_force_n: float
    max_pad_pressure_pa: float
    max_spacing_m: float = 0.30
    deployments: list[ClampDeployment] = field(default_factory=list)

    @property
    def deployed(self) -> bool:
        return bool(self.deployments)


@dataclass
class CompressionSleeveJoint:
    """Live state of one profile inserted into one friction-fit sleeve."""

    identity: str
    sleeve_object: str
    inserted_object: str
    port_identity: str
    inserted_part: str
    penetration_m: float
    max_penetration_m: float
    holding_force_n: float
    release_vector: tuple[float, float, float]
    edge_identity: str
    released: bool = False


def _sleeve_port(identity: str, part_identity: str, position, direction,
                 up_reference, role: str) -> PartPort:
    spec = CompressionSleeveSpec(
        accepted_profile_m=(0.0381, 0.0889),
        profile_tolerance_m=0.0025,
        capture_distance_m=0.045,
        max_penetration_m=0.105,
        compression_force_n=850.0,
        friction_coefficient=0.46,
        up_reference=tuple(up_reference),
        lateral_capacity_n=2_400.0,
        moment_capacity_nm=180.0,
    )
    fastener_areas = (
        FastenerArea(
            f"{identity}.fastener-a", (0.0, 0.030, -0.020),
            (0.0, 1.0, 0.0), (0.038, 0.002, 0.045),
            substrate_thickness_m=0.006, min_edge_distance_m=0.012,
        ),
        FastenerArea(
            f"{identity}.fastener-b", (0.0, -0.030, -0.020),
            (0.0, -1.0, 0.0), (0.038, 0.002, 0.045),
            substrate_thickness_m=0.006, min_edge_distance_m=0.012,
        ),
    )
    glue_surfaces = (
        GlueSurface(
            f"{identity}.glue-bed", (0.0, 0.0, -0.050),
            tuple(-np.asarray(up_reference, dtype=float)),
            (0.045, 0.018, 0.052), "glass-filled-structural-resin",
        ),
    )
    return PartPort(
        identity, part_identity, "friction-fit-sleeve",
        np.asarray(position, dtype=float), np.asarray(direction, dtype=float),
        radius_m=0.050, mating=False, fluid="none", closure="",
        joint="compression-sleeve-fit", compression_sleeve=spec,
        fastener_areas=fastener_areas, glue_surfaces=glue_surfaces, role=role,
    )


def resin_sawhorse_bracket(
        identity: str = "jig.sawhorse-bracket.001",
        position=(0.0, -0.95, 0.145)) -> Machine:
    """One moulded jig with two leg sleeves and one top-beam sleeve."""

    x, y, z = map(float, position)
    part_identity = f"{identity}.moulding"
    part = MachinePart(
        part_identity, "one-piece-sawhorse-end-bracket", (x, y, z),
        (0.120, 0.135, 0.145), mass_kg=0.82,
        material="glass-filled-structural-resin", part_role="assembly-jig",
        attributes={
            "parametric_shape": "three-sleeve-sawhorse-end-bracket",
            "one_physical_part": True,
            "wall_thickness_m": 0.006,
            "leg_splay_deg": 18.0,
            "accepted_stock": "dressed-nominal-2x4-any-length",
        },
    )
    # Port direction is the outward/pull-release direction.  The top beam
    # leaves along +X; the legs leave downward and laterally from the jig.
    leg_z = -math.cos(math.radians(18.0))
    leg_y = math.sin(math.radians(18.0))
    part.ports = [
        _sleeve_port(
            f"{identity}.port.top-beam", part_identity,
            (x + 0.120, y, z + 0.055), (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0), "top-beam",
        ),
        _sleeve_port(
            f"{identity}.port.leg-left", part_identity,
            (x, y - 0.075, z - 0.090), (0.0, -leg_y, leg_z),
            (1.0, 0.0, 0.0), "leg-left",
        ),
        _sleeve_port(
            f"{identity}.port.leg-right", part_identity,
            (x, y + 0.075, z - 0.090), (0.0, leg_y, leg_z),
            (1.0, 0.0, 0.0), "leg-right",
        ),
    ]
    return Machine(
        identity, "one-piece resin sawhorse bracket", "human-held",
        "mechanical", parts=[part],
        actions=[MachineAction(
            f"{identity}.actions.insert", "target-primary",
            "initialize-compression-sleeve-fit", part_identity,
        )],
        interaction_poses=[
            MachinePose("selected", (0.0, -0.30, 0.52), (0.0, 0.0, 0.0)),
        ],
        note=("reusable one-part jig; its three friction-fit sleeves align "
              "ordinary dressed 2x4 stock without requiring a fastener"),
    )


def _rotation_matrix_deg(rotation_deg_xyz) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(rotation_deg_xyz, dtype=float))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.asarray(((cz * cy, cz * sy * sx - sz * cx,
                        cz * sy * cx + sz * sx),
                       (sz * cy, sz * sy * sx + cz * cx,
                        sz * sy * cx - cz * sx),
                       (-sy, cy * sx, cy * cx)), dtype=float)


def _euler_deg(matrix: np.ndarray) -> tuple[float, float, float]:
    sy = min(1.0, max(-1.0, -float(matrix[2, 0])))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1.0e-8:
        rx = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        rz = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        rx = math.atan2(-float(matrix[1, 2]), float(matrix[1, 1]))
        rz = 0.0
    return tuple(map(math.degrees, (rx, ry, rz)))


def _part_profile(part: MachinePart) -> tuple[float, float]:
    dimensions = [2.0 * float(value) for value in part.half_extent_m]
    dimensions.pop(int(np.argmax(dimensions)))
    return tuple(dimensions)


class CompressionSleeveInterface:
    """Initialize and release rigid-until-yield sleeve edges in the world graph."""

    def __init__(self, objects: WorldObjectGraph):
        self.objects = objects
        self.joints: dict[str, CompressionSleeveJoint] = {}
        self._laws = law_pieces(
            ("Woodshop",), law_ids=("eq_WO5_3",))

    def sleeve_ports(self, object_identity: str) -> tuple[PartPort, ...]:
        machine = _machine(self.objects.nodes[object_identity])
        return tuple(port for part in machine.parts for port in part.ports
                     if getattr(port, "compression_sleeve", None) is not None)

    def compatible_ports(self, sleeve_object: str,
                         inserted_object: str) -> tuple[PartPort, ...]:
        inserted = _machine(self.objects.nodes[inserted_object])
        workpieces = [part for part in inserted.parts
                      if part.part_role == "workpiece"]
        if len(workpieces) != 1:
            return ()
        profile = _part_profile(workpieces[0])
        return tuple(port for port in self.sleeve_ports(sleeve_object)
                     if port.connected_to is None
                     and port.compression_sleeve.accepts(profile))

    @staticmethod
    def _port_frame(payload, port: PartPort):
        item = payload if hasattr(payload, "sim") else None
        machine = _machine(payload)
        pivot = np.asarray([part.position for part in machine.parts],
                           dtype=float).mean(axis=0)
        rotation = (_rotation_matrix_deg(item.orientation_deg_xyz)
                    if item is not None else np.eye(3))
        position = pivot + rotation @ (np.asarray(port.position) - pivot)
        release = rotation @ np.asarray(port.direction, dtype=float)
        release /= max(float(np.linalg.norm(release)), 1.0e-12)
        up = rotation @ np.asarray(port.compression_sleeve.up_reference,
                                   dtype=float)
        up -= release * float(np.dot(up, release))
        up /= max(float(np.linalg.norm(up)), 1.0e-12)
        side = np.cross(up, release)
        side /= max(float(np.linalg.norm(side)), 1.0e-12)
        up = np.cross(release, side)
        return position, release, up, side

    def initialize(self, sleeve_object: str, inserted_object: str, *,
                   port_identity: str | None = None,
                   penetration_m: float | None = None) -> CompressionSleeveJoint:
        ports = self.compatible_ports(sleeve_object, inserted_object)
        if port_identity is not None:
            ports = tuple(port for port in ports
                          if port.identity == port_identity)
        if not ports:
            raise ValueError("no unoccupied compression sleeve accepts this stock profile")
        port = ports[0]
        spec = port.compression_sleeve
        penetration = (spec.max_penetration_m if penetration_m is None
                       else min(max(0.0, float(penetration_m)),
                                spec.max_penetration_m))
        inserted_payload = self.objects.nodes[inserted_object]
        inserted_item = (inserted_payload if hasattr(inserted_payload, "sim")
                         else None)
        inserted = _machine(inserted_payload)
        part = next(part for part in inserted.parts
                    if part.part_role == "workpiece")
        position, release, up, side = self._port_frame(
            self.objects.nodes[sleeve_object], port)
        # Local +X runs out of the sleeve, local +Z follows its declared up.
        frame = np.column_stack((release, side, up))
        if inserted_item is not None:
            inserted_item.orientation_deg_xyz = _euler_deg(frame)
        half_length = float(part.half_extent_m[0])
        centre = position + release * (half_length - penetration)
        _translate(inserted, centre)
        if inserted_item is not None:
            inserted_item.sim.graph = inserted.build_graph()
        full_holding = _law(
            self._laws["eq_WO5_3"],
            mu_pad=spec.friction_coefficient,
            F_clamp=spec.compression_force_n,
        )
        holding = full_holding * penetration / spec.max_penetration_m
        identity = f"fit.{port.identity}.{inserted_object}"
        edge_identity = f"{identity}.edge"
        edge = ObjectEdge(
            edge_identity, sleeve_object, inserted_object,
            "compression-sleeve-fit", {
                "joint_type": "compression-sleeve-fit",
                "port": port.identity,
                "inserted_part": part.identity,
                "penetration_m": penetration,
                "max_penetration_m": spec.max_penetration_m,
                "holding_force_n": holding,
                "release_vector": tuple(float(v) for v in release),
                "lateral_capacity_n": spec.lateral_capacity_n,
                "moment_capacity_nm": spec.moment_capacity_nm,
                "rigid": True,
                "condense_rigid_membership": True,
                "fastener_areas": [area.to_data()
                                   for area in port.fastener_areas],
                "glue_surfaces": [surface.to_data()
                                  for surface in port.glue_surfaces],
                "honorary_law": "eq_WO5_3",
            },
        )
        self.objects.add_edge(edge)
        port.connected_to = part.identity
        joint = CompressionSleeveJoint(
            identity, sleeve_object, inserted_object, port.identity,
            part.identity, penetration, spec.max_penetration_m, holding,
            tuple(float(v) for v in release), edge_identity,
        )
        self.joints[identity] = joint
        return joint

    def apply_force(self, joint_identity: str, force_xyz) -> bool:
        """Return True when force along the release vector pulls the fit free."""
        joint = self.joints[joint_identity]
        if joint.released:
            return True
        release_force = max(0.0, float(np.dot(
            np.asarray(force_xyz, dtype=float),
            np.asarray(joint.release_vector, dtype=float),
        )))
        if release_force <= joint.holding_force_n:
            return False
        self.objects.remove_edge(joint.edge_identity)
        port = next(port for port in self.sleeve_ports(joint.sleeve_object)
                    if port.identity == joint.port_identity)
        port.connected_to = None
        joint.penetration_m = 0.0
        joint.holding_force_n = 0.0
        joint.released = True
        return True

    def sync_geometry(self) -> None:
        """Keep every active inserted profile on its sleeve's rigid frame."""
        for joint in self.joints.values():
            if joint.released:
                continue
            port = next(port for port in self.sleeve_ports(joint.sleeve_object)
                        if port.identity == joint.port_identity)
            inserted_payload = self.objects.nodes[joint.inserted_object]
            sleeve_payload = self.objects.nodes[joint.sleeve_object]
            inserted_item = (inserted_payload
                             if hasattr(inserted_payload, "sim") else None)
            sleeve_item = (sleeve_payload
                           if hasattr(sleeve_payload, "sim") else None)
            inserted = _machine(inserted_payload)
            part = next(part for part in inserted.parts
                        if part.identity == joint.inserted_part)
            position, release, up, side = self._port_frame(
                self.objects.nodes[joint.sleeve_object], port)
            if inserted_item is not None:
                inserted_item.orientation_deg_xyz = _euler_deg(
                    np.column_stack((release, side, up)))
            centre = position + release * (
                float(part.half_extent_m[0]) - joint.penetration_m)
            _translate(inserted, centre)
            if inserted_item is not None:
                if (sleeve_item is not None
                        and hasattr(inserted_item, "linear_momentum_kg_m_s")
                        and hasattr(sleeve_item, "linear_momentum_kg_m_s")):
                    sleeve_mass = sum(float(p.mass_kg)
                                      for p in _machine(sleeve_payload).parts)
                    inserted_mass = sum(float(p.mass_kg)
                                        for p in inserted.parts)
                    sleeve_velocity = (
                        np.asarray(sleeve_item.linear_momentum_kg_m_s,
                                   dtype=float)
                        / max(sleeve_mass, 1.0e-12)
                    )
                    inserted_item.linear_momentum_kg_m_s = tuple(
                        sleeve_velocity * inserted_mass)
                inserted_item.sim.graph = inserted.build_graph()


def _clamp_machine(identity: str, label: str, spec: ClampSpec,
                   position=(0.0, 0.75, 0.04)) -> Machine:
    x, y, z = map(float, position)
    rail_length = spec.opening_capacity_m + 0.20
    rail_kind = "steel-pipe" if spec.style == "pipe" else "steel-bar"
    rail = MachinePart(
        f"{identity}.rail", rail_kind, (x, y, z),
        (rail_length / 2.0, 0.009, 0.009),
        mass_kg=2.4 if spec.style == "pipe" else 1.6,
        material="medium-carbon-steel", part_role="clamp-frame",
        attributes={"clamp_spec": spec},
    )
    fixed_x = x - rail_length / 2.0 + 0.04
    moving_x = fixed_x + min(0.18, spec.opening_capacity_m)
    parts = [
        rail,
        MachinePart(f"{identity}.fixed-jaw", "clamp-fixed-jaw",
                    (fixed_x, y, z + spec.throat_depth_m / 2.0),
                    (0.018, 0.018, spec.throat_depth_m / 2.0), 0.42,
                    "ductile-iron", part_role="fixed-jaw"),
        MachinePart(f"{identity}.moving-jaw", "clamp-moving-jaw",
                    (moving_x, y, z + spec.throat_depth_m / 2.0),
                    (0.021, 0.018, spec.throat_depth_m / 2.0), 0.55,
                    "ductile-iron", part_role="moving-jaw"),
        MachinePart(f"{identity}.spindle", "clamp-screw-spindle",
                    (moving_x + 0.045, y, z + spec.throat_depth_m),
                    (0.065, spec.spindle_diameter_m / 2.0,
                     spec.spindle_diameter_m / 2.0), 0.32,
                    "medium-carbon-steel", part_role="spindle"),
        MachinePart(f"{identity}.pad", "swivel-pressure-pad",
                    (moving_x - 0.004, y, z + spec.throat_depth_m),
                    (0.004, math.sqrt(spec.pad_area_m2) / 2.0,
                     math.sqrt(spec.pad_area_m2) / 2.0), 0.08,
                    "steel-rubber-pad", part_role="pressure-pad"),
        MachinePart(f"{identity}.handle", "sliding-tommy-bar",
                    (moving_x + 0.11, y, z + spec.throat_depth_m),
                    (0.09, 0.004, 0.004), 0.12,
                    "steel", part_role="handle"),
    ]
    lines = [
        MachineLine(f"{identity}.fixed-to-rail", rail.identity, parts[1].identity,
                    "solid-welded", 0.008, "mechanical", "steel"),
        MachineLine(f"{identity}.slider-to-rail", rail.identity, parts[2].identity,
                    "single-axis-slider", 0.008, "mechanical", "steel"),
        MachineLine(f"{identity}.spindle-to-slider", parts[2].identity,
                    parts[3].identity, "screw-pair", 0.006,
                    "mechanical", "steel"),
        MachineLine(f"{identity}.pad-to-spindle", parts[3].identity,
                    parts[4].identity, "spherical-seat", 0.006,
                    "mechanical", "steel"),
    ]
    return Machine(
        identity, label, "human-held", "mechanical", parts=parts, lines=lines,
        actions=[MachineAction(
            f"{identity}.actions.deploy", "auto-deploy",
            "apply-woodworking-clamp", parts[4].identity,
            {"release_operation": "release-woodworking-clamp"},
        )],
        note="temporary force-carrying machine; jaw contacts are explicit edges",
    )


def bar_clamp(identity: str = "tool.bar-clamp.001") -> Machine:
    return _clamp_machine(identity, "600 mm F-style bar clamp", ClampSpec(
        "bar", 0.60, 0.085, 5_000.0, 0.014, 0.20, 12.0,
        math.pi * 0.018**2, 0.55, 2.2e6,
    ))


def pipe_clamp(identity: str = "tool.pipe-clamp.001") -> Machine:
    return _clamp_machine(identity, "1200 mm 3/4-inch pipe clamp", ClampSpec(
        "pipe", 1.20, 0.060, 7_000.0, 0.016, 0.20, 16.0,
        math.pi * 0.020**2, 0.50, 2.8e6,
    ), position=(0.0, 0.95, 0.04))


def clamp_spec(machine: Machine) -> ClampSpec:
    for part in machine.parts:
        spec = part.attributes.get("clamp_spec")
        if spec is not None:
            return spec
    raise ValueError(f"{machine.identity} is not a clamp machine")


def _machine(payload) -> Machine:
    return payload.sim.machine if hasattr(payload, "sim") else payload


def _translate(machine: Machine, centre) -> None:
    points = np.asarray([part.position for part in machine.parts], dtype=float)
    delta = np.asarray(centre, dtype=float) - points.mean(axis=0)
    for part in machine.parts:
        part.position = tuple(np.asarray(part.position, dtype=float) + delta)
        for port in part.ports:
            port.position = np.asarray(port.position, dtype=float) + delta
        if part.base_position is not None:
            part.base_position = tuple(np.asarray(part.base_position, dtype=float) + delta)


class WoodworkingJointInterface:
    """Plan and deploy temporary clamp load paths on the world object graph."""

    def __init__(self, objects: WorldObjectGraph):
        self.objects = objects
        self.joints: dict[str, WoodworkingJoint] = {}
        self._laws = law_pieces(
            ("Woodshop",),
            law_ids=tuple(f"eq_WO5_{number}" for number in range(1, 5)),
        )

    def auto_deploy(self, joint: WoodworkingJoint,
                    available_clamps: list[str]) -> list[ClampDeployment]:
        if joint.deployed:
            raise ValueError(f"joint already clamped: {joint.identity}")
        if joint.member_a == joint.member_b:
            raise ValueError("a clamp joint requires two object nodes")
        for identity in (joint.member_a, joint.member_b):
            if identity not in self.objects.nodes:
                raise KeyError(identity)
        seam = np.asarray(joint.seam_end) - np.asarray(joint.seam_start)
        seam_length = float(np.linalg.norm(seam))
        spacing_count = max(1, int(math.ceil(seam_length / joint.max_spacing_m)))
        candidates = []
        for identity in available_clamps:
            machine = _machine(self.objects.nodes[identity])
            spec = clamp_spec(machine)
            if (spec.opening_capacity_m >= joint.required_opening_m
                    and spec.throat_depth_m >= joint.required_throat_m):
                candidates.append((identity, machine, spec))
        if not candidates:
            raise RuntimeError("no available clamp fits the opening and throat")
        candidates.sort(key=lambda item: item[2].rated_force_n, reverse=True)
        force_count = max(1, int(math.ceil(
            joint.target_total_force_n / candidates[0][2].rated_force_n)))
        count = max(spacing_count, force_count)
        if len(candidates) < count:
            raise RuntimeError(f"joint needs {count} compatible clamps; {len(candidates)} available")

        selected = candidates[:count]
        force_each = joint.target_total_force_n / count
        normal = np.asarray(joint.clamp_normal, dtype=float)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        start = np.asarray(joint.seam_start, dtype=float)
        deployments = []
        for index, (identity, machine, spec) in enumerate(selected):
            fraction = (index + 0.5) / count
            centre = start + seam * fraction
            theoretical = _law(self._laws["eq_WO5_1"],
                               T_handle=spec.handle_torque_nm,
                               K_thread=spec.thread_nut_factor,
                               d_spindle=spec.spindle_diameter_m)
            pressure_limit = joint.max_pad_pressure_pa * spec.pad_area_m2
            force = min(force_each, theoretical, spec.rated_force_n, pressure_limit)
            pressure = _law(self._laws["eq_WO5_2"],
                            F_clamp=force, A_pad=spec.pad_area_m2)
            slip = _law(self._laws["eq_WO5_3"],
                        mu_pad=spec.pad_friction, F_clamp=force)
            deflection = _law(self._laws["eq_WO5_4"],
                              F_clamp=force, k_frame=spec.frame_stiffness_n_m)
            a = tuple(centre - normal * joint.required_opening_m / 2.0)
            b = tuple(centre + normal * joint.required_opening_m / 2.0)
            _translate(machine, centre)
            common = {
                "joint": joint.identity, "temporary": True,
                "normal": tuple(float(v) for v in normal),
                "clamp_force_n": force, "pad_pressure_pa": pressure,
                "slip_capacity_n": slip, "frame_deflection_m": deflection,
            }
            edge_a = ObjectEdge(f"{joint.identity}.{identity}.jaw-a",
                                identity, joint.member_a, "clamp-jaw-contact",
                                {**common, "contact": a, "jaw": "fixed"})
            edge_b = ObjectEdge(f"{joint.identity}.{identity}.jaw-b",
                                identity, joint.member_b, "clamp-jaw-contact",
                                {**common, "contact": b, "jaw": "moving"})
            self.objects.add_edge(edge_a)
            self.objects.add_edge(edge_b)
            deployments.append(ClampDeployment(
                identity, a, b, force, pressure, slip, deflection,
                (edge_a.identity, edge_b.identity),
            ))
        joint.deployments = deployments
        self.joints[joint.identity] = joint
        return deployments

    def release(self, joint_identity: str) -> None:
        joint = self.joints[joint_identity]
        for deployment in joint.deployments:
            for edge in deployment.edge_ids:
                self.objects.remove_edge(edge)
        joint.deployments = []


def _law(piece, **values) -> float:
    columns = tuple(
        np.asarray([values[name]], dtype=np.float64)
        for name in piece.argument_names
    )
    return float(np.asarray(piece(*columns)[0], dtype=np.float64).reshape(-1)[0])


__all__ = [
    "ClampDeployment", "ClampSpec", "CompressionSleeveInterface",
    "CompressionSleeveJoint", "WoodworkingJoint",
    "WoodworkingJointInterface", "bar_clamp", "clamp_spec", "pipe_clamp",
    "resin_sawhorse_bracket",
]
