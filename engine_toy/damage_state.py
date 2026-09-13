"""Persistent, geometry-derived damage state for engine parts."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from ballistics import ImpactResult


ATMOSPHERIC_PRESSURE_PA = 101_325.0


@dataclass(frozen=True)
class Puncture:
    material: str
    entry_position_m: tuple[float, float, float]
    exit_position_m: tuple[float, float, float] | None
    direction: tuple[float, float, float]
    radius_m: float
    through: bool
    resisting_thickness_m: float
    penetration_depth_m: float
    energy_before_j: float
    energy_spent_j: float
    damage_mode: str
    incidence_angle_deg: float
    speed_before_m_s: float
    speed_after_m_s: float
    projectile_integrity: float
    fluid_energy_spent_j: float

    @property
    def boundary_holes(self) -> int:
        return 2 if self.through else 1


@dataclass
class PartDamageState:
    identity: str
    kind: str = "mesh-part"
    fluid_volume_l: float = 0.0
    pressure_pa: float = ATMOSPHERIC_PRESSURE_PA
    circuit_identity: str | None = None
    punctures: list[Puncture] = field(default_factory=list)
    impacts: list[ImpactResult] = field(default_factory=list)


def register_graph_parts(states: dict[str, PartDamageState], graph: dict) -> None:
    """Ensure every graph node and edge has a persistent damage record."""
    for node in graph.get("nodes", ()):
        identity = str(node["identity"])
        state = states.setdefault(
            identity,
            PartDamageState(identity=identity, kind=str(node.get("kind", "node"))),
        )
        state.fluid_volume_l = float(node.get("fluid_volume_l", state.fluid_volume_l))
    for edge in graph.get("edges", ()):
        identity = str(edge["identity"])
        states.setdefault(
            identity,
            PartDamageState(identity=identity, kind=str(edge.get("constraint", "edge"))),
        )


# The castings are drawn as many named pieces (a bore, a water jacket, a
# sump, a head casting, a crankcase bulkhead) that are not graph nodes of
# their own -- but every one of them is PART OF something that is. This
# table is that ownership, and it is the single place it is declared, so
# a hole in "cyl3_water_jacket" is a hole in cylinder 3 everywhere: in
# its circuit membership (which decides what leaks out of it), in its
# node condition, in what the dashboard says, and in what a burst takes
# with it.
_MESH_PART_OWNERS: list = [
    (re.compile(r"^cyl(\d+)_"), lambda m: f"powertrain.cylinder_{m.group(1)}"),
    (re.compile(r"cylinder_(\d+)\."), lambda m: f"powertrain.cylinder_{m.group(1)}"),
    (re.compile(r"^(sump|oil_pan|splash|oil_trough|drain_plug)"), lambda m: "powertrain.oil_pan"),
    (re.compile(r"^(head_casting|cam_box|chamber_roof|valve_cover|valley_cover)"), lambda m: "powertrain.head"),
    (re.compile(r"^(crankcase|bedplate|main_pedestal|frame_plate|rear_main|front_cover|crank_main|crank_web|crank_pin)"),
     lambda m: "powertrain.engine"),
    (re.compile(r"^(runner|plenum|stack)"), lambda m: "powertrain.intake_plenum"),
    (re.compile(r"^(log_outlet|collector|primary|header)"), lambda m: "powertrain.exhaust_manifold"),
]


def owning_part(mesh_part: str) -> str | None:
    """The graph part a struck casting piece belongs to, or None when it
    stands alone."""
    for rx, fn in _MESH_PART_OWNERS:
        m = rx.search(mesh_part)
        if m:
            return fn(m)
    return None


def mesh_part_identity(mesh_part: str, graph: dict) -> str:
    """Map a struck mesh label back to the graph identity that owns it.
    An exact node/edge label wins; otherwise the ownership table above
    decides, and only a piece belonging to nothing keeps its own name."""
    candidates = []
    node_ids = set()
    for prefix, items in (("node_", graph.get("nodes", ())), ("edge_", graph.get("edges", ()))):
        for item in items:
            identity = str(item["identity"])
            node_ids.add(identity)
            label = prefix + identity.replace("/", "_").replace(".", "_")
            candidates.append((label, identity))
    for label, identity in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        if mesh_part == label or mesh_part.startswith(label + "_"):
            return identity
    owner = owning_part(mesh_part)
    if owner is not None and owner in node_ids:
        return owner
    return mesh_part


def sync_part_pressures(states: dict[str, PartDamageState], fluid_circuits: Iterable) -> None:
    """Copy each live circuit pressure onto all of its participating nodes."""
    circuits = tuple(fluid_circuits)
    for state in states.values():
        state.pressure_pa, state.circuit_identity = _pressure_for_part(
            state.identity, state.identity, circuits)


def record_penetration(
    states: dict[str, PartDamageState],
    graph: dict,
    fluid_circuits: Iterable,
    penetration,
) -> list[tuple[str, Puncture]]:
    """Persist every part traversal produced by ``RayMesh.penetrate``."""
    circuits = tuple(fluid_circuits)
    recorded: list[tuple[str, Puncture]] = []
    for mesh_part, impact in penetration.impacts:
        identity = mesh_part_identity(mesh_part, graph)
        state = states.setdefault(identity, PartDamageState(identity=identity))
        pressure_pa, circuit_identity = _pressure_for_part(identity, mesh_part, circuits)
        state.pressure_pa = pressure_pa
        state.circuit_identity = circuit_identity
        state.impacts.append(impact)
    for hole in penetration.holes:
        identity = mesh_part_identity(hole.part, graph)
        state = states.setdefault(identity, PartDamageState(identity=identity))
        pressure_pa, circuit_identity = _pressure_for_part(identity, hole.part, circuits)
        state.pressure_pa = pressure_pa
        state.circuit_identity = circuit_identity
        puncture = Puncture(
            material=str(hole.material),
            entry_position_m=_point(hole.position),
            exit_position_m=_point(hole.exit_position) if hole.exit_position is not None else None,
            direction=_point(hole.direction),
            radius_m=float(hole.calibre_m) / 2.0,
            through=bool(hole.through),
            resisting_thickness_m=float(hole.resisting_thickness_m),
            penetration_depth_m=float(hole.penetration_depth_m),
            energy_before_j=float(hole.energy_before_j),
            energy_spent_j=float(hole.energy_spent_j),
            damage_mode=str(hole.damage_mode),
            incidence_angle_deg=float(hole.incidence_angle_deg),
            speed_before_m_s=float(hole.speed_before_m_s),
            speed_after_m_s=float(hole.speed_after_m_s),
            projectile_integrity=float(hole.projectile_integrity),
            fluid_energy_spent_j=float(hole.fluid_energy_spent_j),
        )
        state.punctures.append(puncture)
        recorded.append((identity, puncture))
    return recorded


def _pressure_for_part(identity: str, mesh_part: str, circuits: tuple) -> tuple[float, str | None]:
    for index, circuit in enumerate(circuits):
        if identity in circuit.nodes or any(edge.get("identity") == identity for edge in circuit.edges):
            return float(circuit.pressure_pa), _circuit_identity(circuit, index)
    if "water_jacket" in mesh_part:
        for index, circuit in enumerate(circuits):
            if any("coolant" in node or "radiator" in node or "water_pump" in node for node in circuit.nodes):
                return float(circuit.pressure_pa), _circuit_identity(circuit, index)
    return ATMOSPHERIC_PRESSURE_PA, None


def _circuit_identity(circuit, index: int) -> str:
    identities = sorted(
        str(edge.get("circuit_identity"))
        for edge in circuit.edges
        if edge.get("circuit_identity")
    )
    return identities[0] if identities else f"{circuit.kind_class}:{index}"


def _point(value) -> tuple[float, float, float]:
    return tuple(float(component) for component in value)
