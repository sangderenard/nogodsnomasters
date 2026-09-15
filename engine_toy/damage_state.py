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
    """One tracked piece of the machine, and the fluid it is holding back.

    The pressure across this part is READ THROUGH its circuit rather than
    copied onto it. A machine has a few hundred tracked parts and about
    half a dozen fluid circuits, so broadcasting each circuit pressure onto
    every one of its member parts every tick was writing six distinct
    numbers into hundreds of objects -- redundant by construction, and
    able to go stale the moment anything read a part between the write
    and the next tick. The binding is the index: set once when the graph
    is built (bind_part_circuits), read live.
    """
    identity: str
    kind: str = "mesh-part"
    fluid_volume_l: float = 0.0
    circuit_identity: str | None = None
    punctures: list[Puncture] = field(default_factory=list)
    impacts: list[ImpactResult] = field(default_factory=list)
    # the circuit this part is part of, or None for a part open to the
    # air. Bound by bind_part_circuits; never copied from.
    circuit: object | None = field(default=None, repr=False, compare=False)
    # what to report when this part is on no circuit at all
    ambient_pressure_pa: float = ATMOSPHERIC_PRESSURE_PA

    @property
    def pressure_pa(self) -> float:
        """The live pressure this part is holding back, this instant."""
        circuit = self.circuit
        if circuit is None:
            return self.ambient_pressure_pa
        return float(circuit.pressure_pa)


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


def _bind_one(state: "PartDamageState", mesh_part: str, index: tuple) -> None:
    """Bind one part to its circuit. `mesh_part` is the name the geometry
    knows it by, which is what the water-jacket fallback keys on -- it can
    differ from the graph identity the state is filed under."""
    by_identity, coolant_fallback = index
    found = by_identity.get(state.identity)
    if found is None and "water_jacket" in mesh_part:
        found = coolant_fallback
    if found is None:
        state.circuit, state.circuit_identity = None, None
    else:
        state.circuit, state.circuit_identity = found


def bind_part_circuits(states: dict[str, PartDamageState], fluid_circuits: Iterable) -> None:
    """Point every tracked part at the circuit it belongs to.

    Topology, not state: a circuit's membership is fixed the moment the
    graph is built (FluidCircuit.nodes is a frozenset and .edges a tuple),
    so this runs when the graph changes, not every tick. Pressure is then
    read through the binding by PartDamageState.pressure_pa and is always
    current.
    """
    index = _circuit_index(tuple(fluid_circuits))
    for state in states.values():
        _bind_one(state, state.identity, index)


# The old name, kept because binding is exactly what callers who used to
# ask for a pressure sync actually need -- and it is now idempotent and
# cheap enough that calling it on a graph change is the whole story.
sync_part_pressures = bind_part_circuits


def record_penetration(
    states: dict[str, PartDamageState],
    graph: dict,
    fluid_circuits: Iterable,
    penetration,
) -> list[tuple[str, Puncture]]:
    """Persist every part traversal produced by ``RayMesh.penetrate``."""
    circuits = tuple(fluid_circuits)
    index = _circuit_index(circuits)
    recorded: list[tuple[str, Puncture]] = []
    for mesh_part, impact in penetration.impacts:
        identity = mesh_part_identity(mesh_part, graph)
        state = states.setdefault(identity, PartDamageState(identity=identity))
        _bind_one(state, mesh_part, index)
        state.impacts.append(impact)
    for hole in penetration.holes:
        identity = mesh_part_identity(hole.part, graph)
        state = states.setdefault(identity, PartDamageState(identity=identity))
        _bind_one(state, hole.part, index)
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


def _circuit_index(circuits: tuple) -> tuple[dict, tuple | None]:
    """Every node and edge identity in these circuits -> that
    (circuit, circuit identity), plus the coolant circuit a water jacket
    falls back to. One pass over the circuits instead of one pass per part.

    First circuit wins for an identity carried by more than one, and the
    coolant fallback is the first coolant-ish circuit -- the same answers
    the per-part linear scan gave, since it returned on its first match
    in the same order.
    """
    by_identity: dict[str, tuple] = {}
    coolant_fallback: tuple | None = None
    for index, circuit in enumerate(circuits):
        entry = (circuit, _circuit_identity(circuit, index))
        for node in circuit.nodes:
            by_identity.setdefault(str(node), entry)
        for edge in circuit.edges:
            edge_identity = edge.get("identity")
            if edge_identity is not None:
                by_identity.setdefault(str(edge_identity), entry)
        if coolant_fallback is None and any(
                "coolant" in node or "radiator" in node or "water_pump" in node for node in circuit.nodes):
            coolant_fallback = entry
    return by_identity, coolant_fallback


def _pressure_for_part(identity: str, mesh_part: str, circuits: tuple,
                       index: "tuple[dict, tuple | None] | None" = None) -> tuple[float, str | None]:
    by_identity, coolant_fallback = index if index is not None else _circuit_index(circuits)
    entry = by_identity.get(identity)
    if entry is None and "water_jacket" in mesh_part:
        entry = coolant_fallback
    if entry is None:
        return ATMOSPHERIC_PRESSURE_PA, None
    circuit, circuit_identity = entry
    return float(circuit.pressure_pa), circuit_identity


def _circuit_identity(circuit, index: int) -> str:
    named = circuit.edge_identity          # computed once, see FluidCircuit
    return named if named is not None else f"{circuit.kind_class}:{index}"


def _point(value) -> tuple[float, float, float]:
    return tuple(float(component) for component in value)
