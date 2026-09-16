"""The installable unit, for a machine that is not an engine.

engine_package.py already answers this question for engines: an engine
becomes a PRISM carrying a declared finite set of connectors on its
boundary, split into what ships inside the crate (BUILT_IN) and what
the vehicle has to supply (SUPPLIED_ELSEWHERE), with a real mounting
plan from engine_mounts. That is the right shape and it is not reusable
as written, because `EnginePackage.build` takes an `Engine` and goes
straight to `build_drivetrain_graph`. An autoclave has no crank, no
firing order and no drivetrain graph, so it cannot be packaged at all
-- and an unpackaged machine cannot be installed in anything.

THE THREE THINGS THAT DIFFER, and they are all consequences of the same
fact: a machine authored in the production vocabulary already declares
more about itself than an engine graph does.

  THE PRISM IS MEASURED FROM BODIES, NOT FROM POINTS. EnginePackage
  takes the min/max of node positions, because a drivetrain node is a
  centroid and there is nothing else to take. A production-vocabulary
  body declares `body_half_extent_m`, so its prism is the bounding
  volume of the METAL rather than of a cloud of centres. On this
  autoclave the difference is the vessel's own radius, and a crate
  measured to the centreline of its own pressure vessel would be
  reported as fitting through a door it cannot fit through.

  MOUNTS ARE DECLARED, NOT SOLVED. engine_mounts resolves technique and
  position from a modal solve of the block, which is the right method
  for a thing whose problem is its own vibration. A vessel on saddles
  has no such problem and a real answer already: it sits where its
  saddles are. So the mounts here are the ports the machine DECLARES as
  structural mounts -- and a machine that declares none cannot be
  installed, which is reported rather than patched over by inventing
  four corners of its bounding box.

  `mass_in_total` IS NOT THE MASS TEST. It means "already counted
  somewhere else in a vehicle total", and turret_production's own
  centre-of-gravity routine records what filtering on it did: a 647 kg
  gun came back as 27.6 kg. The declared exclusion is `wrench_point`,
  which is what a prism's surface ports set on themselves -- they carry
  a nominal mass so they draw and can be hit, and counting them added
  119 kg of proxy to that same sum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from assembly_ports import PartPort
from engine_package import PartSourcing
from engine_mounts import (MountHardware, MountingTechnique, StabilityResult,
                           MountAssignment, check_stability, mount_loads,
                           _HARDWARE)
from engine_mass_properties import RigidBodyProperties


def _document(graph) -> dict:
    """Accept either a ProductionGraph or the document it serialises to."""
    if isinstance(graph, dict):
        return graph
    if hasattr(graph, "as_document"):
        return graph.as_document()
    return {"identity": getattr(graph, "identity", "machine"),
            "nodes": list(graph.nodes), "edges": list(graph.edges)}


# ---------------------------------------------------------------------
# what is in the crate
# ---------------------------------------------------------------------

def node_sourcing(node: dict) -> PartSourcing:
    """Built into this machine, or supplied by whatever it is installed in.

    Declared on the node, never inferred from its identity.
    `chassis_side` is engine_package's own existing convention for this
    -- engine_parts.py stamps it on the muffler and the tailpipe, which
    hang from the body and not from the engine -- and it carries over
    unchanged, so nothing has to learn a second word for the same fact.
    """
    if node.get("chassis_side") or node.get("supplied_elsewhere"):
        return PartSourcing.SUPPLIED_ELSEWHERE
    return PartSourcing.BUILT_IN


@dataclass(frozen=True)
class Prism:
    """The bounding volume of the machine's own metal.

    Measured over each built-in body's declared extent, so it is the box
    the machine has to be lifted through rather than the box its centres
    happen to fall in."""
    min_corner: tuple
    max_corner: tuple

    @property
    def size_m(self) -> tuple:
        return tuple(hi - lo for lo, hi in zip(self.min_corner, self.max_corner))

    @property
    def centre(self) -> tuple:
        return tuple((hi + lo) / 2.0
                     for lo, hi in zip(self.min_corner, self.max_corner))

    def contains_point(self, p, tol_m: float = 0.0) -> bool:
        return all(lo - tol_m <= p[i] <= hi + tol_m
                   for i, (lo, hi) in enumerate(zip(self.min_corner,
                                                    self.max_corner)))

    def fits_within(self, half_extent_m) -> bool:
        """Does this crate go in a box of the stated half extents?"""
        return all(s / 2.0 <= float(room) + 1e-9
                   for s, room in zip(self.size_m, half_extent_m))


def measure_prism(nodes) -> Prism:
    lo = None
    hi = None
    for n in nodes:
        if n.get("wrench_point"):
            continue                    # a port is ON the surface already
        p = np.asarray(n["reference_position"], float)
        h = np.asarray(n.get("body_half_extent_m", (0.0, 0.0, 0.0)), float)
        a, b = p - h, p + h
        lo = a if lo is None else np.minimum(lo, a)
        hi = b if hi is None else np.maximum(hi, b)
    if lo is None:
        return Prism((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return Prism(tuple(float(v) for v in lo), tuple(float(v) for v in hi))


def _axis_frame(axis) -> np.ndarray:
    """A rotation whose third column is the body's own axis."""
    a = np.asarray(axis, float)
    n = float(np.linalg.norm(a))
    a = a / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    helper = (np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9
              else np.array([0.0, 1.0, 0.0]))
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return np.column_stack([u, v, a])


def body_inertia_kg_m2(node: dict) -> np.ndarray:
    """One body's own inertia about its own centre, from the shape it
    declares.

    A POINT MASS HAS NO INERTIA OF ITS OWN, and that is not a small
    approximation here. This autoclave's vessel is 498 kg -- fifty-nine
    per cent of the machine -- and it sits within millimetres of the
    centre of gravity, so summing point masses gave it a contribution of
    almost exactly nothing. Its real tube inertia is 56 kg.m2 about a
    transverse axis, against a whole-machine total of 43 that had been
    computed without it. The tensor was not slightly low; the largest
    single term was missing.

    It matters because the rocking modes go as one over the root of it:
    a tensor that is half the truth puts every rocking frequency forty
    per cent high, and a mode table is what those frequencies are for.

    Nothing is guessed. Every body in the production vocabulary already
    declares what shape it is -- emit_prism writes tube_outer_radius_m,
    drum_radius_m or half_extent_m according to the prism it came from
    -- so this reads a declaration rather than inferring a solid from a
    bounding box."""
    m = float(node.get("mass_kg", 0.0))
    if m <= 0.0:
        return np.zeros((3, 3))
    if "tube_outer_radius_m" in node:
        ro = float(node["tube_outer_radius_m"])
        ri = float(node.get("tube_inner_radius_m", 0.0))
        L = float(node.get("drum_length_m", 0.0))
        axial = m * (ro * ro + ri * ri) / 2.0
        trans = m * (3.0 * (ro * ro + ri * ri) + L * L) / 12.0
        local = np.diag([trans, trans, axial])
        R = _axis_frame(node.get("tube_axis", (0.0, 0.0, 1.0)))
        return R @ local @ R.T
    if "drum_radius_m" in node:
        r = float(node["drum_radius_m"])
        L = float(node.get("drum_length_m", 0.0))
        axial = m * r * r / 2.0
        trans = m * (3.0 * r * r + L * L) / 12.0
        local = np.diag([trans, trans, axial])
        R = _axis_frame(node.get("drum_axis", (0.0, 0.0, 1.0)))
        return R @ local @ R.T
    h = np.asarray(node.get("body_half_extent_m", (0.0, 0.0, 0.0)), float)
    return np.diag([m * (h[1] ** 2 + h[2] ** 2) / 3.0,
                    m * (h[0] ** 2 + h[2] ** 2) / 3.0,
                    m * (h[0] ** 2 + h[1] ** 2) / 3.0])


def mass_properties(nodes) -> RigidBodyProperties:
    """What the machine weighs and where it balances.

    Excludes `wrench_point` bodies, which are proxies for a surface and
    not metal, and counts everything else -- see the module docstring on
    why `mass_in_total` is the wrong filter here."""
    masses, positions, counted = [], [], []
    for n in nodes:
        if n.get("wrench_point"):
            continue
        w = float(n.get("mass_kg", 0.0))
        if w <= 0.0:
            continue
        masses.append(w)
        positions.append(n["reference_position"])
        counted.append(n)
    if not masses:
        return RigidBodyProperties(0.0, (0.0, 0.0, 0.0), np.zeros((3, 3)))
    m = np.asarray(masses, float)
    p = np.asarray(positions, float)
    total = float(m.sum())
    cg = (m[:, None] * p).sum(axis=0) / total
    # PARALLEL AXIS, ON TOP OF EACH BODY'S OWN. The point-mass term says
    # where the mass is; the body term says what shape it is. Dropping
    # either one is wrong, and dropping the second was the larger error.
    r = p - cg
    r2 = np.sum(r * r, axis=1)
    inertia = np.zeros((3, 3))
    for ax in range(3):
        inertia[ax, ax] = float(np.sum(m * (r2 - r[:, ax] ** 2)))
    for a in range(3):
        for b in range(3):
            if a != b:
                inertia[a, b] = float(-np.sum(m * r[:, a] * r[:, b]))
    for n in counted:
        inertia = inertia + body_inertia_kg_m2(n)
    return RigidBodyProperties(total, tuple(float(v) for v in cg), inertia)


@dataclass(frozen=True)
class Connector:
    """One member that crosses the machine's own boundary.

    Two declared sources, and neither is a name test. A member the
    author marked `through-port` is one by construction -- the
    vocabulary autoclave_production already emits its vent, drain and
    vacuum lines in. An edge that reaches a node declared `chassis_side`
    is the other: the part on the far end is not this machine's to
    supply."""
    identity: str
    kind: str
    inside_node: str
    outside_node: str
    position: tuple
    bore_m: float = 0.0
    circuit: str = ""

    @property
    def label(self) -> str:
        bits = self.identity.split(".")
        return bits[-2] if len(bits) > 1 else self.identity


def boundary_connectors(nodes, edges) -> list:
    by_id = {n["identity"]: n for n in nodes}
    supplied = {i for i, n in by_id.items()
                if node_sourcing(n) is PartSourcing.SUPPLIED_ELSEWHERE}
    out = []
    for e in edges:
        a, b = e.get("a"), e.get("b")
        crosses = e.get("part_role") == "through-port"
        if not crosses:
            if a in supplied and b not in supplied:
                a, b = b, a
                crosses = True
            elif b in supplied and a not in supplied:
                crosses = True
        if not crosses:
            continue
        far = by_id.get(b) or by_id.get(a) or {}
        pos = far.get("reference_position", [0.0, 0.0, 0.0])
        out.append(Connector(
            identity=e["identity"], kind=e.get("constraint", "?"),
            inside_node=a, outside_node=b,
            position=tuple(float(v) for v in pos),
            bore_m=float(e.get("radius", 0.0)) * 2.0,
            circuit=str(e.get("circuit_identity", ""))))
    return out


# ---------------------------------------------------------------------
# where it stands
# ---------------------------------------------------------------------

#: What a node has to say about itself to be a mounting point. This is
#: the word station_reference.py already stamps on its engine-bay
#: mounts, and the kind assembly_ports.MATE_PAIRS already pairs with
#: itself -- so a declared mount here mates with a declared mount on a
#: chassis through machinery that exists, rather than a second solver.
MOUNT_PORT_ROLE = "structural-mount"


def mount_ports(nodes, owner: str) -> list:
    """The points this machine stands on, as ports the mating solver
    already understands.

    A machine that declares none gets an empty list and cannot be
    installed. That is the honest answer: the four corners of its
    bounding box would look like a mounting and would be a drawing of
    one."""
    out = []
    for n in nodes:
        if n.get("port_role") != MOUNT_PORT_ROLE:
            continue
        outward = n.get("outward", n.get("direction", (0.0, -1.0, 0.0)))
        out.append(PartPort(
            identity=n["identity"], part=owner, kind=MOUNT_PORT_ROLE,
            position=np.asarray(n["reference_position"], float),
            direction=np.asarray(outward, float),
            radius_m=float(n.get("bolt_radius_m", 0.012)), mating=True,
            fluid="", closure="",
            joint=str(n.get("joint", "bolted-flange"))))
    return out


def hardware_for(technique: MountingTechnique,
                 grade: str = "standard") -> MountHardware:
    return _HARDWARE[(technique, grade)]


@dataclass
class MachinePackage:
    """A machine as an installable unit: a prism, what crosses its
    boundary, what it weighs, and where it stands."""
    identity: str
    prism: Prism
    connectors: list
    mounts: list                      # PartPort, kind structural-mount
    rigid_body: RigidBodyProperties
    technique: MountingTechnique
    built_in_node_ids: frozenset
    supplied_node_ids: frozenset
    document: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, graph, *,
              technique: MountingTechnique = MountingTechnique.SOLID_BOLTED,
              identity: str | None = None) -> "MachinePackage":
        doc = _document(graph)
        nodes, edges = doc["nodes"], doc["edges"]
        ident = identity or doc.get("identity", "machine")
        sourcing = {n["identity"]: node_sourcing(n) for n in nodes}
        built_in = frozenset(i for i, s in sourcing.items()
                             if s is PartSourcing.BUILT_IN)
        supplied = frozenset(i for i, s in sourcing.items()
                             if s is PartSourcing.SUPPLIED_ELSEWHERE)
        inside = [n for n in nodes if n["identity"] in built_in]
        return cls(identity=ident,
                   prism=measure_prism(inside),
                   connectors=boundary_connectors(nodes, edges),
                   mounts=mount_ports(nodes, ident),
                   rigid_body=mass_properties(inside),
                   technique=technique,
                   built_in_node_ids=built_in,
                   supplied_node_ids=supplied,
                   document=doc)

    # -- what the mounting itself has to answer for --
    def stability(self) -> StabilityResult:
        """Will it stand on the points it declares?

        engine_mounts' own gate, unchanged: the centre of gravity has to
        fall inside the convex hull of the support points. It takes
        positions and a centre of gravity and knows nothing about
        engines, so a vessel on saddles is checked by exactly the test a
        block on biscuits is."""
        return check_stability(
            [tuple(float(v) for v in p.position) for p in self.mounts],
            self.rigid_body.center_of_gravity)

    def assignments(self, grade: str = "standard") -> list:
        hardware = hardware_for(self.technique, grade)
        return [MountAssignment(identity=p.identity, role="machine",
                                position=tuple(float(v) for v in p.position),
                                hardware=hardware)
                for p in self.mounts]

    def static_mount_loads(self) -> dict:
        """What each mount pushes into whatever carries it."""
        return mount_loads(self.rigid_body, self.assignments())

    def describe(self) -> str:
        sx, sy, sz = self.prism.size_m
        stability = self.stability()
        lines = [
            f"{self.identity}: {self.rigid_body.total_mass_kg:.0f} kg, "
            f"prism {sx:.2f} x {sy:.2f} x {sz:.2f} m",
            "  centre of gravity "
            + ", ".join(f"{v:+.3f}" for v in self.rigid_body.center_of_gravity),
            f"  mounts: {len(self.mounts)} declared, "
            f"{self.technique.value} -- {stability.reason}"]
        loads = self.static_mount_loads()
        for m in self.assignments():
            force = loads.get(m.identity)
            if force is None:
                continue
            lines.append(f"    {m.identity.split('.')[-1]:20s} "
                         f"{abs(force[1]) / 1000.0:6.2f} kN down")
        lines.append(f"  {len(self.connectors)} members cross the boundary:")
        for c in self.connectors:
            lines.append(f"    {c.label:12s} {c.circuit:10s} "
                         f"bore {c.bore_m * 1000:5.1f} mm")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# the same shape, from the engine side
# ---------------------------------------------------------------------

def from_engine_package(package, *, identity: str | None = None
                        ) -> "MachinePackage":
    """An EnginePackage, presented as the same installable unit.

    So there is ONE seam into a chassis rather than two. An engine's
    mounts come from engine_mounts' modal-driven policy and a vessel's
    come from where its saddles are; by the time either reaches a bay
    both are a position, a hardware grade and a face that points at the
    frame, and the bay does not need to know which kind of thing it is
    carrying.

    THE FACE POINTS DOWN. A mount sits ON something, so its mating face
    looks at the frame -- and assembly_ports will only pair two ports
    that face each other, which is what stops a mount being matched to
    one on the same side of the joint."""
    engine_prism = package.prism
    mounts = [PartPort(identity=m.identity,
                       part=identity or getattr(package.engine, "identity",
                                                "engine"),
                       kind=MOUNT_PORT_ROLE,
                       position=np.asarray(m.position, float),
                       direction=np.array([0.0, -1.0, 0.0]),
                       radius_m=0.014, mating=True, fluid="", closure="",
                       joint=("solid-welded"
                              if m.hardware.stiffness_n_per_m is None
                              else "bushed"))
              for m in package.mounting.mounts]
    connectors = [Connector(identity=c.identity, kind=c.kind,
                            inside_node=c.built_in_node,
                            outside_node=c.supplied_node,
                            position=tuple(float(v) for v in c.position))
                  for c in package.connectors]
    return MachinePackage(
        identity=identity or getattr(package.engine, "identity", "engine"),
        prism=Prism(tuple(float(v) for v in engine_prism.min_corner),
                    tuple(float(v) for v in engine_prism.max_corner)),
        connectors=connectors, mounts=mounts,
        rigid_body=package.mounting.rigid_body,
        technique=package.mounting.technique,
        built_in_node_ids=package.built_in_node_ids,
        supplied_node_ids=package.supplied_node_ids,
        document={})
