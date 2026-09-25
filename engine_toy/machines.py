"""Machines: things you can load the way you load an engine.

An engine is one kind of machine, not the only kind. A hydraulic power
pack, a scissor lift, a log splitter and a turret are all machines in
exactly the sense that matters here: they are an assembly of real parts
with real dimensions, they hold fluids, they can be shot at and broken,
and they do work. What they do NOT have is a crank, a firing order, a
subframe, engine mounts, or a torque output anyone would put on a dyno.

So this module builds the same kind of graph `drivetrain_graph.py`
builds -- the same node and edge documents, with the same declared
fields -- which means every system downstream already works on them
without knowing what they are:

  engine_mesh / vehicle_mesh   draw them, because a node with a body
                               extent is a box and an edge with a radius
                               is a tube
  engine_rays                  shoot them
  damage_state / hole_emitters leak them
  fluids / gear_cases          say what is inside them
  equipment / actuators        make them move
  bench                        feed them

The one thing that had to change elsewhere was view membership. The
engine view keeps parts by matching their names against a list of
engine keywords, which was never going to contain "scissor arm" or
"turret ring". Nodes here DECLARE `in_view` instead, and
`engine_mesh.declared_in_view` honours that.

A machine declares where its power comes from:

  self-contained   it carries its own power pack or compressor, and
                   needs nothing but electricity
  shore-supplied   it expects a hose from somewhere else, and the rig
                   makes that up (bench.py)
  engine-driven    it expects to be bolted to an engine's plant

which is the same question `bench.PartRig` asks about unsatisfied
inlets, asked once for the whole machine.
"""
from __future__ import annotations

import math
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine
from src.common.dt_system.error_channels import empty_channels
from src.common.dt_system.time_contracts import HOLD
from src.common.tensors import AbstractTensor

from actuators import LinearActuator, RotaryActuator, FluidMotor
from bench import HydraulicPowerUnit, AirSupply

ATM_PA = 101_325.0


# =====================================================================
#  THE DOCUMENT A MACHINE BUILDS
# =====================================================================
@dataclass
class MachinePart:
    """One real part of a machine: a body with dimensions, a place, a
    material, and whatever it happens to contain."""
    identity: str
    kind: str
    position: tuple[float, float, float]
    half_extent_m: tuple[float, float, float]
    mass_kg: float = 5.0
    material: str = "steel-plate"
    fluid: str | None = None
    fluid_volume_l: float = 0.0
    capacity_kg: float = 0.0
    part_role: str = ""
    actuator: object | None = None        # the live part, if it moves
    # the speed this function is DESIGNED to run at. A scissor lift
    # platform rises at a walking pace and a splitter ram advances at
    # about a hand's width a second; assuming one nominal speed for
    # every machine asked a 6 kW pack for 94 L/min, which no such pack
    # has ever made.
    duty_speed_m_s: float = 0.10
    duty_rate_deg_s: float = 45.0
    # Where this part was BUILT. Motion is always recomputed from here
    # rather than accumulated onto the live position, because an
    # incremental update run every tick drifts and cannot be run twice
    # for the same state.
    base_position: tuple | None = None
    # Non-engine machines use the same explicit boundary as engine
    # castings: a port says what may cross the body boundary and where.
    # These are assembly_ports.PartPort records, kept as objects until
    # build_graph serialises the machine document.
    ports: list = field(default_factory=list)
    temperature_k: float = 293.15
    thermal_capacity_j_k: float = 0.0
    attributes: dict = field(default_factory=dict)


@dataclass
class MachineLine:
    """One real connection between two parts."""
    identity: str
    a: str
    b: str
    constraint: str = "fluid-line"
    radius: float = 0.006
    circuit_identity: str = "hydraulic"
    material: str = "steel-pipe"
    attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MachineWorkEdge:
    """A shaped working edge carried by one real machine part.

    ``a_local`` and ``b_local`` are measured in the owning part's local
    coordinates.  The edge is not another body and it is not a visual-only
    line: it is the finite contact locus through which a cutter, scraper or
    abrasive transfers work into another object.  Tooth form and wave/set are
    declared here so a cutting engine can use the same edge a renderer draws.
    """
    identity: str
    owner: str
    a_local: tuple[float, float, float]
    b_local: tuple[float, float, float]
    thickness_m: float
    wave_amplitude_m: float = 0.0
    wave_length_m: float = 0.0
    tooth_pattern: str = "plain"
    attributes: dict = field(default_factory=dict)

    def to_data(self) -> dict:
        return {
            "identity": self.identity,
            "owner": self.owner,
            "a_local": [float(v) for v in self.a_local],
            "b_local": [float(v) for v in self.b_local],
            "thickness_m": float(self.thickness_m),
            "wave_amplitude_m": float(self.wave_amplitude_m),
            "wave_length_m": float(self.wave_length_m),
            "tooth_pattern": self.tooth_pattern,
            **self.attributes,
        }


@dataclass(frozen=True)
class MachineAction:
    """An input gesture a machine exposes to its owning interaction layer."""
    identity: str
    gesture: str
    operation: str
    destination: str
    attributes: dict = field(default_factory=dict)

    def to_data(self) -> dict:
        return {
            "identity": self.identity,
            "gesture": self.gesture,
            "operation": self.operation,
            "destination": self.destination,
            **self.attributes,
        }


@dataclass(frozen=True)
class MachinePose:
    """A Machine-declared pose relative to an owning hand.

    ``offset_hand_m`` is (screen-right, up, forward). ``rotation_deg_xyz``
    rotates the machine after its local X axis has been aligned with the
    hand's forward direction. Pose names are interaction states, not new
    objects; the same Machine and parts remain authoritative throughout.
    """
    name: str
    offset_hand_m: tuple[float, float, float]
    rotation_deg_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_data(self) -> dict:
        return {
            "name": self.name,
            "offset_hand_m": [float(v) for v in self.offset_hand_m],
            "rotation_deg_xyz": [float(v) for v in self.rotation_deg_xyz],
        }


@dataclass
class Machine:
    """A loadable machine that is not an engine."""
    identity: str
    label: str
    power: str = "self-contained"        # self-contained | shore-supplied | engine-driven
    medium: str = "hydraulic"
    parts: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    # its own supply, when it carries one
    hpu: HydraulicPowerUnit | None = None
    air: AirSupply | None = None
    note: str = ""
    # a mount that fires something carries a recoil system, and the
    # rotating mass is what the traverse actuator has to accelerate
    recoil: object | None = None
    rotating_mass_kg: float = 0.0
    sight_cartridge: str = ""
    sight_zero_range_m: float = 100.0
    weapon: "Weapon | None" = None
    recoiling_mass_kg: float = 0.0
    linkages: list = field(default_factory=list)
    # Working edges and input actions are ordinary machine declarations.
    # They carry no private runtime and introduce no second object system;
    # engines which understand them read these records from the same graph.
    work_edges: list[MachineWorkEdge] = field(default_factory=list)
    actions: list[MachineAction] = field(default_factory=list)
    interaction_poses: list[MachinePose] = field(default_factory=list)
    # A MACHINE MAY BE AUTHORED IN THE PRODUCTION VOCABULARY INSTEAD.
    # When it is, that document IS the structure -- every body, every
    # member, every declared motion group -- and the `parts` list below
    # carries only the plant that drives it (pump, accumulator, rams),
    # which is runtime state rather than geometry. This is how the two
    # descriptions of this turret were reconciled: there is now one
    # structure, authored once in `turret_production`, and the machine
    # layer supplies the hydraulics that move it rather than a second
    # opinion about where its parts are.
    production_graph: dict | None = None

    # ---------------------------------------------------------------
    def sync_supply_to_parts(self) -> None:
        """The power pack holds what this machine's declared reservoir
        holds. Leaving the supply on its own default gave a 12 L scissor
        lift a 54 L tank, which is the kind of disagreement between two
        descriptions of one thing that this project keeps fixing."""
        if self.hpu is None:
            return
        res = next((p for p in self.parts if p.part_role == "reservoir"
                    and p.fluid_volume_l > 0.0), None)
        if res is not None:
            self.hpu.oil_l = float(res.fluid_volume_l)
            self.hpu.reservoir_l = float(res.fluid_volume_l) * 1.2

    def build_graph(self) -> dict:
        """The same document an engine builds, so everything downstream
        works without being told what this is.

        If this machine was authored in the production vocabulary, that
        document supplies the structure and the parts below are merged
        in as the plant that drives it."""
        nodes = []
        for p in self.parts:
            n = {"identity": p.identity, "kind": p.kind,
                 "reference_position": [float(v) for v in p.position],
                 "body_half_extent_m": [float(v) for v in p.half_extent_m],
                 "mass_kg": float(p.mass_kg), "mass_in_total": True,
                 "material": p.material,
                 "temperature_k": float(p.temperature_k),
                 "thermal_capacity_j_k": float(
                     p.thermal_capacity_j_k or p.mass_kg * 500.0),
                 # declared, so the view keeps it without anyone having
                 # to add "scissor arm" to a list of engine keywords
                 "in_view": True}
            if p.fluid:
                n["fluid"] = p.fluid
                n["fluid_volume_l"] = float(p.fluid_volume_l)
            if p.capacity_kg > 0.0:
                n["capacity_kg"] = float(p.capacity_kg)
            if p.part_role:
                n["part_role"] = p.part_role
            n.update(p.attributes)
            if p.ports:
                n["ports"] = [{
                    "identity": port.identity,
                    "part": port.part,
                    "kind": port.kind,
                    "position": [float(v) for v in port.position],
                    "direction": [float(v) for v in port.direction],
                    "radius_m": float(port.radius_m),
                    "mating": bool(port.mating),
                    "connected_to": port.connected_to,
                    "fluid": port.fluid,
                    "joint": port.joint,
                    "joint_axis": port.joint_axis,
                    **({"compression_sleeve":
                        port.compression_sleeve.to_data()}
                       if getattr(port, "compression_sleeve", None) is not None
                       else {}),
                    **({"fastener_areas": [area.to_data() for area in
                                           port.fastener_areas]}
                       if getattr(port, "fastener_areas", ()) else {}),
                    **({"glue_surfaces": [surface.to_data() for surface in
                                          port.glue_surfaces]}
                       if getattr(port, "glue_surfaces", ()) else {}),
                    **({"port_role": port.role}
                       if getattr(port, "role", "") else {}),
                } for port in p.ports]
            nodes.append(n)
        edges = []
        for line in self.lines:
            edge = {
                "identity": line.identity, "a": line.a, "b": line.b,
                "constraint": line.constraint, "radius": float(line.radius),
                "circuit_identity": line.circuit_identity,
                "material": line.material, "in_view": True,
            }
            edge.update(line.attributes)
            edges.append(edge)
        if self.production_graph is not None:
            base = self.production_graph
            # the plant sits inside the well with the structure; a
            # plant part that repeats a body already authored in the
            # production document would be a second opinion about one
            # thing, so it is refused rather than silently shadowing it
            authored = {n["identity"] for n in base["nodes"]}
            clash = authored & {n["identity"] for n in nodes}
            if clash:
                raise ValueError(
                    f"{self.identity}: {sorted(clash)} authored both in the "
                    f"production document and as machine parts")
            return {**base, "identity": f"{self.identity}/machine",
                    "machine": True, "label": self.label,
                    "nodes": list(base["nodes"]) + nodes,
                    "edges": list(base["edges"]) + edges,
                    "work_edges": [edge.to_data() for edge in self.work_edges],
                    "actions": [action.to_data() for action in self.actions],
                    "interaction_poses": [pose.to_data()
                                          for pose in self.interaction_poses]}
        return {"schema": "engine-toy-drivetrain-graph-v1",
                "identity": f"{self.identity}/machine",
                "machine": True, "label": self.label,
                "nodes": nodes, "edges": edges,
                "work_edges": [edge.to_data() for edge in self.work_edges],
                "actions": [action.to_data() for action in self.actions],
                "interaction_poses": [pose.to_data()
                                      for pose in self.interaction_poses]}

    def interaction_pose(self, name: str) -> MachinePose:
        """Resolve a declared hand pose, with a usable generic fallback."""
        pose = next((item for item in self.interaction_poses
                     if item.name == name), None)
        if pose is not None:
            return pose
        selected = next((item for item in self.interaction_poses
                         if item.name == "selected"), None)
        if selected is not None:
            return selected
        return MachinePose("selected", (0.0, -0.34, 0.58))

    @property
    def moving_parts(self) -> list:
        return [p for p in self.parts if p.actuator is not None]

    def describe(self) -> list[str]:
        out = [f"  {self.label}  ({self.power}, {self.medium})",
               f"    {len(self.parts)} parts, {len(self.moving_parts)} of them powered, "
               f"{sum(p.mass_kg for p in self.parts):6.0f} kg"]
        if self.note:
            out.append(f"    {self.note}")
        if self.hpu is not None:
            out.extend(self.hpu.describe())
        if self.air is not None:
            out.extend(self.air.describe())
        return out


@dataclass
class Weapon:
    """A gun, by the three numbers that decide what the mount has to
    survive: what leaves the barrel, how fast, and how much propellant
    goes with it."""
    identity: str
    projectile_kg: float
    muzzle_m_s: float
    charge_kg: float
    rate_per_min: float = 0.0

    @property
    def impulse_n_s(self) -> float:
        """Momentum out of the muzzle, and therefore momentum into the
        mount. The propellant gas counts: it leaves at roughly one and a
        half times the projectile's speed, and on a large charge it is a
        third of the whole recoil impulse -- which is why a muzzle brake
        works on the gas rather than on the shell."""
        return self.projectile_kg * self.muzzle_m_s + self.charge_kg * 1.5 * self.muzzle_m_s

    def free_recoil(self, recoiling_mass_kg: float) -> tuple[float, float]:
        """(velocity, energy) of the recoiling mass if nothing stopped
        it."""
        v = self.impulse_n_s / max(recoiling_mass_kg, 1e-6)
        return v, 0.5 * recoiling_mass_kg * v * v


WEAPONS = {
    # Bofors 40 mm L/70: the classic mount-sized autocannon
    "40mm-l70": Weapon("40 mm L/70", projectile_kg=0.87, muzzle_m_s=1010.0,
                       charge_kg=0.31, rate_per_min=330.0),
    "30mm-chain": Weapon("30 mm chain gun", projectile_kg=0.36, muzzle_m_s=1080.0,
                         charge_kg=0.13, rate_per_min=200.0),
    "50-bmg": Weapon(".50 BMG", projectile_kg=0.043, muzzle_m_s=890.0,
                     charge_kg=0.015, rate_per_min=550.0),
}


def size_recoil_for(weapon: "Weapon", recoiling_mass_kg: float, stroke_m: float,
                    bore_m: float = 0.095, peak_force_n: float = 120_000.0) -> object:
    """Size an oleo-pneumatic recoil system to a gun.

    This is real gun-mount arithmetic, and it is all consequence:

      The RECUPERATOR (the gas side) has one job that sets its minimum
      charge -- it must return the barrel to battery and hold it there
      against gravity and the mount's own angle. Too little and the gun
      does not run out; too much and it fights the recoil stroke it is
      supposed to allow.

      The BRAKE (the oil side) must absorb the free-recoil energy inside
      the stroke available. Average force is simply energy over stroke,
      and the orifice is then sized so that the loss at the recoil
      velocity produces about that force. Because orifice loss goes with
      the SQUARE of velocity, a system sized for one charge is badly
      wrong for another -- which is exactly why real guns with variable
      charges have a metering pin that changes the orifice area through
      the stroke instead of a fixed hole.
    """
    from actuators import GasOverOilStrut, OIL_DENSITY_KG_M3
    v, energy = weapon.free_recoil(recoiling_mass_kg)
    area = math.pi * bore_m * bore_m / 4.0
    # the brake has to do `energy` joules of work in `stroke` metres
    mean_force = energy / max(stroke_m, 1e-6)
    force = min(max(mean_force * 1.6, 1.0), peak_force_n)     # peak is above the mean
    # orifice from that force at the recoil velocity:
    #   dp = rho/2 * (A v / Ao)^2  and  F = dp * A
    dp = force / max(area, 1e-9)
    a_o = area * v * math.sqrt(OIL_DENSITY_KG_M3 / (2.0 * max(dp, 1.0)))
    orifice_d = math.sqrt(max(a_o, 1e-10) / 0.7 * 4.0 / math.pi)
    # the recuperator: enough to hold the recoiling mass at full
    # elevation with margin, and no more
    hold_n = recoiling_mass_kg * 9.81 * 1.35
    charge_pa = max(5e5, hold_n / area)
    swept_l = area * stroke_m * 1000.0
    return GasOverOilStrut(identity=f"recoil/{weapon.identity}", bore_m=bore_m, stroke_m=stroke_m,
                           gas_volume_l=swept_l * 2.6,          # ~2.6:1 at full recoil
                           charge_pressure_pa=charge_pa,
                           orifice_diameter_m=orifice_d, polytropic_n=1.35)


@dataclass
class Linkage:
    """PRESCRIBED motion -- a STAND-IN, not the answer.

    This moves geometry because an actuator says so. That is kinematics,
    not dynamics: nothing here solves forces, nothing resists, and a
    turret whose shell rises because a rod extended is not obeying
    physics, it is obeying a rule that says it should look as though it
    did. It is explicitly NOT what this machine should ship with.

    The real path is the production vehicle physics
    (turing/src/compiler): a node-and-member network with a
    geometry-agnostic elastic/plastic/fracture member law
    (`vehicle_mechanical_material.symbolic_vehicle_member_material_
    equations` -- a J2 beam/tube return map taking axial, bending and
    shear strains and their rates). In that solver a hydraulic cylinder
    is a MEMBER WHOSE REST LENGTH IS COMMANDED by the fluid volume
    pushed into it, with a stiffness set by the fluid's own bulk
    modulus, and the structure articulates because the members pull it
    there -- which is the only way the pressure/force relationship this
    module already computes actually communicates power into the world.

    Kept for now so the geometry is not frozen while that integration is
    built, and so the two can be compared directly.

    What an actuator physically moves, and in what way.

    THIS IS THE PIECE THAT WAS MISSING. Up to now a machine's hydraulics
    were real -- pressure, flow, force, timing all computed from the
    hardware -- but nothing in the geometry ever moved, so a render
    showed a static armature no matter what the actuator was doing. An
    actuator that makes 32 kN and extends 850 mm is not visibly doing
    anything unless something is bolted to its rod.

    Three modes, which is all a mount of this kind needs:

      lift        the parts ride the rod: y increases with extension.
      rotate-y    the parts swing about the machine's vertical axis by
                  the actuator's angle. A rack-and-pinion turning a slew
                  ring is exactly this.
      pitch       the parts rotate about a horizontal axis through a
                  declared pivot -- elevation. The actuator is a
                  straight-line device and the barrel goes round an arc,
                  so the relation between them is the geometry of the
                  triangle they form: extension along one side, angle at
                  the pivot. Approximated here as a linear map across
                  the declared angular range, which is what a real
                  mount's own elevation table does too.
    """
    actuator: str                  # the part identity whose actuator drives this
    mode: str                      # "lift" | "rotate-y" | "pitch"
    parts: tuple = ()              # the identities it carries
    pivot: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 1.0, 0.0)
    angle_range_deg: tuple = (-10.0, 45.0)   # pitch only: retracted -> extended


def _rotate_y(point, pivot, degrees: float):
    c, sn = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    dx, dz = point[0] - pivot[0], point[2] - pivot[2]
    return (pivot[0] + dx * c - dz * sn, point[1], pivot[2] + dx * sn + dz * c)


def _rotate_pitch(point, pivot, degrees: float):
    """Rotate about the X axis through the pivot: nose up is positive."""
    c, sn = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    dy, dz = point[1] - pivot[1], point[2] - pivot[2]
    return (point[0], pivot[1] + dy * c - dz * sn, pivot[2] + dy * sn + dz * c)


# =====================================================================
#  A MACHINE UNDER POWER
# =====================================================================
@dataclass
class MachineSim:
    """A machine running: its supply, its actuators, and every hole
    anybody puts in it.

    Deliberately the same shape as the engine sim where it matters --
    `state.part_damage`, `state.absent_parts`, `hole_emitters` -- so the
    damage, leak and view code paths do not have to care which one they
    are looking at."""
    machine: Machine
    graph: dict = field(default_factory=dict)
    commands: dict = field(default_factory=dict)
    loads: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    # supply readings
    supply_pressure_pa: float = ATM_PA
    supply_flow_l_min: float = 0.0
    demand_l_min: float = 0.0
    made_up_by_bench_l_min: float = 0.0
    notes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        from damage_state import register_graph_parts
        from hole_emitters import HoleEmitterField
        from burst import BurstField
        from fire import FireField
        from fittings import FittingField
        from ordnance import OrdnanceField
        from types import SimpleNamespace
        for _p in self.machine.parts:
            if _p.base_position is None:
                _p.base_position = tuple(_p.position)
        # A PREBUILT GRAPH IS HONOURED. The sim built its own from the
        # Machine every time, which meant the live loop could only ever
        # run the catalogue turret -- the production graph, with the
        # joints, the journal, the ring joints and the real sections,
        # had no way into the game engine at all and was only ever
        # solved offline in scripts. Passing `graph=` runs THAT in here.
        if not self.graph:
            self.graph = self.machine.build_graph()
        self.state = SimpleNamespace(part_damage={}, absent_parts=set(),
                                     coolant_lost_l=0.0, crank_angle_deg=0.0)
        register_graph_parts(self.state.part_damage, self.graph)
        self.hole_emitters = HoleEmitterField()
        self.bursts = BurstField()
        self.fires = FireField()
        self.fittings = FittingField()
        self.ordnance = OrdnanceField()
        self.damage_events: list = []
        self.cascade_log: list = []
        self._cascade_depth = 0
        self.ray_mesh_factory = None
        # A machine may be coupled to the graph-discovered fluid system by
        # its owning scene.  This is a reference to that system's existing
        # circuit objects, never another circuit ledger.
        self._fluid_circuits = ()
        # a machine with no supply of its own gets one, and is told so
        if self.machine.hpu is None and self.machine.medium == "hydraulic":
            self.machine.hpu = HydraulicPowerUnit(identity=f"{self.machine.identity}-hpu")
            if self.machine.power != "self-contained":
                self.notes.append(f"  supply: {self.machine.identity} is {self.machine.power}; "
                                  f"the rig is providing the hose")
        self.machine.sync_supply_to_parts()
        if self.machine.air is None and self.machine.medium == "pneumatic":
            self.machine.air = AirSupply(identity=f"{self.machine.identity}-air")

    def snapshot(self):
        # Circuit state is checkpointed by FluidCircuitSystem when it is a
        # sibling dt participant.  Copying it here would split the one shared
        # ledger into two object graphs on restore.
        values = {key: value for key, value in vars(self).items()
                  if key != "_fluid_circuits"}
        return copy.deepcopy(values)

    def restore(self, snapshot) -> None:
        circuits = self._fluid_circuits
        vars(self).clear()
        vars(self).update(copy.deepcopy(snapshot))
        self._fluid_circuits = circuits

    def bind_fluid_circuits(self, circuits) -> None:
        """Bind damage and emitters to the owning fluid system's ledger."""
        from damage_state import bind_part_circuits
        self._fluid_circuits = tuple(circuits)
        bind_part_circuits(self.state.part_damage, self._fluid_circuits)

    def apply_penetration(self, penetration):
        """Apply the common ray-damage ABI to any graph-backed machine."""
        from damage_state import record_penetration
        recorded = record_penetration(
            self.state.part_damage, self.graph, self._fluid_circuits,
            penetration)
        made = self.hole_emitters.add_from_punctures(
            recorded, self.graph, self._fluid_circuits)
        for emitter in self.hole_emitters.emitters:
            emitter.spray_resolved = False
        if recorded:
            self.damage_events.append({
                "kind": "penetration",
                "parts": tuple(identity for identity, _ in recorded),
                "punctures": len(recorded),
                "boundary_holes": sum(p.boundary_holes for _, p in recorded),
                "emitters": tuple(emitter.identity for emitter in made),
            })
        return recorded

    # ---------------------------------------------------------------
    def command(self, name: str, value: float) -> None:
        self.commands[name] = value

    def load(self, name: str, value: float) -> None:
        self.loads[name] = value

    def fire_recoil(self, impulse_n_s: float) -> None:
        """A shot lands on the recoil slide.

        The velocity it adds is ADDED to whatever the strut is already
        doing, not set. That is the whole of "partial recovery": if the
        slide has not finished returning to battery from the last round
        -- still compressed, still moving -- the next round's impulse
        stacks onto that state instead of starting fresh from zero. A
        real gun run near its cyclic rate does exactly this, which is
        why the stroke is sized generously: the margin is what makes
        stacked recoil survivable rather than a structural event."""
        strut = getattr(self.machine, "recoil", None)
        mass = getattr(self.machine, "recoiling_mass_kg", 0.0)
        if strut is None or mass <= 0.0:
            return
        # POSITIVE velocity_m_s means increasing compression in this
        # module's own convention (GasOverOilStrut.step:
        # `compression_m + velocity_m_s * dt`), so a shot -- which
        # drives the slide INTO recoil, i.e. compression increasing --
        # ADDS a positive kick. Subtracting it here first drove the
        # velocity negative, which the compression clamp at 0 then
        # swallowed instantly: every shot showed 0 mm of travel and
        # "partial recovery" could never be observed at all.
        strut.velocity_m_s += impulse_n_s / mass

    def step(self, dt: float) -> dict:
        """One tick: work out what the actuators want, serve it from the
        machine's own supply, and move them with what they got."""
        self.elapsed_s += dt
        strut = getattr(self.machine, "recoil", None)
        mass = getattr(self.machine, "recoiling_mass_kg", 0.0)
        if strut is not None and mass > 0.0:
            # the slide runs on its own clock every tick, independent of
            # whether anything just fired: a real spring/damper that
            # keeps moving -- and keeps carrying whatever velocity is
            # left in it -- until it settles at battery on its own
            strut.dynamics_step(dt, mass)
        moving = [p for p in self.machine.moving_parts
                  if p.identity not in self.state.absent_parts]
        demand = 0.0
        need_p = ATM_PA
        for p in moving:
            a = p.actuator
            cmd = abs(float(self.commands.get(p.identity, 0.0)))
            if hasattr(a, "area_extend_m2"):
                demand += cmd * a.area_extend_m2 * p.duty_speed_m_s * 60_000.0
                seal = getattr(a, "seal_friction_frac", 0.05)
                if abs(getattr(a, "velocity_m_s", 0.0)) < 0.002:
                    seal *= 1.8
                need_p = max(need_p, ATM_PA + abs(float(self.loads.get(p.identity, 0.0)))
                             / max(a.area_extend_m2 * (1.0 - seal), 1e-9))
            elif hasattr(a, "displacement_l_per_rev"):
                demand += cmd * a.displacement_l_per_rev * (p.duty_rate_deg_s / 360.0) * 60.0
                rated = getattr(a, "rated_pressure_pa", 21e6)
                at_rated = a.torque_at(rated)
                if at_rated > 0.0:
                    need_p = max(need_p, ATM_PA + abs(float(self.loads.get(p.identity, 0.0)))
                                 / at_rated * rated)
        self.demand_l_min = demand

        if self.machine.hpu is not None:
            out = self.machine.hpu.step(dt, demand, need_p)
            self.supply_pressure_pa = out["pressure_pa"]
            self.supply_flow_l_min = out["delivered_l_min"]
            self.made_up_by_bench_l_min = max(0.0, demand - out["delivered_l_min"])
        elif self.machine.air is not None:
            out = self.machine.air.step(dt, demand * 6.0)
            self.supply_pressure_pa = out["pressure_pa"]
            self.supply_flow_l_min = out["delivered_l_min"] / 6.0

        results = {}
        for p in moving:
            a = p.actuator
            cmd = float(self.commands.get(p.identity, 0.0))
            want = abs(cmd) * (a.area_extend_m2 * p.duty_speed_m_s * 60_000.0
                               if hasattr(a, "area_extend_m2")
                               else getattr(a, "displacement_l_per_rev", 0.1)
                               * (p.duty_rate_deg_s / 360.0) * 60.0)
            share = want / demand if demand > 1e-9 else 0.0
            results[p.identity] = a.step(dt, self.supply_pressure_pa,
                                         self.supply_flow_l_min * share, cmd,
                                         float(self.loads.get(p.identity, 0.0)))
        self.apply_linkage()
        # holes leak whatever the machine holds
        self.hole_emitters.step(dt, self._fluid_circuits, 0.0, 293.15)
        if self.bursts.bursts:
            self.bursts.step(dt, self.graph)
        return results


    def apply_linkage(self) -> None:
        """Move the geometry to where the actuators have put it.

        Recomputed from each part's BASE pose every time, so calling it
        twice for one state is harmless and nothing drifts. The graph's
        own node positions are updated in place, which means a re-mesh,
        a ray cast and a render all see the machine in its real
        attitude rather than in the pose it was built in."""
        m = self.machine
        if not m.linkages:
            return
        by_id = {p.identity: p for p in m.parts}
        node_by_id = {n["identity"]: n for n in self.graph["nodes"]}
        moved: dict = {}
        for link in m.linkages:
            src = by_id.get(link.actuator)
            if src is None or src.actuator is None:
                continue
            a = src.actuator
            for ident in link.parts:
                part = by_id.get(ident)
                if part is None:
                    continue
                base = moved.get(ident) or part.base_position or part.position
                if link.mode == "lift":
                    pos = (base[0], base[1] + float(getattr(a, "position_m", 0.0)), base[2])
                elif link.mode == "rotate-y":
                    pos = _rotate_y(base, link.pivot, float(getattr(a, "angle_deg", 0.0)))
                elif link.mode == "pitch":
                    lo, hi = link.angle_range_deg
                    frac = (float(getattr(a, "position_m", 0.0))
                            / max(float(getattr(a, "stroke_m", 1.0)), 1e-9))
                    pos = _rotate_pitch(base, link.pivot, lo + (hi - lo) * frac)
                else:
                    continue
                moved[ident] = pos
        for ident, pos in moved.items():
            part = by_id.get(ident)
            if part is not None:
                part.position = pos
            node = node_by_id.get(ident)
            if node is not None:
                node["reference_position"] = [float(v) for v in pos]

    # ---------------------------------------------------------------
    def burst_part(self, identity: str) -> None:
        """Destroy one part and spill what it held, exactly as the
        engine sim does."""
        import burst as burst_module
        from fluids import fluid_key as _fk
        from hole_emitters import HoleEmitter
        node = next((n for n in self.graph["nodes"] if n["identity"] == identity), None)
        if node is None or identity in self.state.absent_parts:
            return
        pts = [n["reference_position"] for n in self.graph["nodes"]]
        floor_y = min(p[1] for p in pts) - 0.25 if pts else -0.5
        b = burst_module.make_burst(self.graph, identity, 1500.0, self.bursts.rng, floor_y)
        self.bursts.bursts.append(b)
        self.state.absent_parts.add(identity)
        litres = float(node.get("fluid_volume_l", 0.0) or 0.0)
        key = _fk(node.get("fluid"))
        if litres > 0.0 and key and key != "gas":
            half = node.get("body_half_extent_m") or (0.05, 0.05, 0.05)
            self.hole_emitters.emitters.append(HoleEmitter(
                identity=f"{identity}.contents", part=identity, circuit=None, fluid=key,
                position=tuple(float(v) for v in node["reference_position"]),
                direction=(0.0, -1.0, 0.0), radius_m=max(0.01, 0.35 * min(float(half[0]), float(half[2]))),
                through=False, kind="contained", contained_l=litres,
                contained_head_m=max(0.05, float(half[1]) * 2.0)))

    def summary(self) -> list[str]:
        out = list(self.machine.describe())
        out.extend(self.notes)
        for p in self.machine.moving_parts:
            a = p.actuator
            gone = "  (DESTROYED)" if p.identity in self.state.absent_parts else ""
            if hasattr(a, "position_m"):
                out.append(f"   {p.identity.split('.')[-1]:18s} {a.position_m * 1000:6.0f} mm  "
                           f"{a.velocity_m_s * 1000:7.1f} mm/s  {a.force_n / 1000:7.1f} kN"
                           + ("  STALLED" if a.stalled else "") + gone)
            elif hasattr(a, "angle_deg"):
                out.append(f"   {p.identity.split('.')[-1]:18s} {a.angle_deg:6.1f} deg  "
                           f"{a.rate_deg_s:7.1f} deg/s  {a.torque_nm:7.0f} Nm" + gone)
        if self.demand_l_min > 0.0:
            out.append(f"   supply: wants {self.demand_l_min:6.1f} L/min, got "
                       f"{self.supply_flow_l_min:6.1f} at {self.supply_pressure_pa / 1e5:4.0f} bar"
                       + (f", short {self.made_up_by_bench_l_min:.1f}" if self.made_up_by_bench_l_min > 0.1 else ""))
        live = [e for e in self.hole_emitters.emitters if e.regime not in ("none", "fitted")]
        for em in live[:4]:
            out.append(f"   LEAK {em.part.split('.')[-1]}: {em.fluid} {em.regime} "
                       f"{em.mass_flow_kg_s * 1000:.0f} g/s [{em.character}]")
        return out


# =====================================================================
#  THE MACHINES
# =====================================================================
class MachineSystem(DtCompatibleEngine):
    """Managed-dt boundary for the complete mechanical machine simulation."""

    def __init__(self, sim: MachineSim):
        self.sim = sim
        self.world_time = 0.0
        self.observer_time = 0.0
        self.last_metrics = None

    def step(self, dt: float, state=None, state_table=None):
        self.sim.step(float(dt))
        channels = empty_channels()
        metrics = Metrics(
            max_vel=0.0, max_flux=float(self.sim.supply_flow_l_min),
            div_inf=0.0, mass_err=0.0,
            pub_exchange_time=AbstractTensor.tensor([0.0]),
            pub_exchange_time_present=AbstractTensor.tensor([0.0]),
            pub_contract=AbstractTensor.tensor([HOLD]),
            pub_dt_limit=AbstractTensor.tensor([0.0]),
            pub_dt_limit_present=AbstractTensor.tensor([0.0]),
            pub_values=channels.copy(), pub_present=AbstractTensor.zeros_like(channels),
            pub_limits=AbstractTensor.zeros_like(channels),
            pub_limits_present=AbstractTensor.zeros_like(channels),
            advanced_dt=float(dt),
        )
        self.last_metrics = metrics
        return True, metrics, state

    def get_state(self, state=None):
        return self.sim if state is None else state

    def snapshot(self):
        return (self.sim.snapshot(), float(self.world_time),
                float(self.observer_time), self.last_metrics)

    def restore(self, snapshot) -> None:
        sim, self.world_time, self.observer_time, self.last_metrics = snapshot
        self.sim.restore(sim)


MACHINES: dict[str, object] = {}


def _register(fn):
    MACHINES[fn.__name__.replace("_", "-")] = fn
    return fn


@_register
def hydraulic_power_pack() -> Machine:
    """The simplest real machine: a motor, a pump, a tank and a cooler.

    Worth having as a machine in its own right because it is what every
    other hydraulic machine is standing on, and because it fails in the
    ways that matter -- run it against a closed valve and the relief
    valve turns the whole motor's output into heat in the tank."""
    m = Machine(identity="hydraulic-power-pack", label="7.5 kW hydraulic power pack",
                power="self-contained",
                note="everything downstream of the relief valve is heat; that is the point of the cooler",
                hpu=HydraulicPowerUnit(identity="pack", motor_kw=7.5,
                                       pump_displacement_cc_rev=16.0, accumulator_l=2.5))
    m.parts = [
        MachinePart("pack.reservoir", "fluid-reservoir", (0.0, 0.0, 0.0), (0.32, 0.22, 0.24),
                    mass_kg=28.0, fluid="hydraulic-oil", fluid_volume_l=54.0, part_role="reservoir"),
        MachinePart("pack.motor", "electric-motor", (0.0, 0.38, 0.0), (0.16, 0.14, 0.14),
                    mass_kg=52.0, material="cast-iron", part_role="prime-mover"),
        MachinePart("pack.pump", "gear-pump", (0.26, 0.34, 0.0), (0.08, 0.07, 0.07),
                    mass_kg=9.0, material="cast-iron", fluid="hydraulic-oil",
                    fluid_volume_l=0.4, part_role="pump"),
        MachinePart("pack.relief_valve", "relief-valve", (0.26, 0.20, 0.12), (0.04, 0.05, 0.04),
                    mass_kg=1.4, part_role="relief"),
        MachinePart("pack.accumulator", "high-pressure-canister", (-0.30, 0.34, 0.10), (0.09, 0.20, 0.09),
                    mass_kg=12.0, material="pressed-steel", capacity_kg=2.2, part_role="accumulator"),
        MachinePart("pack.cooler", "fluid-cooler", (0.0, 0.30, -0.30), (0.20, 0.14, 0.04),
                    mass_kg=6.5, fluid="hydraulic-oil", fluid_volume_l=1.2, part_role="cooler"),
        MachinePart("pack.manifold", "manifold", (0.30, 0.10, 0.0), (0.06, 0.06, 0.10),
                    mass_kg=5.0, material="steel-plate", part_role="manifold"),
    ]
    m.lines = [
        MachineLine("pack.tank_to_pump", "pack.reservoir", "pack.pump", radius=0.010),
        MachineLine("pack.motor_to_pump", "pack.motor", "pack.pump", constraint="torque-shaft",
                    radius=0.016, circuit_identity="drive", material="steel-shaft"),
        MachineLine("pack.pump_to_manifold", "pack.pump", "pack.manifold", radius=0.008),
        MachineLine("pack.manifold_to_relief", "pack.manifold", "pack.relief_valve", radius=0.006),
        MachineLine("pack.relief_to_cooler", "pack.relief_valve", "pack.cooler", radius=0.008),
        MachineLine("pack.cooler_to_tank", "pack.cooler", "pack.reservoir", radius=0.010),
        MachineLine("pack.manifold_to_accumulator", "pack.manifold", "pack.accumulator", radius=0.008),
    ]
    return m


@_register
def scissor_lift() -> Machine:
    """A scissor lift, which is a lesson in mechanical advantage: the
    cylinder is nearly horizontal when the platform is down, so it has
    almost no vertical component and needs enormous force to start the
    lift -- and almost none to hold it near the top. That is why these
    are slow off the ground and quick at height, and why the cylinder is
    sized for the first inch rather than the last."""
    lift = LinearActuator(identity="lift-ram", bore_m=0.090, rod_m=0.050, stroke_m=0.900,
                          mounting="clevis-both-ends")
    m = Machine(identity="scissor-lift", label="3.2 m electric scissor lift",
                power="self-contained",
                note="the ram is nearly flat at the bottom of travel: that is where the force goes",
                hpu=HydraulicPowerUnit(identity="lift-pack", motor_kw=3.0,
                                       pump_displacement_cc_rev=8.0, relief_pressure_pa=18e6))
    m.parts = [
        MachinePart("lift.chassis", "frame", (0.0, 0.0, 0.0), (0.90, 0.12, 0.40),
                    mass_kg=310.0, part_role="frame"),
        MachinePart("lift.reservoir", "fluid-reservoir", (-0.60, 0.16, 0.28), (0.16, 0.12, 0.10),
                    mass_kg=14.0, fluid="hydraulic-oil", fluid_volume_l=12.0, part_role="reservoir"),
        MachinePart("lift.pack", "electric-motor", (-0.60, 0.16, -0.28), (0.14, 0.12, 0.12),
                    mass_kg=26.0, material="cast-iron", part_role="prime-mover"),
        MachinePart("lift.scissor_lower", "structural-arm", (0.0, 0.45, 0.0), (0.80, 0.05, 0.36),
                    mass_kg=95.0, part_role="structure"),
        MachinePart("lift.scissor_upper", "structural-arm", (0.0, 0.95, 0.0), (0.80, 0.05, 0.36),
                    mass_kg=95.0, part_role="structure"),
        MachinePart("lift.ram", "hydraulic-cylinder", (0.10, 0.60, 0.0), (0.45, 0.05, 0.05),
                    mass_kg=38.0, material="steel-plate", fluid="hydraulic-oil",
                    fluid_volume_l=6.4, part_role="actuator", actuator=lift,
                    duty_speed_m_s=0.06),      # a platform rises at a walking pace
        MachinePart("lift.platform", "platform", (0.0, 1.45, 0.0), (0.95, 0.05, 0.45),
                    mass_kg=180.0, part_role="platform"),
    ]
    m.linkages = [Linkage("lift.ram", "lift",
                          ("lift.platform", "lift.scissor_upper"))]
    m.lines = [
        MachineLine("lift.tank_to_pack", "lift.reservoir", "lift.pack", radius=0.008),
        MachineLine("lift.pack_to_ram", "lift.pack", "lift.ram", radius=0.006),
        MachineLine("lift.ram_to_tank", "lift.ram", "lift.reservoir", radius=0.006),
        MachineLine("lift.lower_to_upper", "lift.scissor_lower", "lift.scissor_upper",
                    constraint="pinned-joint", radius=0.020, circuit_identity="structure",
                    material="steel-shaft"),
        MachineLine("lift.upper_to_platform", "lift.scissor_upper", "lift.platform",
                    constraint="pinned-joint", radius=0.020, circuit_identity="structure",
                    material="steel-shaft"),
        MachineLine("lift.chassis_to_lower", "lift.chassis", "lift.scissor_lower",
                    constraint="pinned-joint", radius=0.020, circuit_identity="structure",
                    material="steel-shaft"),
    ]
    return m


@_register
def log_splitter() -> Machine:
    """A log splitter: one big cylinder, a wedge, and a two-stage pump.

    The two-stage pump is the interesting part and a genuinely clever
    piece of real engineering: it runs a big displacement at low
    pressure to move the ram quickly through the air, and when the wedge
    meets the log and pressure rises, it dumps the large stage and runs
    only the small one -- same motor power, ten times the force, a tenth
    of the speed, with no controls and no operator input at all."""
    ram = LinearActuator(identity="split-ram", bore_m=0.100, rod_m=0.050, stroke_m=0.600,
                         rated_pressure_pa=24e6)
    m = Machine(identity="log-splitter", label="25-ton log splitter",
                power="self-contained",
                note="two-stage pump: fast and weak until the wedge bites, then slow and strong",
                hpu=HydraulicPowerUnit(identity="splitter-pack", motor_kw=6.0,
                                       pump_displacement_cc_rev=22.0, relief_pressure_pa=24e6,
                                       second_stage_cc_rev=4.5,
                                       stage_switch_pressure_pa=4.5e6))
    m.parts = [
        MachinePart("split.beam", "frame", (0.0, 0.0, 0.0), (0.85, 0.09, 0.12),
                    mass_kg=120.0, part_role="frame"),
        MachinePart("split.reservoir", "fluid-reservoir", (-0.70, 0.22, 0.0), (0.16, 0.14, 0.14),
                    mass_kg=11.0, fluid="hydraulic-oil", fluid_volume_l=18.0, part_role="reservoir"),
        MachinePart("split.pump", "gear-pump", (-0.44, 0.20, 0.0), (0.08, 0.07, 0.07),
                    mass_kg=7.0, material="cast-iron", fluid="hydraulic-oil",
                    fluid_volume_l=0.3, part_role="pump"),
        MachinePart("split.ram", "hydraulic-cylinder", (0.10, 0.22, 0.0), (0.35, 0.055, 0.055),
                    mass_kg=42.0, material="steel-plate", fluid="hydraulic-oil",
                    fluid_volume_l=4.7, part_role="actuator", actuator=ram,
                    duty_speed_m_s=0.09),      # about a hand's width a second on the fast stage
        MachinePart("split.wedge", "wedge", (0.62, 0.26, 0.0), (0.05, 0.16, 0.02),
                    mass_kg=14.0, material="hardened-steel", part_role="tool"),
        MachinePart("split.control_valve", "spool-valve", (-0.20, 0.30, 0.10), (0.06, 0.05, 0.05),
                    mass_kg=3.2, part_role="valve"),
    ]
    m.lines = [
        MachineLine("split.tank_to_pump", "split.reservoir", "split.pump", radius=0.010),
        MachineLine("split.pump_to_valve", "split.pump", "split.control_valve", radius=0.008),
        MachineLine("split.valve_to_ram", "split.control_valve", "split.ram", radius=0.008),
        MachineLine("split.ram_to_tank", "split.ram", "split.reservoir", radius=0.008),
        MachineLine("split.ram_to_wedge", "split.ram", "split.wedge", constraint="rigid-mount",
                    radius=0.030, circuit_identity="structure", material="steel-plate"),
    ]
    return m


@_register
def pop_up_turret() -> Machine:
    """A hydraulic pop-up turret: stowed flush, raised to fire.

    Every part of this is a real mechanism doing the job it is actually
    good at, which is why it is worth building out of pieces already
    here rather than as one bespoke thing.

    THE HOIST is a long-stroke cylinder lifting the whole rotating mass
    out of its well, and it is the reason this machine needs an
    ACCUMULATOR rather than a bigger pump. A pop-up turret is only worth
    having if it is quick, and quick here means moving several hundred
    kilograms through most of a metre in about two seconds. That is a
    large flow for a short time and a small volume overall -- precisely
    the demand an accumulator answers, and precisely the demand a pump
    sized for it would be absurd for. The pump then refills the
    accumulator at its leisure, which is why a second pop-up in quick
    succession is slower than the first.

    THE TRAVERSE is a rotary actuator on the slew ring, not a cylinder:
    it has to turn continuously and hold anywhere, and a rack-and-pinion
    gives constant torque through the whole swing.

    THE ELEVATION is an ordinary cylinder working against a load that
    CHANGES SIGN as it passes the balance point -- the barrel's weight
    opposes it on one side and helps on the other. That is what the
    equilibrator is for: a gas spring carrying the barrel's moment so
    the actuator only has to handle the difference.

    THE RECOIL SYSTEM is an oleo-pneumatic strut, the same
    `GasOverOilStrut` as an aircraft landing gear, because it is the
    same device solving the same problem. Gas is the recuperator that
    returns the barrel to battery; oil forced through an orifice is the
    brake that absorbs the shot. Without it the entire recoil impulse
    arrives at the mount, the ring and the hoist ram in a couple of
    milliseconds, which no structure survives.
    """
    hoist = LinearActuator(identity="hoist-ram", bore_m=0.125, rod_m=0.070, stroke_m=0.850,
                           mounting="flange-rigid", cushion_m=0.040, rated_pressure_pa=25e6)
    traverse = RotaryActuator(identity="traverse", kind="rack-and-pinion", swing_deg=360.0,
                              piston_bore_m=0.080, pinion_radius_m=0.090, pistons=2,
                              rated_pressure_pa=25e6)
    elevation = LinearActuator(identity="elevation-ram", bore_m=0.063, rod_m=0.036, stroke_m=0.320,
                               mounting="clevis-both-ends", rated_pressure_pa=25e6)
    m = Machine(identity="pop-up-turret", label="hydraulic pop-up turret",
                power="self-contained", medium="hydraulic",
                note="the accumulator is what makes it pop; the pump only refills it between deployments",
                hpu=HydraulicPowerUnit(identity="turret-pack", motor_kw=11.0,
                                       pump_displacement_cc_rev=14.0, relief_pressure_pa=25e6,
                                       accumulator_l=20.0, accumulator_precharge_pa=12e6))
    m.parts = [
        MachinePart("turret.well", "structure", (0.0, -0.55, 0.0), (0.85, 0.55, 0.85),
                    mass_kg=900.0, material="armour-plate", part_role="structure"),
        MachinePart("turret.reservoir", "fluid-reservoir", (-0.62, -0.50, 0.55), (0.20, 0.18, 0.16),
                    mass_kg=24.0, fluid="hydraulic-oil", fluid_volume_l=40.0, part_role="reservoir"),
        MachinePart("turret.pack", "electric-motor", (-0.62, -0.50, -0.55), (0.16, 0.14, 0.14),
                    mass_kg=62.0, material="cast-iron", part_role="prime-mover"),
        MachinePart("turret.accumulator", "high-pressure-canister", (0.62, -0.45, 0.55),
                    (0.13, 0.42, 0.13), mass_kg=54.0, material="pressed-steel",
                    capacity_kg=17.4, part_role="accumulator"),
        MachinePart("turret.dump_valve", "spool-valve", (0.40, -0.30, 0.40), (0.07, 0.06, 0.06),
                    mass_kg=6.0, part_role="valve"),
        MachinePart("turret.hoist_ram", "hydraulic-cylinder", (0.0, -0.20, 0.0), (0.075, 0.42, 0.075),
                    mass_kg=86.0, material="steel-plate", fluid="hydraulic-oil",
                    fluid_volume_l=10.4, part_role="actuator", actuator=hoist,
                    duty_speed_m_s=0.42),
        MachinePart("turret.ring_gear", "slew-ring", (0.0, 0.30, 0.0), (0.52, 0.07, 0.52),
                    mass_kg=210.0, material="hardened-steel", part_role="bearing"),
        MachinePart("turret.traverse_actuator", "rotary-actuator", (0.34, 0.30, 0.34),
                    (0.12, 0.09, 0.12), mass_kg=48.0, fluid="hydraulic-oil", fluid_volume_l=1.8,
                    part_role="actuator", actuator=traverse, duty_rate_deg_s=60.0),
        MachinePart("turret.shell", "armour-shell", (0.0, 0.62, 0.0), (0.50, 0.28, 0.50),
                    mass_kg=430.0, material="armour-plate", part_role="structure"),
        MachinePart("turret.elevation_ram", "hydraulic-cylinder", (0.16, 0.68, 0.0),
                    (0.20, 0.04, 0.04), mass_kg=16.0, fluid="hydraulic-oil", fluid_volume_l=1.1,
                    part_role="actuator", actuator=elevation, duty_speed_m_s=0.10),
        MachinePart("turret.equilibrator", "gas-spring", (-0.16, 0.68, 0.0), (0.18, 0.04, 0.04),
                    mass_kg=9.0, part_role="equilibrator"),
        MachinePart("turret.cradle", "cradle", (0.0, 0.72, 0.30), (0.10, 0.08, 0.32),
                    mass_kg=70.0, material="steel-plate", part_role="structure"),
        MachinePart("turret.recoil_brake", "oleo-strut", (0.09, 0.72, 0.34), (0.05, 0.05, 0.26),
                    mass_kg=22.0, fluid="hydraulic-oil", fluid_volume_l=1.6, part_role="recoil"),
        MachinePart("turret.recuperator", "gas-cylinder", (-0.09, 0.72, 0.34), (0.05, 0.05, 0.26),
                    mass_kg=18.0, part_role="recoil"),
        MachinePart("turret.barrel", "gun-barrel", (0.0, 0.74, 0.95), (0.045, 0.045, 0.60),
                    mass_kg=145.0, material="gun-steel", part_role="weapon"),
        MachinePart("turret.hatch", "hatch", (0.0, 0.02, 0.62), (0.45, 0.04, 0.20),
                    mass_kg=55.0, material="armour-plate", part_role="closure"),
    ]
    m.lines = [
        MachineLine("turret.tank_to_pack", "turret.reservoir", "turret.pack", radius=0.012),
        MachineLine("turret.pack_to_accumulator", "turret.pack", "turret.accumulator", radius=0.010),
        MachineLine("turret.accumulator_to_dump", "turret.accumulator", "turret.dump_valve", radius=0.014),
        MachineLine("turret.dump_to_hoist", "turret.dump_valve", "turret.hoist_ram", radius=0.014),
        MachineLine("turret.hoist_to_ring", "turret.hoist_ram", "turret.ring_gear",
                    constraint="rigid-mount", radius=0.040, circuit_identity="structure",
                    material="steel-plate"),
        MachineLine("turret.ring_to_shell", "turret.ring_gear", "turret.shell",
                    constraint="rigid-mount", radius=0.040, circuit_identity="structure",
                    material="armour-plate"),
        MachineLine("turret.pack_to_traverse", "turret.pack", "turret.traverse_actuator", radius=0.008),
        MachineLine("turret.traverse_to_ring", "turret.traverse_actuator", "turret.ring_gear",
                    constraint="geared-drive", radius=0.030, circuit_identity="structure",
                    material="hardened-steel"),
        MachineLine("turret.pack_to_elevation", "turret.pack", "turret.elevation_ram", radius=0.006),
        MachineLine("turret.elevation_to_cradle", "turret.elevation_ram", "turret.cradle",
                    constraint="pinned-joint", radius=0.022, circuit_identity="structure",
                    material="steel-shaft"),
        MachineLine("turret.equilibrator_to_cradle", "turret.equilibrator", "turret.cradle",
                    constraint="pinned-joint", radius=0.018, circuit_identity="structure",
                    material="steel-shaft"),
        MachineLine("turret.cradle_to_brake", "turret.cradle", "turret.recoil_brake",
                    constraint="rigid-mount", radius=0.026, circuit_identity="structure",
                    material="steel-plate"),
        MachineLine("turret.cradle_to_recuperator", "turret.cradle", "turret.recuperator",
                    constraint="rigid-mount", radius=0.026, circuit_identity="structure",
                    material="steel-plate"),
        MachineLine("turret.brake_to_barrel", "turret.recoil_brake", "turret.barrel",
                    constraint="sliding-joint", radius=0.030, circuit_identity="structure",
                    material="gun-steel"),
        MachineLine("turret.hatch_to_well", "turret.hatch", "turret.well",
                    constraint="pinned-joint", radius=0.016, circuit_identity="structure",
                    material="steel-shaft"),
    ]
    # the gun it mounts, and a recoil system sized to THAT gun rather
    # than to a round number: barrel plus cradle plus the recoiling
    # parts of the brake and recuperator are what actually slides
    # WHAT EACH ACTUATOR CARRIES. The hoist lifts the entire rotating
    # assembly out of the well; the traverse turns everything above the
    # ring about the machine's axis; the elevation ram pitches the gun
    # and the recoil system with it, about the trunnion in the cradle.
    _rides_up = ("turret.ring_gear", "turret.traverse_actuator", "turret.shell",
                 "turret.elevation_ram", "turret.equilibrator", "turret.cradle",
                 "turret.recoil_brake", "turret.recuperator", "turret.barrel")
    _turns = ("turret.shell", "turret.elevation_ram", "turret.equilibrator",
              "turret.cradle", "turret.recoil_brake", "turret.recuperator", "turret.barrel")
    _pitches = ("turret.barrel", "turret.recoil_brake", "turret.recuperator")
    m.linkages = [
        Linkage("turret.hoist_ram", "lift", _rides_up),
        Linkage("turret.traverse_actuator", "rotate-y", _turns, pivot=(0.0, 0.0, 0.0)),
        Linkage("turret.elevation_ram", "pitch", _pitches,
                pivot=(0.0, 0.72, 0.30), angle_range_deg=(-8.0, 42.0)),
    ]
    # THE SIGHT AND THE FIRE-CONTROL COMPUTER. A gun this size is not
    # aimed by eye: it is aimed by a computer that knows the trajectory,
    # and the mount then has to physically get there. Declaring the
    # cartridge and the zero here is what binds the turret to the baked
    # ballistics (fire_control.SolutionTable) -- the same table the
    # player's scoped rifle uses, for the same reason.
    m.sight_cartridge = "40mm l70"
    m.sight_zero_range_m = 500.0
    m.weapon = WEAPONS["40mm-l70"]
    m.recoiling_mass_kg = 145.0 + 70.0 + 22.0 + 18.0
    m.recoil = size_recoil_for(m.weapon, m.recoiling_mass_kg, stroke_m=0.320)
    m.rotating_mass_kg = sum(p.mass_kg for p in m.parts
                             if p.part_role in ("structure", "weapon", "actuator", "bearing", "recoil")
                             and p.identity not in ("turret.well",))
    return m



@_register
def gimbal_cannon_station() -> Machine:
    """THE canonical turret: one structure, authored once.

    This is the reconciliation of the two turrets this project carried
    for a while. `pop_up_turret` below describes a mount as a list of
    boxes with a kinematic stand-in moving them; this one takes its
    entire structure from `turret_production.build_gimbal_cannon_station`
    -- every body, every member, the real load paths, the declared
    motion groups -- and adds only the plant that drives it.

    Three mechanisms in series, and a fourth aiming on its own:

      THE RING carries the whole mount round, driven by pinions that are
      real bodies in a real mesh rather than a torque appearing between
      two hubs. The pinions are CAPTIVE: a second ring laid over the
      first holds them in mesh from both sides, which cancels the
      separating force and straddles their shafts
      (`slew_drive.CaptivePinionRing`). The same call with the upper
      ring grounded to the frame instead makes it a differential
      planetary, which is why it is a parameter and not a fact.

      THE TRUNNION is at the barrel's own centre of gravity, so the
      elevation ram carries the imbalance rather than the barrel.

      THE SIX-LEG PLATFORM at the back of the tube is the fine stage: a
      Stewart platform, twenty-five millimetres of travel per leg,
      which is a fraction of a degree of aim and the last of the
      pointing error.

      THE MACHINE GUN has its own yaw and its own trunnion on the same
      carriage, and a hemisphere and a bit of its own to aim in, so it
      can engage while the cannon is busy or parked.

    The plant below is the only thing this layer adds, because it is the
    only part that is state rather than structure.
    """
    import turret_production as tp

    graph = tp.build_gimbal_cannon_station(bore_mm=40.0).as_document()
    hoist = LinearActuator(identity="hoist-ram", bore_m=0.125, rod_m=0.070,
                           stroke_m=0.850, mounting="flange-rigid", cushion_m=0.040,
                           rated_pressure_pa=25e6)
    traverse = RotaryActuator(identity="traverse", kind="rack-and-pinion", swing_deg=360.0,
                              piston_bore_m=0.080, pinion_radius_m=0.090, pistons=2,
                              rated_pressure_pa=25e6)
    elevation = LinearActuator(identity="elevation-ram", bore_m=0.063, rod_m=0.036,
                               stroke_m=0.320, mounting="clevis-both-ends",
                               rated_pressure_pa=25e6)
    m = Machine(identity="gimbal-cannon-station",
                label="40 mm gimbal cannon station",
                power="self-contained", medium="hydraulic",
                production_graph=graph,
                note="three mechanisms in series on one gun, plus a machine gun "
                     "that aims for itself",
                hpu=HydraulicPowerUnit(identity="station-pack", motor_kw=11.0,
                                       pump_displacement_cc_rev=14.0,
                                       relief_pressure_pa=25e6, accumulator_l=20.0,
                                       accumulator_precharge_pa=12e6))
    m.parts = [
        MachinePart("plant.reservoir", "fluid-reservoir", (-0.62, -0.50, 0.55),
                    (0.20, 0.18, 0.16), mass_kg=24.0, fluid="hydraulic-oil",
                    fluid_volume_l=40.0, part_role="reservoir"),
        MachinePart("plant.pack", "electric-motor", (-0.62, -0.50, -0.55),
                    (0.16, 0.14, 0.14), mass_kg=62.0, material="cast-iron",
                    part_role="prime-mover"),
        MachinePart("plant.accumulator", "high-pressure-canister", (0.62, -0.45, 0.55),
                    (0.13, 0.42, 0.13), mass_kg=54.0, material="pressed-steel",
                    capacity_kg=17.4, part_role="accumulator"),
        MachinePart("plant.dump_valve", "spool-valve", (0.40, -0.30, 0.40),
                    (0.07, 0.06, 0.06), mass_kg=6.0, part_role="valve"),
        MachinePart("plant.hoist_ram", "hydraulic-cylinder", (0.0, -0.20, 0.0),
                    (0.075, 0.42, 0.075), mass_kg=86.0, material="steel-plate",
                    fluid="hydraulic-oil", fluid_volume_l=10.4, part_role="actuator",
                    actuator=hoist, duty_speed_m_s=0.42),
        MachinePart("plant.traverse_actuator", "rotary-actuator", (0.34, -0.30, 0.34),
                    (0.12, 0.09, 0.12), mass_kg=48.0, fluid="hydraulic-oil",
                    fluid_volume_l=1.8, part_role="actuator", actuator=traverse,
                    duty_rate_deg_s=60.0),
        MachinePart("plant.elevation_ram", "hydraulic-cylinder", (0.16, -0.30, 0.0),
                    (0.20, 0.04, 0.04), mass_kg=16.0, fluid="hydraulic-oil",
                    fluid_volume_l=1.1, part_role="actuator", actuator=elevation,
                    duty_speed_m_s=0.10),
    ]
    m.lines = [
        MachineLine("plant.tank_to_pack", "plant.reservoir", "plant.pack", radius=0.012),
        MachineLine("plant.pack_to_accumulator", "plant.pack", "plant.accumulator",
                    radius=0.010),
        MachineLine("plant.accumulator_to_dump", "plant.accumulator", "plant.dump_valve",
                    radius=0.014),
        MachineLine("plant.dump_to_hoist", "plant.dump_valve", "plant.hoist_ram",
                    radius=0.014),
        MachineLine("plant.pack_to_traverse", "plant.pack", "plant.traverse_actuator",
                    radius=0.008),
        MachineLine("plant.pack_to_elevation", "plant.pack", "plant.elevation_ram",
                    radius=0.006),
    ]
    # THE STRUCTURE ALREADY SAYS WHAT MOVES. The linkage stand-in is not
    # used here at all: the mount is posed through
    # `turret_production.pose_frames`, off the declared motion groups,
    # and the mesh follows through `articulation.ArticulatedMesh`. That
    # is the difference between geometry that moves because a rule says
    # it should look as though it did, and geometry that moves because
    # the structure it belongs to says so.
    m.sight_cartridge = "40mm l70"
    m.sight_zero_range_m = 500.0
    m.weapon = WEAPONS["40mm-l70"]
    recoiling = next(n for n in graph["nodes"] if n["identity"] == "turret.weapon")
    slide = next(e for e in graph["edges"] if e["identity"] == "turret.recoil_slide")
    m.recoiling_mass_kg = float(slide.get("recoiling_mass_kg", recoiling["mass_kg"]))
    m.recoil = size_recoil_for(m.weapon, m.recoiling_mass_kg,
                               stroke_m=float(slide.get("recoil_stroke_m", 0.32)))
    m.rotating_mass_kg = sum(
        float(n.get("mass_kg", 0.0)) for n in graph["nodes"]
        if n.get("motion_group") not in ("frame", None))
    return m


def get(identity: str) -> Machine:
    key = identity.replace("_", "-")
    if key not in MACHINES:
        raise KeyError(f"unknown machine {identity!r}; known: {', '.join(sorted(MACHINES))}")
    return MACHINES[key]()


def roster() -> list[str]:
    return sorted(MACHINES)


def fire_control_for(machine, muzzle_position=None, ray_mesh=None):
    """Build the fire-control unit for a machine that carries a sight.

    The slew rates come from the machine's OWN actuators rather than
    from a number typed here, so a turret on a small pump slews slowly
    and the fire control knows it. That is the honest coupling: the
    computer's answer arrives in milliseconds and the mount still takes
    seconds, and the two reticles show exactly that gap.
    """
    from fire_control import SolutionTable, TurretFireControl
    if not getattr(machine, "sight_cartridge", ""):
        raise ValueError(f"{machine.identity} carries no sight")
    table = SolutionTable.load_or_bake(machine.sight_cartridge,
                                       zero_range_m=machine.sight_zero_range_m)
    traverse = next((p.duty_rate_deg_s for p in machine.moving_parts
                     if hasattr(p.actuator, "angle_deg")), 45.0)
    # the elevation ram's real speed, expressed as an angular rate over
    # its declared travel
    elevation_rate = 12.0
    for part in machine.moving_parts:
        if "elevation" in part.identity:
            a = part.actuator
            span = 50.0          # the declared elevation range, degrees
            elevation_rate = span * part.duty_speed_m_s / max(a.stroke_m, 1e-6)
            break
    if muzzle_position is None:
        barrel = next((p for p in machine.parts if p.part_role == "weapon"), None)
        muzzle_position = barrel.position if barrel else (0.0, 0.0, 0.0)
    return TurretFireControl(
        table=table, muzzle_position=__import__("numpy").asarray(muzzle_position, dtype=float),
        ray_mesh=ray_mesh, traverse_rate_deg_s=float(traverse),
        elevation_rate_deg_s=float(elevation_rate))
