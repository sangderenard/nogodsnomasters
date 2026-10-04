"""Orbital craft, build step 8: the craft as an engine_toy MACHINE.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``;
continuation: ``turing/docs/concordance_census/
CONTINUATION_orbital_craft_machine.md``.

The craft is authored in the production vocabulary
(``turret_production.ProductionGraph``) and loaded as a
``machines.Machine`` whose ``production_graph`` IS the structure, exactly
as the gimbal cannon station is.  Everything about the craft is read back
from that one document:

    structure     three 6061 rings (fore / mid / aft) of four load nodes,
                  longerons, ring members and one diagonal per bay face; a
                  docking ring declared ``port_role="structural-mount"``
                  (the craft's mount: where it is held when docked or
                  installed); a thrust structure across the aft ring
    tanks         drum nodes (``part_role="propellant-tank"``) declaring
                  ``fluid`` (a ``fluids`` row), ``capacity_kg`` and
                  ``fill_kg``; their node mass is the SHELL, the contents
                  are charges (``operating_states.OperatingState``)
    thrusters     nodes declaring ``part_role="thruster"`` and a
                  ``thruster`` record (role, max thrust, kind, mount point,
                  nominal axis, cone, gimbal axis and slew, throttle slew,
                  deadband); RCS and retro nozzles are condensed into the
                  body that carries them (``solver_condensed_into``)
    gimbal        the main engine hangs from the thrust structure on a
                  ``universal-joint`` (the Cardan gimbal) and two
                  ``linear-hydraulic-actuator`` TVC rams from the aft ring
    fuel network  routed ``fuel-line`` edges from each tank to each
                  thruster it feeds, carrying the propellant circuit and the
                  thruster's ``feed_share`` (its mixture ratio); a
                  thruster's feeds ARE its incident fuel lines
    wheels        three ``electric-motor`` nodes (``part_role=
                  "reaction-wheel"``) on orthogonal craft axes, each
                  declaring its ROTOR in the operating_states vocabulary
                  (``rotor_axis``, ``rotor_rated_rpm`` = its top speed,
                  ``rotor_inertia_kg_m2``, balance, ``runs_in``) and its
                  motor (``motor_max_torque_n_m``,
                  ``motor_torque_constant_n_m_a``,
                  ``motor_winding_resistance_ohm``) and momentum dumping
                  (``momentum_dump_fraction``, ``momentum_dump_time_s``)

Mass properties are THE machine reduction, ``machine_package.
mass_properties`` (point masses with parallel axis plus each body's own
declared shape), on the nodes charged with the tanks' contents
(``mode_table._charged_nodes``).  It gives a full tensor about the centre
of mass; the craft is deliberately NOT symmetric (an equipment bay on -y,
the RCS tank on +z), so the tensor has products of inertia and the centre
of mass is off the main engine's axis -- the main engine must gimbal to
push through it, and the trim changes as the tanks drain.  The dynamics
therefore use the full tensor (``eq_N1_6`` as a matrix law).

In the dt pieces the reduction is a law of the tank columns: the reduction
is linear in each charge, so the host evaluates it once for the dry craft
and once per kilogram of each tank's contents, and the piece sums
``M = M_dry + sum m_t``, ``S = S_dry + sum m_t p_t``, ``c = S / M``,
``J = J_dry + sum m_t J_t`` (second moments about the machine origin) and
``I = J - (|S|^2 1 - S S^T) / M`` (``eq_N10_1``'s parallel-axis shift to
``c``).  ``test_orbital_craft_machine`` checks the piece against the
reduction at a partial fill.

The dt graph starts with those declared mass-property reductions, then
manifests the actual dt-system RK4Integrator.step call.  Its callbacks share
one coupled state: position, momentum, attitude, body angular velocity,
rotor momenta, tanks, impulse and work integrals.  Every callback evaluates
the authored bounded actuator ramp at that stage's elapsed time and reduces
the tank charges to that stage's full mass/COM/inertia.  The physical state
commits once after the library stages; all scratch and physical columns
belong to the existing PieceState transaction.

Separate translation, body and rotor stores publish energy and power.
Open-system energy and inertial-angular-momentum defects use work/impulse
from the same library quadrature.  Incremental Gram error and wheel speed
excess are independent declared dt error channels.  A failed attempt is
restored and refined by the existing controller.

The seam is the jumper's (``r()``, ``F()``, ``throttle(u)``) plus
``gimbal(angles)``, ``allocate(force, torque)`` and the state readings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

import honorary_engine_equation_catalogue as honorary
from honorary_engine_equation_catalogue import equation_piece
from orbital_green import craft_coast_equation_sets
from machine_package import mass_properties
from machines import Machine
from mode_table import _charged_nodes
from operating_states import OperatingState, source_of
from turret_production import ProductionGraph
from wrench_paths import wrench_paths

from orbital_actuation import (
    AXES,
    CENTRE_OF_MASS,
    DT,
    PROPELLANT_FLOW,
    THRUSTER_KINDS,
    Allocation,
    ReactionWheel,
    Thruster,
    allocate_wrench,
    attitude_symbols,
    delivered_throttle_rhs,
    delivered_throttles,
    feed_symbol,
    gimbal_state_rhs,
    machine_force_rhs,
    machine_thruster_symbols,
    machine_thrust,
    machine_thruster_columns,
    machine_torque_rhs,
    tank_flow_rhs,
    tank_supply_rhs,
    tank_symbols,
    throttle_state_rhs,
    thruster_symbols,
    wheel_columns,
    wheel_power_rhs,
    wheel_rate_rhs,
    wheel_relative_speed,
    wheel_spin_inertia,
    wheel_symbols,
    wheel_torque_rhs,
)
from orbital_jumper import (
    OrbitalJumper,
    attitude_rate_rhs,
    declare_binding,
    integrated_state_equations,
    mechanical_endpoint_metrics,
    energy_metric_equations,
    gravity_force_rhs,
    thrust_cost_integrand,
    variable_mass_momentum_rate,
)
from orbital_guidance import MachineGuidancePiece
from src.common.dt_system.dt import SuperstepPlan
from src.common.dt_system.dt_controller import STController
from src.common.dt_system.dt_graph import ControllerNode, RoundNode

#: The propellant-tank role and the thruster role, as declared on nodes.
TANK_ROLE = "propellant-tank"
THRUSTER_ROLE = "thruster"
#: The routed member that carries propellant from a tank to a thruster.
FEED_LINE = "fuel-line"
#: The reaction-wheel role, as declared on nodes.
WHEEL_ROLE = "reaction-wheel"
#: The six independent components of a symmetric tensor, in this order.
PAIRS = ("xx", "yy", "zz", "xy", "xz", "yz")


@dataclass(frozen=True)
class PropellantTank:
    """A tank as its node declares it."""

    identity: str
    fluid: str
    position_m: tuple[float, float, float]
    capacity_kg: float
    fill_kg: float


def _pair_index(pair: str) -> tuple[int, int]:
    return AXES.index(pair[0]), AXES.index(pair[1])


# ===================================================================
# THE DOCUMENT
# ===================================================================
class CraftMachine:
    """A craft: one production document, read back for everything.

    Also the ``design`` the jumper reads (``thrusters``,
    ``thruster_count``, ``mass_kg``, ``dry_mass_kg``, ``propellant_kg``,
    ``identity``, ``body_size_m``), so the dt glue is the jumper's own."""

    def __init__(self, graph: ProductionGraph, *, identity: str,
                 label: str = ""):
        self.graph = graph
        self.identity = identity
        self.machine = Machine(identity=identity, label=label or identity,
                               power="self-contained", medium="propellant",
                               production_graph=graph.as_document())
        self.document = self.machine.build_graph()
        nodes = {n["identity"]: n for n in self.document["nodes"]}
        self.tanks = tuple(
            PropellantTank(identity=n["identity"], fluid=n["fluid"],
                           position_m=tuple(n["reference_position"]),
                           capacity_kg=float(n["capacity_kg"]),
                           fill_kg=float(n["fill_kg"]))
            for n in self.document["nodes"]
            if n.get("part_role") == TANK_ROLE)
        self.tank_index = {tank.identity: index
                           for index, tank in enumerate(self.tanks)}
        feeds: dict = {}
        for edge in self.document["edges"]:
            if edge["constraint"] != FEED_LINE:
                continue
            ends = (edge["a"], edge["b"])
            tank = next(end for end in ends if end in self.tank_index)
            thruster = next(end for end in ends if end != tank)
            feeds.setdefault(thruster, []).append(
                (tank, float(edge["feed_share"])))
        thrusters = []
        for node in self.document["nodes"]:
            if node.get("part_role") != THRUSTER_ROLE:
                continue
            record = dict(node["thruster"])
            thrusters.append(Thruster(
                identity=node["identity"],
                feeds=tuple(feeds.get(node["identity"], ())), **record))
        self.thrusters = tuple(thrusters)
        wheels = []
        for node in self.document["nodes"]:
            if node.get("part_role") != WHEEL_ROLE:
                continue
            rotor = source_of(node)
            if rotor is None:
                raise ValueError(f"{node['identity']}: a reaction wheel "
                                 "declares its rotor (rotor_axis, ...)")
            wheels.append(ReactionWheel(
                identity=node["identity"], axis=rotor.axis,
                rotor_inertia_kg_m2=rotor.inertia_kg_m2,
                max_speed_rad_s=rotor.rated_rpm * 2.0 * math.pi / 60.0,
                max_torque_n_m=float(node["motor_max_torque_n_m"]),
                torque_constant_n_m_a=float(
                    node["motor_torque_constant_n_m_a"]),
                winding_resistance_ohm=float(
                    node["motor_winding_resistance_ohm"]),
                dump_fraction=float(node["momentum_dump_fraction"]),
                dump_time_s=float(node["momentum_dump_time_s"]),
                mass_kg=rotor.mass_kg,
                position_m=tuple(float(v) for v in rotor.position)))
        self.wheels = tuple(wheels)
        self.nodes = nodes

    # ---------------------------------------------------- the design view
    @property
    def thruster_count(self) -> int:
        return len(self.thrusters)

    @property
    def propellant_kg(self) -> float:
        return float(sum(tank.fill_kg for tank in self.tanks))

    @property
    def dry_mass_kg(self) -> float:
        return self.mass_properties({}).total_mass_kg

    @property
    def mass_kg(self) -> float:
        return self.mass_properties().total_mass_kg

    @property
    def body_size_m(self) -> tuple:
        """The prism of the craft's own metal (machine_package's)."""
        from machine_package import measure_prism
        return measure_prism(self.document["nodes"]).size_m

    def thrusters_by_role(self, role: str) -> list[int]:
        return [k for k, t in enumerate(self.thrusters) if t.role == role]

    # -------------------------------------------------------- the checks
    def check(self) -> list[str]:
        """``ProductionGraph.check()`` on the authored structure."""
        return self.graph.check()

    def load_paths(self) -> dict:
        """``wrench_paths`` from every massive body to the declared mount."""
        return wrench_paths(self.document)

    # ----------------------------------------------- the machine reduction
    def charged_nodes(self, tank_kg: dict | None = None) -> list:
        """The document's nodes with each tank's contents as its charge
        (``tank_kg`` by identity; default: the declared fill)."""
        charges = ({tank.identity: tank.fill_kg for tank in self.tanks}
                   if tank_kg is None else dict(tank_kg))
        state = OperatingState(name="propellant", charges_kg=charges)
        return _charged_nodes(self.document["nodes"], state)

    def mass_properties(self, tank_kg: dict | None = None):
        """THE reduction (``machine_package.mass_properties``) at a fill."""
        return mass_properties(self.charged_nodes(tank_kg))

    def reduction_coefficients(self) -> dict:
        """The reduction as a linear function of the tank contents:
        ``dry_mass``, ``dry_moment`` (3), ``dry_second`` (3x3, about the
        machine origin), and per tank ``position`` and ``unit_second``
        (3x3 per kilogram).  Each is the reduction itself, evaluated on the
        dry craft or on one kilogram of contents."""
        dry = self.mass_properties({})
        mass = dry.total_mass_kg
        centre = np.asarray(dry.center_of_gravity, dtype=float)
        second = (np.asarray(dry.inertia_tensor_kg_m2, dtype=float)
                  + mass * (float(centre @ centre) * np.eye(3)
                            - np.outer(centre, centre)))
        tanks = []
        for tank in self.tanks:
            node = dict(self.nodes[tank.identity])
            node["mass_kg"] = 1.0
            unit = mass_properties([node])
            p = np.asarray(unit.center_of_gravity, dtype=float)
            tanks.append({"position": p,
                          "unit_second": (np.asarray(
                              unit.inertia_tensor_kg_m2, dtype=float)
                              + float(p @ p) * np.eye(3) - np.outer(p, p))})
        return {"dry_mass": mass, "dry_moment": mass * centre,
                "dry_second": second, "tanks": tanks}

    def spin_free_inertia(self, tank_kg: dict | None = None) -> np.ndarray:
        """The tensor the attitude law carries: the reduction's without
        the wheels' axial rotor inertia, ``I' = I - sum I_w a a^T`` (the
        rotors' spin is their own momentum)."""
        return (np.asarray(self.mass_properties(tank_kg).inertia_tensor_kg_m2,
                           dtype=float) - wheel_spin_inertia(self.wheels))

    def principal_inertia(self, tank_kg: dict | None = None) -> np.ndarray:
        """The diagonal of the reduction's tensor (for readers that only
        take a diagonal; the dynamics use the full tensor)."""
        return np.diag(np.asarray(
            self.mass_properties(tank_kg).inertia_tensor_kg_m2)).copy()


# ===================================================================
# THE CRAFT, AUTHORED
# ===================================================================
def _ring(g, name, x, radius, mass_kg):
    """Four load nodes at +y, +z, -y, -z."""
    ids = []
    for label, (y, z) in (("py", (radius, 0.0)), ("pz", (0.0, radius)),
                          ("ny", (-radius, 0.0)), ("nz", (0.0, -radius))):
        identity = f"hull.{name}.{label}"
        g.node(identity, (x, y, z), "chassis-load-node", mass_kg=mass_kg,
               material="aluminium-plate", half_extent_m=(0.05, 0.05, 0.05),
               note="a ring fitting carrying its share of frame and skin")
        ids.append(identity)
    return ids


def _member(g, identity, a, b, radius=0.022):
    g.edge(identity, a, b, "rigid-distance", radius=radius, alloy="6061t6")


def orbital_craft(*, identity: str = "orbital-craft") -> CraftMachine:
    """The step-8 craft: a storable-bipropellant service vehicle.

    Craft frame: +x forward (the main engine pushes +x), the machine origin
    on the axis at the mid ring.  Numbers are of the real hardware class:
    main engine 4 kN (R-40B class), retro 2 x 400 N (R-4D class), RCS
    16 x 22 N hydrazine (MR-106 class); MMH/NTO at mixture ratio 1.65.
    """
    g = ProductionGraph(identity)
    ring_r = 0.85
    g.assembly = "hull"
    fore = _ring(g, "fore", 1.2, ring_r, 9.0)
    mid = _ring(g, "mid", 0.2, ring_r, 9.0)
    aft = _ring(g, "aft", -0.8, ring_r, 9.0)
    for name, ring in (("fore", fore), ("mid", mid), ("aft", aft)):
        for i in range(4):
            _member(g, f"hull.{name}.ring.{i}", ring[i], ring[(i + 1) % 4])
    for bay, (a_ring, b_ring) in (("fwd", (fore, mid)), ("aft", (mid, aft))):
        for i in range(4):
            _member(g, f"hull.{bay}.longeron.{i}", a_ring[i], b_ring[i],
                    radius=0.028)
            _member(g, f"hull.{bay}.diagonal.{i}", a_ring[i],
                    b_ring[(i + 1) % 4])
    g.node("hull.docking_ring", (1.45, 0.0, 0.0), "load-bearing-structure",
           mass_kg=22.0, material="aluminium-plate", shape="ring",
           ring_outer_radius_m=0.40, ring_inner_radius_m=0.32,
           ring_thickness_m=0.06, half_extent_m=(0.03, 0.40, 0.40),
           port_role="structural-mount", outward=(1.0, 0.0, 0.0),
           joint="docking-latch")
    for i, node in enumerate(fore):
        _member(g, f"hull.docking.strut.{i}", "hull.docking_ring", node,
                radius=0.025)
    g.node("hull.thrust_structure", (-0.95, 0.0, 0.0),
           "load-bearing-structure", mass_kg=18.0, material="aluminium-plate",
           half_extent_m=(0.06, 0.30, 0.30))
    for i, node in enumerate(aft):
        _member(g, f"hull.thrust_structure.beam.{i}",
                "hull.thrust_structure", node, radius=0.032)

    # ---- the tanks: drums along x, struts to the rings ----
    g.assembly = "propulsion"
    # 880 kg at the engine's 1.65 oxidizer/fuel mixture ratio gives the
    # declared farthest transfer about ten percent ideal delta-v margin;
    # attitude control keeps its separate hydrazine reserve below.
    tanks = (
        ("tank.mmh", "monomethylhydrazine", (0.60, 0.0, 0.0), 0.42, 0.80,
         15.0, 340.0, 880.0 / 2.65, mid + fore[:1]),
        ("tank.nto", "nitrogen-tetroxide", (-0.25, 0.0, 0.0), 0.42, 0.80,
         15.0, 560.0, 880.0 * 1.65 / 2.65, mid + aft[:1]),
        ("tank.hydrazine", "hydrazine", (0.15, 0.0, 0.60), 0.17, 0.70,
         5.0, 62.0, 62.0, (mid[1], fore[1], aft[1])),
    )
    for ident, fluid, at, radius, length, shell, capacity, fill, holds in tanks:
        g.node(ident, at, "high-pressure-canister", mass_kg=shell,
               material="titanium-plate", shape="drum",
               drum_axis=(1.0, 0.0, 0.0), drum_radius_m=radius,
               drum_length_m=length,
               half_extent_m=(length / 2, radius, radius),
               part_role=TANK_ROLE, fluid=fluid, capacity_kg=capacity,
               fill_kg=fill, supply_kind="pressure-fed-tank")
        for i, node in enumerate(holds):
            _member(g, f"{ident}.strut.{i}", ident, node, radius=0.018)

    # ---- the equipment bay: avionics and batteries, on -y ----
    g.assembly = "avionics"
    g.node("bay.equipment", (0.55, -0.60, 0.0), "load-bearing-structure",
           mass_kg=40.0, material="aluminium-plate",
           half_extent_m=(0.15, 0.12, 0.20))
    for i, node in enumerate((fore[2], mid[2], mid[1])):
        _member(g, f"bay.equipment.strut.{i}", "bay.equipment", node,
                radius=0.018)

    # ---- attitude control: three reaction wheels on the craft axes ----
    # Large-wheel class: 0.2 kg m^2 rotor at 6000 rpm (125.7 N m s), 1 N m
    # motor (the Hubble RWA's 0.82 N m is the precedent for the torque),
    # 12 kg each with motor and housing.  Three orthogonal wheels, not a
    # four-wheel pyramid: the craft has three torque axes to hold and no
    # fault to survive in this model, and three orthogonal wheels make the
    # allocation an exact per-axis box (a pyramid's fourth wheel adds a
    # null space whose spin must then be chosen, and buys only fault
    # tolerance and ~15 % more envelope on the diagonals).  They sit in the
    # (+y, -z) quadrant of the hull, across from the equipment bay (-y) and
    # the hydrazine tank (+z).
    g.assembly = "attitude-control"
    for label, axis, at, holds in (
            ("x", (1.0, 0.0, 0.0), (-0.30, 0.45, -0.45),
             (mid[0], mid[3], aft[0])),
            ("y", (0.0, 1.0, 0.0), (0.20, 0.45, -0.45),
             (mid[0], mid[3], fore[3])),
            ("z", (0.0, 0.0, 1.0), (0.70, 0.45, -0.45),
             (fore[0], fore[3], mid[0]))):
        ident = f"wheel.{label}"
        g.node(ident, at, "electric-motor", mass_kg=12.0,
               material="aluminium-plate", shape="drum", drum_axis=axis,
               drum_radius_m=0.19, drum_length_m=0.12,
               half_extent_m=tuple(0.06 if a else 0.19 for a in axis),
               part_role=WHEEL_ROLE,
               rotor_axis=axis, rotor_rated_rpm=6000.0,
               rotor_inertia_kg_m2=0.2, rotor_mass_kg=5.5,
               balance_grade_mm_s=0.4, runs_in=("*",),
               motor_max_torque_n_m=1.0, motor_torque_constant_n_m_a=0.04,
               motor_winding_resistance_ohm=0.25,
               momentum_dump_fraction=0.7, momentum_dump_time_s=20.0)
        for i, node in enumerate(holds):
            _member(g, f"{ident}.strut.{i}", ident, node, radius=0.016)

    biprop = (("tank.mmh", 1.0 / 2.65), ("tank.nto", 1.65 / 2.65))
    thrusters = []          # (node identity, feeds)

    # ---- main engine on its Cardan gimbal ----
    g.assembly = "main-engine"
    pivot = (-1.05, 0.0, 0.0)
    g.node("main.engine", (-1.40, 0.0, 0.0), "thrust-chamber", mass_kg=10.4,
           material="titanium-plate", half_extent_m=(0.35, 0.16, 0.16),
           part_role=THRUSTER_ROLE,
           thruster=dict(role="main", position_m=pivot,
                         direction=(1.0, 0.0, 0.0), max_thrust_n=4000.0,
                         kind="bipropellant",
                         cone_half_angle_rad=math.radians(7.0),
                         gimbal_axis=(0.0, 1.0, 0.0),
                         gimbal_slew_rad_s=math.radians(10.0),
                         throttle_slew_per_s=2.0, deadband=0.4))
    g.edge("main.gimbal", "hull.thrust_structure", "main.engine",
           "universal-joint", radius=0.030, pivot_m=list(pivot),
           load_path="thrust-through-the-gimbal-into-the-thrust-structure")
    for i, node in enumerate((aft[0], aft[1])):
        g.edge(f"main.tvc.{i}", node, "main.engine",
               "linear-hydraulic-actuator", radius=0.016,
               family="linear-hydraulic-actuator",
               kind="thrust-vector-control-actuator",
               commanded_rest_length_m=0.0)
    thrusters.append(("main.engine", biprop))

    # ---- brake: two fore Cardan-gimballed retros, pushing -x ----
    g.assembly = "brake"
    for side, carrier in (("py", fore[0]), ("ny", fore[2])):
        y = 0.95 if side == "py" else -0.95
        ident = f"brake.retro.{side}"
        g.node(ident, (1.0, y, 0.0), "thrust-chamber", mass_kg=3.6,
               material="titanium-plate", half_extent_m=(0.12, 0.05, 0.05),
               solver_condensed_into=carrier, part_role=THRUSTER_ROLE,
               thruster=dict(role="brake", position_m=(1.0, y, 0.0),
                             direction=(-1.0, 0.0, 0.0), max_thrust_n=400.0,
                             kind="bipropellant",
                             cone_half_angle_rad=math.radians(7.0),
                             gimbal_axis=(0.0, 1.0, 0.0),
                             gimbal_slew_rad_s=math.radians(15.0),
                             throttle_slew_per_s=5.0, deadband=0.25))
        g.edge(f"{ident}.gimbal", ident, carrier, "universal-joint",
               radius=0.012, pivot_m=[1.0, y, 0.0],
               load_path="retro-thrust-through-gimbal-into-fore-ring")
        for i, anchor in enumerate((fore[1], fore[3])):
            g.edge(f"{ident}.tvc.{i}", anchor, ident,
                   "linear-hydraulic-actuator", radius=0.010,
                   family="linear-hydraulic-actuator",
                   kind="thrust-vector-control-actuator",
                   commanded_rest_length_m=0.0)
        thrusters.append((ident, biprop))

    # ---- navigation: four RCS quads on the mid ring ----
    g.assembly = "rcs"
    for label, angle in (("py", 0.0), ("pz", 0.5 * math.pi),
                         ("ny", math.pi), ("nz", 1.5 * math.pi)):
        radial = np.asarray((0.0, math.cos(angle), math.sin(angle)))
        tangent = np.asarray((0.0, -math.sin(angle), math.cos(angle)))
        pod = f"rcs.{label}"
        centre = np.asarray((0.2, 0.0, 0.0)) + 0.97 * radial
        g.node(pod, tuple(centre), "load-bearing-structure", mass_kg=2.0,
               material="aluminium-plate", half_extent_m=(0.08, 0.06, 0.06))
        for i, ring in enumerate((fore, mid, aft)):
            ring_node = ring[("py", "pz", "ny", "nz").index(label)]
            _member(g, f"{pod}.strut.{i}", pod, ring_node, radius=0.014)
        for name, direction in (("fwd", np.asarray((1.0, 0.0, 0.0))),
                                ("aft", np.asarray((-1.0, 0.0, 0.0))),
                                ("cw", tangent), ("ccw", -tangent)):
            mount = centre - 0.07 * direction
            ident = f"{pod}.{name}"
            g.node(ident, tuple(mount), "thrust-chamber", mass_kg=0.33,
                   material="titanium-plate",
                   half_extent_m=(0.03, 0.03, 0.03),
                   solver_condensed_into=pod, part_role=THRUSTER_ROLE,
                   thruster=dict(role="navigation",
                                 position_m=tuple(float(v) for v in mount),
                                 direction=tuple(float(v) for v in direction),
                                 max_thrust_n=22.0, kind="monopropellant",
                                 throttle_slew_per_s=50.0))
            g.edge(f"{ident}.flange", ident, pod, "bolted-flange-mount",
                   radius=0.008)
            thrusters.append((ident, (("tank.hydrazine", 1.0),)))

    # ---- the fuel network: one routed line per tank per thruster ----
    g.assembly = "feed-system"
    for ident, feeds in thrusters:
        for tank, share in feeds:
            fluid = next(t[1] for t in tanks if t[0] == tank)
            g.edge(f"feed.{tank.split('.')[-1]}.{ident}", tank, ident,
                   FEED_LINE, radius=0.004,
                   circuit_identity=f"propellant-{fluid}",
                   fluid=fluid, feed_share=share)
    return CraftMachine(g, identity=identity,
                        label="orbital service craft (MMH/NTO, N2H4 RCS)")


# ===================================================================
# THE LAWS
# ===================================================================
def _tensor(prefix: str) -> sp.Matrix:
    """A symmetric 3x3 of columns ``{prefix}_{pair}``."""
    symbols = {pair: sp.Symbol(f"{prefix}_{pair}") for pair in PAIRS}

    def entry(i, j):
        a, b = sorted((i, j))
        return symbols[AXES[a] + AXES[b]]
    return sp.Matrix(3, 3, entry)


def mass_rhs(tank_count: int) -> sp.Expr:
    """Canonical mass: the registered dry mass and tank charges."""
    return sp.Symbol("dry_mass") + sum(
        (tank_symbols(t)["propellant"] for t in range(tank_count)), sp.Integer(0))


def mass_property_rhs(tank_count: int) -> dict:
    """``{column: rhs}``: the reduction as a law of the tank columns
    (module docstring)."""
    masses = [tank_symbols(t)["propellant"] for t in range(tank_count)]
    total = mass_rhs(tank_count)
    moment = sp.Matrix([sp.Symbol(f"dry_moment_{a}") for a in AXES])
    second = _tensor("dry_second")
    for t, m in enumerate(masses):
        moment += m * sp.Matrix([sp.Symbol(f"tank{t}_{a}") for a in AXES])
        second += m * _tensor(f"tank{t}_unit_second")
    # eq_N10_1 for the whole mass at c: J = I_c + M (|c|^2 1 - c c^T)
    shift = (moment.dot(moment) * sp.eye(3) - moment * moment.T) / total
    inertia = second - shift
    out = {f"centre_of_mass_{a}": moment[i] / total
           for i, a in enumerate(AXES)}
    for pair in PAIRS:
        i, j = _pair_index(pair)
        out[f"inertia_{pair}"] = inertia[i, j]
    return out


def mass_property_change_rhs(tank_count, next_state):
    """Exact inertia change using tank increments, with no large subtraction."""
    mass0 = sp.Symbol("dry_mass")
    moment0 = sp.Matrix([sp.Symbol(f"dry_moment_{a}") for a in AXES])
    dm, dq, dsecond = sp.Integer(0), sp.zeros(3, 1), sp.zeros(3)
    for t in range(tank_count):
        propellant = tank_symbols(t)["propellant"]
        delta = next_state[propellant.name] - propellant
        point = sp.Matrix([sp.Symbol(f"tank{t}_{a}") for a in AXES])
        mass0 += propellant
        moment0 += propellant * point
        dm += delta
        dq += delta * point
        dsecond += delta * _tensor(f"tank{t}_unit_second")
    numerator0 = moment0.dot(moment0) * sp.eye(3) - moment0 * moment0.T
    dnumerator = ((2 * moment0.dot(dq) + dq.dot(dq)) * sp.eye(3)
                  - moment0 * dq.T - dq * moment0.T - dq * dq.T)
    return dsecond - dnumerator / (mass0 + dm) + numerator0 * dm / (mass0 * (mass0 + dm))


def spin_free_tensor(wheel_count: int = 0) -> sp.Matrix:
    """``I' = I - sum_w I_w a_w a_w^T`` on the ``inertia_*`` columns and
    each wheel's ``rotor_inertia``/``axis`` columns."""
    inertia = _tensor("inertia")
    for w in range(wheel_count):
        s = wheel_symbols(w)
        a = sp.Matrix([s["axis"][axis] for axis in AXES])
        inertia = inertia - s["inertia"] * a * a.T
    return inertia


def inverse_inertia_rhs(wheel_count: int = 0) -> dict:
    """``{column: rhs}``: ``I'^-1`` (adjugate over determinant) of the
    tensor the mass-properties piece published this substep, without the
    rotors' axial spin (:func:`spin_free_tensor`).  Its own piece: a piece
    reads its inputs, so it must find this substep's tensor already in the
    ``inertia_*`` columns."""
    inertia = spin_free_tensor(wheel_count)
    det = inertia.det(method="berkowitz")
    adjugate = inertia.adjugate()
    out = {}
    for pair in PAIRS:
        i, j = _pair_index(pair)
        out[f"inverse_inertia_{pair}"] = adjugate[i, j] / det
    return out


def euler_rate_tensor_rhs(wheel_count: int = 0) -> dict:
    """``eq_N1_6`` with the full tensor on the TOTAL angular momentum:
    ``d omega/dt = I'^-1 (tau - tau_w - omega x (I' omega + h_w))``, with
    ``I'`` the craft without its rotors' axial spin, ``I'^-1`` from the
    ``inverse_inertia_*`` columns, ``tau_w = sum tau_m a`` the motor
    torques' sum (``wheel_torque_*``; the craft feels its reaction) and
    ``h_w = sum h a`` the wheels' momenta (``wheel_momentum_*``).  The
    catalogue's scalar form is checked to be exactly I^-1 (tau - w x I w)
    first; the wheels enter as torque and as stored momentum in that same
    shape.  No wheels: the craft alone."""
    t = honorary.t
    (rate,) = sp.solve(honorary.eq_N1_6,
                       sp.Derivative(honorary.omega_B(t), t))
    crosses = [term for term in rate.atoms(sp.Function)
               if term.func == honorary.cross]
    if len(crosses) != 1:
        raise RuntimeError("eq_N1_6 no longer has one w x I w term")
    probe_x, probe_tau, probe_i = sp.symbols("probe_x probe_tau probe_i")
    shape = rate.xreplace({crosses[0]: probe_x, honorary.tau_B(t): probe_tau,
                           honorary.I_B(t): probe_i})
    if sp.simplify(shape - (probe_tau - probe_x) / probe_i) != 0:
        raise RuntimeError("eq_N1_6 is no longer I^-1 (tau - w x I w)")
    omega = sp.Matrix([sp.Symbol(f"angular_velocity_{a}") for a in AXES])
    torque = sp.Matrix([sp.Symbol(f"torque_{a}") for a in AXES])
    inertia = spin_free_tensor(wheel_count)
    momentum = inertia * omega
    if wheel_count:
        torque = torque - sp.Matrix([sp.Symbol(f"wheel_torque_{a}")
                                     for a in AXES])
        momentum = momentum + sp.Matrix([sp.Symbol(f"wheel_momentum_{a}")
                                         for a in AXES])
    gyroscopic = omega.cross(momentum)
    rates = _tensor("inverse_inertia") * (torque - gyroscopic)
    return {a: rates[i] for i, a in enumerate(AXES)}


def rotational_state_laws(wheel_count: int):
    """Coupled continuous N1.6/N1.3 laws, including the motor's work.

    The old update changed rotor momentum, then Euler-stepped body rate,
    then Cayley-stepped attitude.  At the audited first two-second step
    those three state points disagreed by 0.080483321 N m s.  Here the
    dt-system integrator receives one state and evaluates all three laws
    at each of its stages.
    """
    omega = sp.Matrix([sp.Symbol(f"angular_velocity_{a}") for a in AXES])
    torque = sp.Matrix([sp.Symbol(f"torque_{a}") for a in AXES])
    rotation = sp.Matrix(3, 3, lambda r, c: attitude_symbols()[(r, c)])
    inertia = spin_free_tensor(wheel_count)
    motors = [wheel_torque_rhs(w) for w in range(wheel_count)]
    axial = [sp.Matrix([wheel_symbols(w)["axis"][a] for a in AXES])
             for w in range(wheel_count)]
    momenta = [wheel_symbols(w)["momentum"] for w in range(wheel_count)]
    rotor_momentum = sum((a * h for a, h in zip(axial, momenta)), sp.zeros(3, 1))
    rotor_torque = sum((a * t for a, t in zip(axial, motors)), sp.zeros(3, 1))
    substitutions = {
        **{sp.Symbol(f"wheel_momentum_{a}"): rotor_momentum[i]
           for i, a in enumerate(AXES)},
        **{sp.Symbol(f"wheel_torque_{a}"): rotor_torque[i]
           for i, a in enumerate(AXES)},
    }
    rates = {f"angular_velocity_{a}": rhs.xreplace(substitutions)
             for a, rhs in euler_rate_tensor_rhs(wheel_count).items()}
    rates.update(attitude_rate_rhs())
    for w in range(wheel_count):
        rates[f"wheel{w}_momentum"] = wheel_rate_rhs(w, motors[w])
    mechanical_power = torque.dot(omega) + sum(
        (motors[w] * wheel_relative_speed(w) for w in range(wheel_count)),
        sp.Integer(0))
    rates["rotation_work"] = mechanical_power
    electrical = [wheel_power_rhs(w, motors[w], wheel_relative_speed(w))
                  for w in range(wheel_count)]
    rates["wheel_energy"] = sum(electrical, sp.Integer(0))
    # The authored motor law separates back-EMF work from I^2 R loss.
    # This is accumulated dissipated energy, not a modeled temperature.
    rates["wheel_copper_loss"] = sum(
        (electrical[w] - motors[w] * wheel_relative_speed(w)
         for w in range(wheel_count)), sp.Integer(0))
    external = rotation * torque
    rates.update({f"angular_impulse_world_{a}": external[i]
                  for i, a in enumerate(AXES)})
    return rates, inertia, rotor_momentum


def rotational_step_equations(wheel_count: int):
    """The actual library RK4 stages and consistent endpoint diagnostics.

    This fixed-inertia slice is also the smallest production physics gate;
    mass-property evolution is composed into the full craft's laws.
    """
    rates, inertia, rotor_momentum = rotational_state_laws(wheel_count)

    def dynamics(_elapsed, state):
        values = {sp.Symbol(name): value for name, value in state.items()}
        return {name: rhs.xreplace(values) for name, rhs in rates.items()}

    stages, next_state = integrated_state_equations(rates, dynamics)
    old = {name: sp.Symbol(name) for name in next_state}
    changes = {name: next_state[name] - old[name] for name in next_state}
    omega = sp.Matrix([old[f"angular_velocity_{a}"] for a in AXES])
    omega_next = sp.Matrix([next_state[f"angular_velocity_{a}"] for a in AXES])
    domega = sp.Matrix([changes[f"angular_velocity_{a}"] for a in AXES])
    # Difference of quadratics: preserve tiny mechanical work without
    # subtracting two nearly equal stored energies in native float64.
    delta_energy = (omega_next + omega).dot(inertia * domega) / 2
    dh = sp.zeros(3, 1)
    for w in range(wheel_count):
        s = wheel_symbols(w)
        delta = changes[f"wheel{w}_momentum"]
        delta_energy += delta * (next_state[f"wheel{w}_momentum"]
                                 + s["momentum"]) / (2 * s["inertia"])
        dh += sp.Matrix([s["axis"][a] for a in AXES]) * delta
    rotation = sp.Matrix(3, 3, lambda r, c: old[attitude_symbols()[(r, c)].name])
    rotation_next = sp.Matrix(3, 3, lambda r, c:
                             next_state[attitude_symbols()[(r, c)].name])
    drotation = sp.Matrix(3, 3, lambda r, c:
                         changes[attitude_symbols()[(r, c)].name])
    dmomentum = (drotation * (inertia * omega + rotor_momentum)
                 + rotation_next * (inertia * domega + dh))
    impulse = sp.Matrix([changes[f"angular_impulse_world_{a}"] for a in AXES])
    angular_error = dmomentum - impulse
    # The controller judges the error introduced by THIS attempt.  Publishing
    # accumulated R.T*R-I exhausted the previous steps' budget and ratcheted
    # the four-second rotor gate to 3047 attempts.  This cancellation-free
    # identity is exactly R_next.T*R_next - R.T*R.
    orthogonal_error = (rotation.T * drotation + drotation.T * rotation
                        + drotation.T * drotation)
    next_substitution = {old[name]: value for name, value in next_state.items()}
    excess = [sp.Max(sp.Integer(0),
                     sp.Abs(wheel_relative_speed(w).xreplace(next_substitution))
                     - wheel_symbols(w)["max_speed"])
              for w in range(wheel_count)]
    eq = lambda name, rhs: sp.Eq(sp.Symbol(name), rhs, evaluate=False)
    commit = tuple(eq(f"{name}_next", value) for name, value in next_state.items()) + (
        eq("orbital_rotation_energy_residual_j",
           sp.Abs(delta_energy - changes["rotation_work"])),
        eq("orbital_angular_momentum_residual_n_m_s", sp.sqrt(angular_error.dot(angular_error))),
        eq("orbital_attitude_orthogonality", sp.sqrt(sum(value**2 for value in orthogonal_error))),
        eq("orbital_wheel_speed_excess_rad_s", sp.Max(*excess) if excess else sp.Integer(0)),
    )
    return stages, commit


def craft_machine_equations(craft: CraftMachine, center_count: int,
                            *, green_coast: bool = False):
    """One coupled catalogue state, integrated by the library's RK4 pieces."""
    thrusters, tank_index = craft.thrusters, craft.tank_index
    n, tanks, wheels = craft.thruster_count, len(craft.tanks), len(craft.wheels)
    gimballed = [k for k, thruster in enumerate(thrusters) if thruster.gimballed]
    tag = f"orbital_machine_rk4_{craft.identity.replace('-', '_')}_t{n}_k{tanks}_w{wheels}_c{center_count}"
    eq = lambda name, rhs: sp.Eq(sp.Symbol(name), rhs, evaluate=False)
    properties = mass_property_rhs(tanks)
    inverses = inverse_inertia_rhs(wheels)
    # These are the machine's existing declared constitutive reductions.
    props = (f"{tag}_properties", tuple(
        eq(f"{name}_next", rhs) for name, rhs in properties.items()))
    inverse = (f"{tag}_inverse", tuple(
        eq(f"{name}_next", rhs) for name, rhs in inverses.items()))
    rates, inertia, rotor_momentum = rotational_state_laws(wheels)
    p = sp.Matrix([sp.Symbol(f"momentum_{axis}") for axis in AXES])
    omega = sp.Matrix([sp.Symbol(f"angular_velocity_{axis}") for axis in AXES])
    rotation = sp.Matrix(3, 3, lambda r, c: attitude_symbols()[r, c])
    mass, flow = sp.Symbol("mass"), PROPELLANT_FLOW
    applied = sp.Matrix([sp.Symbol(f"applied_force_{axis}") for axis in AXES])
    torque = sp.Matrix([sp.Symbol(f"torque_{axis}") for axis in AXES])
    gravity = sp.Matrix([gravity_force_rhs(axis, center_count) for axis in AXES])
    velocity = p / mass
    potential = sp.Integer(0)
    for index in range(center_count):
        radius = sp.sqrt(sum((sp.Symbol(f"position_{axis}") - sp.Symbol(f"center{index}_{axis}"))**2
                             for axis in AXES))
        potential -= sp.Symbol(f"center{index}_mu") / radius
    for i, axis in enumerate(AXES):
        rates[f"momentum_{axis}"] = variable_mass_momentum_rate(axis).xreplace(
            {sp.Symbol(f"force_{axis}"): gravity[i] + applied[i]})
        rates[f"position_{axis}"] = velocity[i]
        rates[f"applied_impulse_world_{axis}"] = applied[i]
        rates[f"applied_delta_v_world_{axis}"] = applied[i] / mass
        rates[f"torque_impulse_body_{axis}"] = torque[i]
    for tank in range(tanks):
        rates[f"tank{tank}_propellant"] = -tank_symbols(tank)["flow"]
    thrusts = [machine_thrust(k, thruster, tank_index) for k, thruster in enumerate(thrusters)]
    for k in range(n):
        rates[f"thruster{k}_impulse"] = thrusts[k]
    rates["fuel_impulse"] = sum(thrusts, sp.Integer(0)) + thrust_cost_integrand("raw_force")
    translation_flux = -flow * (velocity.dot(velocity)/2 + potential)
    rates["translation_work"] = applied.dot(velocity) + translation_flux
    # N1.6 is the catalogue's I(t)*omega_dot form.  The mass-property flux
    # below is its exact open-system contribution to dE/dt and dH/dt.
    inertia_dot = sp.Matrix(3, 3, lambda i, j: sum(
        (-sp.diff(properties[f"inertia_{''.join(sorted((AXES[i], AXES[j])))}"],
                  tank_symbols(tank)["propellant"]) * tank_symbols(tank)["flow"]
         for tank in range(tanks)), sp.Integer(0)))
    rotation_flux = omega.dot(inertia_dot * omega) / 2
    rates["rotation_work"] += rotation_flux
    angular_flux = rotation * inertia_dot * omega
    for i, axis in enumerate(AXES):
        rates[f"angular_impulse_world_{axis}"] += angular_flux[i]
    motors = [wheel_torque_rhs(w) for w in range(wheels)]
    rotor_torque = sum((sp.Matrix([wheel_symbols(w)["axis"][axis] for axis in AXES]) * motors[w]
                        for w in range(wheels)), sp.zeros(3, 1))
    rates["translation_exchange"] = (sp.Abs(applied.dot(velocity))
                                      + sp.Abs(gravity.dot(velocity)) + sp.Abs(translation_flux))
    rates["rotation_exchange"] = (sp.Abs(torque.dot(omega))
                                   + sp.Abs(rotor_torque.dot(omega)) + sp.Abs(rotation_flux))
    # This participant owns electrical consumption/regeneration, including
    # copper loss.  Sum absolute motor powers before integrating: opposing
    # motors must not hide their exchanges by cancelling one another.
    rates["wheel_exchange"] = sum(
        (sp.Abs(wheel_power_rhs(w, motors[w], wheel_relative_speed(w)))
         for w in range(wheels)), sp.Integer(0))
    force_laws = {axis: machine_force_rhs(axis, thrusters, tank_index)
                  + sp.Symbol(f"raw_force_{axis}") for axis in AXES}
    torque_laws = {axis: machine_torque_rhs(axis, thrusters, tank_index) for axis in AXES}
    flow_laws = [tank_flow_rhs(tank, thrusters, tank_index) for tank in range(tanks)]
    full_supply = {tank_symbols(tank)["supply"]: sp.Integer(1) for tank in range(tanks)}
    for tank in range(tanks):
        rates[f"tank{tank}_demand"] = flow_laws[tank].xreplace(full_supply)

    stage_number = 0

    def dynamics(elapsed, state):
        nonlocal stage_number
        prefix = f"rk_stage{stage_number}_"
        stage_number += 1
        values = {sp.Symbol(name): value for name, value in state.items()}
        preparations = []

        def declare(expressions):
            # Constitutive dependencies remain edges in the same existing
            # piece graph.  Inlining Piecewise delivered throttle into the
            # tank-supply condition made SymPy recurse at the first stage;
            # a declared value retains its exact authored law without that
            # expression duplication (the original slew/supply precedent).
            names = {symbol: sp.Symbol(prefix + symbol.name + "_value") for symbol in expressions}
            preparations.append(tuple(eq(f"{names[symbol].name}_next", rhs)
                                      for symbol, rhs in expressions.items()))
            return names

        def actuator_at_stage(rhs, slew):
            if elapsed == 0:
                # An instantaneous actuator takes its commanded right limit
                # at ignition/cutoff.  Finite-rate actuators start at their
                # old state.  Derive both from the same authored slew law;
                # with infinite rate every positive duration has this limit.
                return sp.Piecewise(
                    (rhs.xreplace({DT: sp.Integer(1), slew: sp.oo}), sp.Eq(slew, sp.oo)),
                    (rhs.xreplace({DT: sp.Integer(0)}), True))
            return rhs.xreplace({DT: elapsed})

        stage_properties = {sp.Symbol(name): rhs.xreplace(values) for name, rhs in properties.items()}
        stage_properties[mass] = mass_rhs(tanks).xreplace(values)
        stage_properties.update({machine_thruster_symbols(k)["delivered"]:
                                 actuator_at_stage(delivered_throttle_rhs(k),
                                                   machine_thruster_symbols(k)["throttle_slew"])
                                 for k in range(n)})
        for k in gimballed:
            for actuator in ("a", "b"):
                stage_properties[machine_thruster_symbols(k)[f"gimbal_{actuator}"]] = actuator_at_stage(
                    gimbal_state_rhs(k, actuator), machine_thruster_symbols(k)["gimbal_slew"])
        stage_inputs = declare(stage_properties)
        # The supply law owns a whole-step average from the ORIGINAL tank,
        # against this stage's demand.  No physical tank is updated here.
        relations = {tank_symbols(tank)["supply"]:
                     tank_supply_rhs(tank, thrusters, tank_index).xreplace(stage_inputs)
                     for tank in range(tanks)}
        relations.update({sp.Symbol(name): rhs.xreplace(stage_inputs)
                          for name, rhs in inverses.items()})
        stage_inputs.update(declare(relations))
        exchanges = {sp.Symbol(f"applied_force_{axis}"):
                     rhs.xreplace(values).xreplace(stage_inputs) for axis, rhs in force_laws.items()}
        exchanges.update({sp.Symbol(f"torque_{axis}"):
                          rhs.xreplace(values).xreplace(stage_inputs) for axis, rhs in torque_laws.items()})
        exchanges.update({tank_symbols(tank)["flow"]: flow_laws[tank].xreplace(stage_inputs)
                          for tank in range(tanks)})
        exchanges[flow] = sum((flow_law.xreplace(stage_inputs) for flow_law in flow_laws), sp.Integer(0))
        stage_inputs.update(declare(exchanges))
        return tuple(preparations), {name: rhs.xreplace(values).xreplace(stage_inputs)
                                      for name, rhs in rates.items()}

    stages, next_state = integrated_state_equations(rates, dynamics)
    unclamped_tank_changes = {}
    for tank in range(tanks):
        name = f"tank{tank}_propellant"
        old = sp.Symbol(name)
        unclamped_tank_changes[tank] = next_state[name] - old
        # Max(0, old + delta) == old + Max(-old, delta).  Keeping the
        # library increment exposed lets the conservation laws cancel old
        # symbolically instead of subtracting nearly equal tank charges.
        next_state[name] = old + sp.Max(-old, unclamped_tank_changes[tank])
    mass_before = mass_rhs(tanks)
    final_values = {sp.Symbol(name): value
                    for name, value in next_state.items()}
    mass_after = mass_before.xreplace(final_values)
    properties_after = {name: rhs.xreplace(final_values)
                        for name, rhs in properties.items()}
    dinertia = mass_property_change_rhs(tanks, next_state)
    green_pieces = ()
    if green_coast:
        green_pieces = craft_coast_equation_sets(
            center_count, inertia=inertia,
            inverse_inertia=_tensor("inverse_inertia"),
            internal_angular_momentum=rotor_momentum)
        # Qualify coast from the baked machine aggregate, not its component
        # list.  This is the same boundary as engine/machine baking: summed
        # mass, centre of mass, inertia, and internal rotor momentum.  Green
        # may own translation while that inertial reduction is static and no
        # non-gravity impulse was delivered.  It may own rotation as well
        # only while the summed internal gyro state and external angular
        # impulse are static.
        # ``fuel_impulse`` integrates thrust magnitude plus the norm of the
        # raw-force seam.  Unlike a net vector impulse it cannot hide a
        # reversal inside the attempt, so any non-gravity acceleration keeps
        # the machine translation endpoint authoritative.
        activity = sp.Abs(next_state["fuel_impulse"]
                          - sp.Symbol("fuel_impulse"))
        # Tank charges are the authoritative dynamic inputs to the baked
        # mass/COM/inertia reduction.  If every charge is unchanged, that
        # whole reduction is static by construction.  Re-evaluating the
        # derived COM and tensor and then demanding bitwise equality made a
        # neutral machine miss Green because two evaluations can differ by
        # an ULP even though no physical store changed.
        activity += sum(
            (sp.Abs(next_state[f"tank{tank}_propellant"]
                    - tank_symbols(tank)["propellant"])
             for tank in range(tanks)), sp.Integer(0))
        translation_coast = sp.Eq(activity, sp.Integer(0))
        rigid_activity = activity + sum(
            (sp.Abs(next_state[f"angular_impulse_world_{axis}"]
                    - sp.Symbol(f"angular_impulse_world_{axis}"))
             for axis in AXES), sp.Integer(0))
        rigid_activity += sum(
            (sp.Abs(next_state[f"wheel{wheel}_momentum"]
                    - wheel_symbols(wheel)["momentum"])
             for wheel in range(wheels)), sp.Integer(0))
        rigid_activity += sp.Abs(next_state["wheel_exchange"]
                                 - sp.Symbol("wheel_exchange"))
        rigid_coast = sp.Eq(rigid_activity, sp.Integer(0))
        for axis in AXES:
            for prefix in ("position", "momentum"):
                name = f"{prefix}_{axis}"
                next_state[name] = sp.Piecewise(
                    (sp.Symbol(f"green_coast_{name}"), translation_coast),
                    (next_state[name], True))
            name = f"angular_velocity_{axis}"
            next_state[name] = sp.Piecewise(
                (sp.Symbol(f"green_coast_{name}"), rigid_coast),
                (next_state[name], True))
        for symbol in attitude_symbols().values():
            next_state[symbol.name] = sp.Piecewise(
                (sp.Symbol(f"green_coast_{symbol.name}"), rigid_coast),
                (next_state[symbol.name], True))
        # ``translation_exchange`` is the local RK lane's accumulated
        # gravity-through-velocity activity.  When Green owns translation,
        # that path was not accepted and its participant power must be
        # quiet.  The Green endpoint's translation-energy conservation
        # defect remains live below, alongside its collocation defect.
        next_state["translation_exchange"] = sp.Piecewise(
            (sp.Symbol("translation_exchange"), translation_coast),
            (next_state["translation_exchange"], True))
        next_state["rotation_exchange"] = sp.Piecewise(
            (sp.Symbol("rotation_exchange"), rigid_coast),
            (next_state["rotation_exchange"], True))
        next_state["green_coast_translation_active"] = sp.Piecewise(
            (sp.Integer(1), translation_coast), (sp.Integer(0), True))
        next_state["green_coast_rigid_active"] = sp.Piecewise(
            (sp.Integer(1), rigid_coast), (sp.Integer(0), True))
    remaining = sum((next_state[f"tank{tank}_propellant"] for tank in range(tanks)), sp.Integer(0))
    metrics = mechanical_endpoint_metrics(next_state, mass_before=mass_before, mass_after=mass_after,
                                          inertia_before=inertia, inertia_change=dinertia,
                                          center_count=center_count, wheel_count=wheels)
    if green_coast:
        # Orbital speed is the ordinary local integrator's CFL scale.  A
        # selected Green coast is instead governed by its declared
        # collocation defect (and the shared energy/rotation channels).  Do
        # not let the dormant local translation lane keep publishing its
        # 0.5*dx/|v| ceiling while Green owns the accepted endpoint.
        metrics["max_vel"] = sp.Piecewise(
            (sp.Integer(0), translation_coast), (metrics["max_vel"], True))
    metrics["orbital_green_collocation_residual_m"] = (
        sp.Piecewise((sp.Symbol("green_coast_collocation_residual_m"),
                      translation_coast),
                     (sp.Integer(0), True))
        if green_coast else sp.Integer(0))
    # Sequential pieces publish their outputs immediately.  Conservation
    # therefore reads the old physical columns and the library's exact
    # increments before the single canonical commit.  The endpoint
    # constitutive reductions follow that commit, as they do at entry;
    # substituting the whole new tensor into its inverse duplicated a large
    # expression graph in the former combined endpoint.
    metric_equations = tuple(eq(name, value) for name, value in metrics.items())
    diagnostics = []
    commit = [eq(f"{name}_next", value) for name, value in next_state.items()]
    final_values = {sp.Symbol(name): value for name, value in next_state.items()}
    mean_flows = [(tank_symbols(tank)["propellant"] - next_state[f"tank{tank}_propellant"]) / DT
                   for tank in range(tanks)]
    diagnostics.extend(eq(f"tank{tank}_flow_next", mean_flows[tank]) for tank in range(tanks))
    mean_flow = sum(mean_flows, sp.Integer(0))
    supplies_after = []
    for tank in range(tanks):
        demand = next_state[f"tank{tank}_demand"] - sp.Symbol(f"tank{tank}_demand")
        # For positive demand D, -Max(-P, delta)/D is
        # Min(P/D, -delta/D). Cancel the common library quadrature factor
        # in the latter ratio before lowering. The former spelling emitted
        # (-6/dt)*(1/demand_rates)*(dt/6*supplied_rates), reporting one ULP
        # below full supply even when every stage delivered its exact demand.
        delivered_fraction = sp.Min(
            tank_symbols(tank)["propellant"] / demand,
            sp.cancel(-unclamped_tank_changes[tank] / demand))
        fraction = sp.Piecewise((delivered_fraction, demand > 0), (sp.Integer(1), True))
        supplies_after.append(fraction)
        diagnostics.append(eq(f"tank{tank}_supply_next", fraction))
    diagnostics.extend((eq("propellant_flow_next", mean_flow),
                        eq("propellant_supply_next", sp.Min(*supplies_after) if tanks else sp.Integer(1))))
    commit.extend((eq("propellant_mass_next", remaining), eq("mass_next", mass_after)))
    for k in range(n):
        commit.extend((eq(f"thruster{k}_throttle_state_next", throttle_state_rhs(k)),
                       eq(f"thruster{k}_delivered_next", delivered_throttle_rhs(k))))
    for k in gimballed:
        for actuator in ("a", "b"):
            commit.append(eq(f"thruster{k}_gimbal_{actuator}_next", gimbal_state_rhs(k, actuator)))
    stored_momentum = []
    for axis in AXES:
        mean_force = (next_state[f"applied_impulse_world_{axis}"] - sp.Symbol(f"applied_impulse_world_{axis}")) / DT
        mean_torque = (next_state[f"torque_impulse_body_{axis}"] - sp.Symbol(f"torque_impulse_body_{axis}")) / DT
        diagnostics.extend((eq(f"applied_force_{axis}_next", mean_force), eq(f"torque_{axis}_next", mean_torque)))
        wheel_h = sum((wheel_symbols(w)["axis"][axis] * next_state[f"wheel{w}_momentum"]
                       for w in range(wheels)), sp.Integer(0))
        wheel_tau = sum((wheel_symbols(w)["axis"][axis] * (next_state[f"wheel{w}_momentum"]
                         - wheel_symbols(w)["momentum"]) / DT for w in range(wheels)), sp.Integer(0))
        stored_momentum.append(wheel_h)
        diagnostics.extend((eq(f"wheel_momentum_{axis}_next", wheel_h), eq(f"wheel_torque_{axis}_next", wheel_tau)))
    for store in ("translation", "rotation", "wheel"):
        diagnostics.append(eq(f"{store}_power_next", (next_state[f"{store}_exchange"]
                              - sp.Symbol(f"{store}_exchange")) / DT))
    pieces = [props, inverse]
    pieces.extend((f"{tag}_stage{k}", equations)
                  for k, equations in enumerate(stages))
    pieces.extend(green_pieces)
    pieces.extend(((f"{tag}_metrics", metric_equations),
                   (f"{tag}_diagnostics", tuple(diagnostics)),
                   (f"{tag}_commit", tuple(commit)),
                   (f"{tag}_properties_after", props[1]),
                   (f"{tag}_inverse_after", inverse[1])))
    pieces.extend(energy_metric_equations(tag, center_count))
    return tuple(pieces)


def craft_machine_dt_pieces(craft: CraftMachine, center_count: int,
                            batch: int = 1, *, green_coast: bool = False,
                            retain_compilation: bool = False):
    pieces = tuple(equation_piece(name, equations, batch=batch,
                                 retain_compilation=retain_compilation)
                   for name, equations in craft_machine_equations(
                       craft, center_count, green_coast=green_coast))
    return declare_binding(pieces), ("mass properties", "inverse inertia",
                                    *(piece.entry for piece in pieces[2:]))


def craft_machine_columns(craft: CraftMachine) -> dict:
    """The machine's own columns (one lane): reduction coefficients, tank
    contents and the thruster state machines.  The initial mass-property
    outputs are the reduction at the declared fill."""
    columns = {
        "guidance_enabled": 0.0,
        "guidance_time_s": 0.0,
        "guidance_force_start_s": 0.0,
        "guidance_force_gate_rad": math.pi,
        "guidance_attitude_frequency": 0.0,
        "guidance_attitude_damping": 1.0,
        "guidance_attitude_error_rad": 0.0,
    }
    for axis in AXES:
        columns[f"guidance_force_{axis}"] = 0.0
    for a in AXES:
        for b in AXES:
            columns[f"guidance_target_{a}{b}"] = float(a == b)
    coefficients = craft.reduction_coefficients()
    for i, a in enumerate(AXES):
        columns[f"dry_moment_{a}"] = coefficients["dry_moment"][i]
    for pair in PAIRS:
        i, j = _pair_index(pair)
        columns[f"dry_second_{pair}"] = coefficients["dry_second"][i, j]
    for t, (tank, entry) in enumerate(zip(craft.tanks,
                                          coefficients["tanks"])):
        columns[f"tank{t}_propellant"] = tank.fill_kg
        columns[f"tank{t}_supply"] = 1.0
        columns[f"tank{t}_flow"] = 0.0
        columns[f"tank{t}_demand"] = 0.0
        for i, a in enumerate(AXES):
            columns[f"tank{t}_{a}"] = entry["position"][i]
        for pair in PAIRS:
            i, j = _pair_index(pair)
            columns[f"tank{t}_unit_second_{pair}"] = entry["unit_second"][i, j]
    rigid = craft.mass_properties()
    tensor = np.asarray(rigid.inertia_tensor_kg_m2, dtype=float)
    inverse = np.linalg.inv(craft.spin_free_inertia())
    columns["wheel_energy"] = 0.0
    for a in AXES:
        columns[f"wheel_torque_{a}"] = 0.0
        columns[f"wheel_momentum_{a}"] = 0.0
    for i, a in enumerate(AXES):
        columns[f"centre_of_mass_{a}"] = rigid.center_of_gravity[i]
    for pair in PAIRS:
        i, j = _pair_index(pair)
        columns[f"inertia_{pair}"] = tensor[i, j]
        columns[f"inverse_inertia_{pair}"] = inverse[i, j]
    out = {name: np.full(1, float(value)) for name, value in columns.items()}
    out.update(machine_thruster_columns(craft.thrusters, craft.tank_index))
    out.update(wheel_columns(craft.wheels))
    return out


# ===================================================================
# THE CRAFT IN THE DT SYSTEM
# ===================================================================
@dataclass(frozen=True)
class ThrusterGeometry:
    """One thruster as a drawing or a game needs it, NOW: world-oriented,
    relative to the current centre of mass."""

    identity: str
    role: str
    mount_m: np.ndarray
    exhaust: np.ndarray
    throttle: float


class MachineCraft(OrbitalJumper):
    """The machine craft in one persistent lockstep dt state.

    The jumper's dt glue and seam, with the machine's pieces and columns
    (``_dt_pieces``/``_initial_columns``), plus the machine's own seam:
    ``gimbal(angles)``, ``allocate(force, torque)``, ``throttle_states()``,
    ``gimbal_states()``, ``tank_propellant_kg()``, ``centre_of_mass()``,
    ``inertia_tensor()``, ``thruster_geometry()``.  One lane."""

    def __init__(self, centers, craft: CraftMachine, *, position_m,
                 velocity_m_s, length_scale_m: float, window_s: float,
                 cfl: float = 0.5, attitude=None,
                 angular_velocity_rad_s=(0.0, 0.0, 0.0),
                 error_limits=None, best_effort_metrics=(),
                 green_coast: bool = False):
        self.craft = craft
        self.green_coast = bool(green_coast)
        super().__init__(centers, design=craft, position_m=position_m,
                         velocity_m_s=velocity_m_s,
                         length_scale_m=length_scale_m, window_s=window_s,
                         cfl=cfl, attitude=attitude,
                         angular_velocity_rad_s=angular_velocity_rad_s,
                         error_limits=error_limits,
                         best_effort_metrics=best_effort_metrics)

    # ------------------------------------------------- the jumper's hooks
    def _initial_mass_kg(self):
        # The native commit's mass is the authored Add's dry-first left
        # fold, then its tank increments. Summing charged graph nodes at
        # construction differed by two ULPs before an entirely empty burn.
        # Use the same terms/order here, before the base initializer derives
        # momentum and stored energies from this one initial mass.
        values = {sp.Symbol("dry_mass"): self.craft.dry_mass_kg}
        values.update({tank_symbols(t)["propellant"]: tank.fill_kg
                       for t, tank in enumerate(self.craft.tanks)})
        terms = mass_rhs(len(self.craft.tanks)).as_ordered_terms()
        mass = values[terms[0]]
        for term in terms[1:]:
            mass += values[term]
        return mass

    def _inertia_kg_m2(self) -> np.ndarray:
        return self.craft.principal_inertia()

    def _dt_pieces(self):
        return craft_machine_dt_pieces(self.craft, len(self.centers),
                                       batch=self.batch,
                                       green_coast=self.green_coast)

    def _compose_dt_graph(self, inner_graph, targets, dt_init):
        """Nest the compiled craft below its shared-span guidance round."""
        from llvm_dt_system import piece_leaf

        guidance = MachineGuidancePiece(self.craft, batch=self.batch)
        self.inner_dt_controller = self.dt_controller
        self.dt_controller = STController(dt_min=None, dt_max=None)
        self.piece_labels = (guidance.entry, inner_graph.label)
        return RoundNode(
            plan=SuperstepPlan(round_max=self.window_s, dt_init=dt_init,
                               allow_increase_mid_round=True),
            controller=ControllerNode(ctrl=self.dt_controller,
                                      targets=targets, dx=1.0),
            children=(piece_leaf(guidance, label=guidance.entry), inner_graph),
            schedule="sequential", label="machine-guidance",
        )

    def _initial_columns(self, position_m, velocity_m_s, attitudes,
                         angular_velocity_rad_s) -> dict:
        columns = super()._initial_columns(position_m, velocity_m_s,
                                           attitudes, angular_velocity_rad_s)
        columns.update(craft_machine_columns(self.craft))
        omega = self._lanes(angular_velocity_rad_s, 3)
        columns["rotation_stored"] = np.einsum(
            "bi,ij,bj->b", omega, self.craft.spin_free_inertia(), omega) / 2
        return columns

    # ------------------------------------------------------- the readings
    def _thruster_values(self, suffix: str) -> np.ndarray:
        return np.asarray([self._scalar(f"thruster{k}_{suffix}")
                           for k in range(self.craft.thruster_count)])

    def throttle_states(self) -> np.ndarray:
        """Each thruster's throttle STATE (what it delivers, before the
        deadband)."""
        return self._thruster_values("throttle_state")

    def gimbal_states(self) -> np.ndarray:
        """(n, 2) gimbal actuator angles (zero for fixed thrusters)."""
        out = np.zeros((self.craft.thruster_count, 2))
        for k, thruster in enumerate(self.craft.thrusters):
            if thruster.gimballed:
                out[k] = (self._scalar(f"thruster{k}_gimbal_a"),
                          self._scalar(f"thruster{k}_gimbal_b"))
        return out

    def tank_propellant_kg(self) -> dict:
        return {tank.identity: self._scalar(f"tank{t}_propellant")
                for t, tank in enumerate(self.craft.tanks)}

    @property
    def propellant_kg(self):
        """The sum of the authoritative tank charges in mass_property_rhs."""
        return sum((self._scalar(f"tank{t}_propellant")
                    for t in range(len(self.craft.tanks))), 0.0)

    # ------------------------------------------------------- the wheels
    def wheel_momenta(self) -> np.ndarray:
        """Each wheel's absolute axial angular momentum (N m s)."""
        return np.asarray([self._scalar(f"wheel{w}_momentum")
                           for w in range(len(self.craft.wheels))])

    def wheel_speeds(self) -> np.ndarray:
        """Each rotor's speed relative to the craft (rad/s), ``h / I_w -
        a . w``: what its motor and its speed limit see."""
        omega = self.angular_velocity()
        return np.asarray([
            h / wheel.rotor_inertia_kg_m2
            - float(np.asarray(wheel.axis) @ omega)
            for h, wheel in zip(self.wheel_momenta(), self.craft.wheels)])

    def wheel_commands(self) -> np.ndarray:
        """The motor torque commands (N m) the next round applies."""
        return np.asarray([self._scalar(f"wheel{w}_torque_command")
                           for w in range(len(self.craft.wheels))])

    def wheel_torque(self, torques) -> None:
        """Set each wheel's motor torque COMMAND (N m, on the rotor; the
        craft feels its reaction).  The stage law holds it to the motor's
        limit and the speed band (saturation is the law's)."""
        torques = np.asarray(torques, dtype=float).reshape(
            len(self.craft.wheels))
        for w, value in enumerate(torques):
            self._write_lanes(f"wheel{w}_torque_command", value)

    @property
    def wheel_energy_j(self) -> float:
        """The electrical energy the wheel motors have drawn (J; the motor
        law, net of regeneration)."""
        return (self._scalar("wheel_energy") if self.craft.wheels else 0.0)

    @property
    def wheel_copper_loss_j(self) -> float:
        """Accumulated winding I^2 R dissipation; no temperature is modeled."""
        return self._scalar("wheel_copper_loss") if self.craft.wheels else 0.0

    def spin_free_inertia(self) -> np.ndarray:
        """``I' = I - sum I_w a a^T`` at the tensor the last substep used:
        what the attitude law carries besides the wheels' momenta."""
        return self.inertia_tensor() - wheel_spin_inertia(self.craft.wheels)

    @property
    def propagation_mode(self) -> str:
        """Accepted propagation owner reported by the registered commit."""
        if self.green_coast and self._scalar("green_coast_rigid_active") > 0.5:
            return "GREEN RIGID"
        if (self.green_coast
                and self._scalar("green_coast_translation_active") > 0.5):
            return "GREEN ORBIT"
        return "INTEGRATOR"

    def angular_momentum(self) -> np.ndarray:
        """The craft's TOTAL angular momentum about its centre of mass,
        world frame: ``R (I' w + sum h a)``.  Thrusters change it; the
        wheels only move it between their rotors and the craft."""
        stored = np.zeros(3)
        for h, wheel in zip(self.wheel_momenta(), self.craft.wheels):
            stored += h * np.asarray(wheel.axis, dtype=float)
        return self.attitude() @ (self.spin_free_inertia()
                                  @ self.angular_velocity() + stored)

    @property
    def propellant_supply(self):
        """The fraction of the demand the last substep delivered: the
        minimum of the registered ``tank{t}_supply`` columns; ``1`` with
        no tanks."""
        return min((self._scalar(f"tank{t}_supply")
                    for t in range(len(self.craft.tanks))), default=1.0)

    def centre_of_mass(self) -> np.ndarray:
        """The centre of mass (machine frame) the last substep used."""
        return self._vector("centre_of_mass")

    def inertia_tensor(self) -> np.ndarray:
        """The full inertia tensor (craft axes, about the centre of mass)
        the last substep used."""
        out = np.empty((3, 3))
        for pair in PAIRS:
            i, j = _pair_index(pair)
            out[i, j] = out[j, i] = self._scalar(f"inertia_{pair}")
        return out

    def thruster_geometry(self) -> list[ThrusterGeometry]:
        """Every thruster's mount (relative to the centre of mass) and
        exhaust direction, world-oriented, at the CURRENT gimbal states, and
        its delivered throttle."""
        rotation = self.attitude()
        centre = self.centre_of_mass()
        delivered = delivered_throttles(self.craft.thrusters,
                                        self.throttle_states())
        out = []
        for k, (thruster, angles) in enumerate(zip(self.craft.thrusters,
                                                   self.gimbal_states())):
            direction = thruster.direction_at(*angles)
            out.append(ThrusterGeometry(
                identity=thruster.identity, role=thruster.role,
                mount_m=rotation @ (np.asarray(thruster.position_m, float)
                                    - centre),
                exhaust=-(rotation @ direction),
                throttle=float(delivered[k])))
        return out

    def actuation_matrix(self) -> np.ndarray:
        """World force per unit throttle at the current gimbal states."""
        rotation = self.attitude()
        return np.stack([rotation @ (t.max_thrust_n * t.direction_at(*g))
                         for t, g in zip(self.craft.thrusters,
                                         self.gimbal_states())], axis=1)

    def commanded_force(self) -> np.ndarray:
        """The world force the current STATES deliver (deadband applied,
        full supply) plus the raw force."""
        force, _torque = self.commanded_wrench()
        return force

    def commanded_wrench(self):
        force, torque = _host_wrench(self.craft, self.throttle_states(),
                                     self.gimbal_states(),
                                     self.centre_of_mass(), self.attitude())
        return force + self._vector("raw_force"), torque

    # ------------------------------------------------------- the commands
    def gimbal(self, angles) -> None:
        """Set the gimbal actuator COMMANDS ((n, 2); fixed thrusters'
        rows are ignored).  A command outside the declared cone is
        refused."""
        angles = np.asarray(angles, dtype=float).reshape(
            self.craft.thruster_count, 2)
        for k, thruster in enumerate(self.craft.thrusters):
            if not thruster.gimballed:
                continue
            a, b = angles[k]
            if (math.cos(a) * math.cos(b)
                    < math.cos(thruster.cone_half_angle_rad) - 1.0e-12):
                raise ValueError(f"{thruster.identity}: gimbal command "
                                 f"({a}, {b}) is outside its cone")
            self._write_lanes(f"thruster{k}_gimbal_a_command", a)
            self._write_lanes(f"thruster{k}_gimbal_b_command", b)

    #: Declared: ``allocate`` applies the whole command (throttles and
    #: gimbal targets) itself, so a driver's ``throttle(u)`` after it is
    #: redundant (``orbital_tracker.fly`` skips it on this declaration).
    applies_allocation = True

    def allocate(self, wrench, torque_n_m=None, *,
                 round_s: float | None = None, apply: bool = True,
                 **weights) -> Allocation:
        """The seam the tracker drives: ``allocate(wrench)`` with a wrench
        carrying ``.force_n`` (world) and ``.torque_n_m`` (craft frame,
        about the current centre of mass) -- or ``allocate(force, torque)``
        -- answered with the commands that best achieve it from the CURRENT
        states within ``round_s`` (default: the window), under every
        declared limit (``orbital_actuation.allocate_wrench``).  Returns the
        :class:`orbital_actuation.Allocation` (``.achieved``,
        ``.throttles``, ``.gimbal_rad``, the shortfalls).  ``apply`` writes
        the throttle AND gimbal commands for the next round."""
        if torque_n_m is None:
            force_n, torque_n_m = wrench.force_n, wrench.torque_n_m
        else:
            force_n = wrench
        allocation = allocate_wrench(
            self.craft.thrusters, force_n, torque_n_m,
            centre_of_mass_m=self.centre_of_mass(), attitude=self.attitude(),
            throttle_state=self.throttle_states(),
            gimbal_state=self.gimbal_states(),
            tank_propellant_kg=self.tank_propellant_kg(),
            round_s=self.window_s if round_s is None else round_s,
            wheels=self.craft.wheels,
            wheel_momentum_n_m_s=self.wheel_momenta(),
            angular_velocity_rad_s=self.angular_velocity(), **weights)
        if apply:
            self.throttle(allocation.throttles)
            self.gimbal(allocation.gimbal_rad)
            if self.craft.wheels:
                self.wheel_torque(allocation.wheel_torque_n_m)
        return allocation

    def guidance_intent(self, target_attitude, force_n, gains, *,
                        force_gate_rad=math.pi, force_start_s=None) -> None:
        """Publish the wrench/attitude intent consumed by the outer dt law."""
        target = np.asarray(target_attitude, dtype=float).reshape(3, 3)
        force = np.asarray(force_n, dtype=float).reshape(3)
        self._write_lanes("guidance_enabled", 1.0)
        self._write_lanes("guidance_force_gate_rad", float(force_gate_rad))
        self._write_lanes("guidance_force_start_s",
                          self.time_s if force_start_s is None
                          else float(force_start_s))
        self._write_lanes("guidance_attitude_frequency",
                          float(gains.attitude_frequency_rad_s))
        self._write_lanes("guidance_attitude_damping",
                          float(gains.attitude_damping_ratio))
        for index, axis in enumerate(AXES):
            self._write_lanes(f"guidance_force_{axis}", force[index])
        for i, a in enumerate(AXES):
            for j, b in enumerate(AXES):
                self._write_lanes(f"guidance_target_{a}{b}", target[i, j])

    def clear_guidance(self) -> None:
        """Leave the outer guidance participant inert between flight calls."""
        self._write_lanes("guidance_enabled", 0.0)

def _host_wrench(craft, throttle_states, gimbal_states, centre, attitude):
    from orbital_actuation import machine_wrench
    return machine_wrench(craft.thrusters, throttle_states, gimbal_states,
                          centre, attitude)


__all__ = ["CraftMachine", "MachineCraft", "PropellantTank",
           "WHEEL_ROLE", "spin_free_tensor", "rotational_state_laws",
           "ThrusterGeometry", "craft_machine_columns",
           "craft_machine_dt_pieces", "euler_rate_tensor_rhs",
           "mass_property_rhs", "orbital_craft", "THRUSTER_KINDS",
           "feed_symbol", "attitude_symbols", "PROPELLANT_FLOW",
           "CENTRE_OF_MASS"]
