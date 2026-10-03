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

Pieces, in causal order (one ``RoundNode``, ``sequential``):

    mass properties   centre_of_mass_*, inertia_ab
    inverse inertia   inverse_inertia_ab: the inverse of the craft's tensor
                      without its rotors' axial spin, I' = I - sum I_w a a^T
                      (the spin is the wheels' own momentum column)
    slew              thruster throttle states, delivered throttles
                      (deadband) and gimbal actuator states
                      (publishes ``dt_limit``: the actuators' sampling
                      bound, ``orbital_actuation.actuator_dt_limit_rhs``)
    supply            tank{t}_supply (step 7's law per tank)
    actuation         applied_force_* (world), torque_* (craft, about the
                      current centre of mass), tank{t}_flow, propellant_flow
    wheels            wheel{w}_momentum (motor torque held to the motor
                      limit and the speed band), wheel_torque_*,
                      wheel_momentum_* (craft frame), wheel_energy (the
                      motor law's electrical energy drawn); publishes
                      ``dt_limit``, the gyroscopic bound of the stored
                      momentum (:func:`wheel_gyroscopic_dt_limit_rhs`)
    N4.1 gravity      the jumper's piece
    momentum          N7.2; tank draws; mass; N1.6 on the TOTAL angular
                      momentum (craft + wheels) with the full tensor;
                      publishes ``dt_limit``, the propellant step bound
    position          the jumper's piece (N1.1, N1.3 Cayley, max_vel,
                      dt_limit)
    thrust cost       per-thruster impulse, fuel_impulse

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
    actuator_dt_limit_rhs,
    allocate_wrench,
    attitude_symbols,
    delivered_throttle_rhs,
    delivered_throttles,
    feed_symbol,
    gimbal_state_rhs,
    machine_force_rhs,
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
    ATTITUDE_STEP_MAX,
    ATTITUDE_STEP_RAD,
    OrbitalJumper,
    declare_binding,
    exchange_publication,
    leapfrog_momentum,
    orbital_jumper_dt_pieces,
    propellant_dt_limit_rhs,
    thrust_cost_integrand,
    variable_mass_momentum_rate,
)

#: The propellant-tank role and the thruster role, as declared on nodes.
TANK_ROLE = "propellant-tank"
THRUSTER_ROLE = "thruster"
#: The routed member that carries propellant from a tank to a thruster.
FEED_LINE = "fuel-line"
#: The reaction-wheel role, as declared on nodes.
WHEEL_ROLE = "reaction-wheel"
#: The craft's declared impulse step (N s): the most impulse one substep may
#: misplace by sampling a throttle or gimbal state that is still moving
#: (``orbital_actuation.actuator_dt_limit_rhs``).  10 N s is 0.01 m/s on
#: this ~1 t craft, an eighth of the tracker's coast velocity band at 7000 km.
IMPULSE_STEP_N_S = 10.0
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
    tanks = (
        ("tank.mmh", "monomethylhydrazine", (0.60, 0.0, 0.0), 0.38, 0.72,
         13.0, 260.0, 240.0, mid + fore[:1]),
        ("tank.nto", "nitrogen-tetroxide", (-0.25, 0.0, 0.0), 0.38, 0.72,
         13.0, 420.0, 396.0, mid + aft[:1]),
        ("tank.hydrazine", "hydrazine", (0.15, 0.0, 0.60), 0.17, 0.70,
         5.0, 62.0, 60.0, (mid[1], fore[1], aft[1])),
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

    # ---- brake: two retro thrusters at the fore ring, pushing -x ----
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
                             throttle_slew_per_s=5.0, deadband=0.25))
        g.edge(f"{ident}.flange", ident, carrier, "bolted-flange-mount",
               radius=0.012)
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


def mass_property_rhs(tank_count: int) -> dict:
    """``{column: rhs}``: the reduction as a law of the tank columns
    (module docstring)."""
    dry_mass = sp.Symbol("dry_mass")
    masses = [tank_symbols(t)["propellant"] for t in range(tank_count)]
    total = dry_mass + sum(masses, sp.Integer(0))
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


def wheel_gyroscopic_dt_limit_rhs(momentum) -> sp.Expr:
    """The wheels' stability bound on the step: ``attitude_step_max /
    (|h_w| ||I'^-1||_F)``.  Stored momentum makes the attitude law
    ``I' w' = ... - w x h_w``, a rotation of ``w`` at up to ``|h_w| /
    I'_min`` rad/s whatever the craft's own rate (the nutation of a craft
    carrying a spinning wheel); explicit Euler steps that oscillator with
    gain ``sqrt(1 + (lambda dt)^2)`` per step, so the step is held to the
    same declared phase per step as the rotation's own bound
    (``orbital_jumper.attitude_dt_limit_rhs``), with the Frobenius norm of
    ``I'^-1`` (an upper bound on its largest eigenvalue).  +inf with the
    wheels at rest."""
    norm2 = sum(sp.Symbol(f"inverse_inertia_{pair}")**2
                * (1 if pair[0] == pair[1] else 2) for pair in PAIRS)
    stored = sp.sqrt(sum(component**2 for component in momentum))
    return ATTITUDE_STEP_MAX / (stored * sp.sqrt(norm2))


def wheel_piece_equations(wheel_count: int) -> tuple:
    """The wheel piece (module docstring), ``{column: rhs}`` pairs plus
    its ``dt_limit``."""
    torques = [wheel_torque_rhs(w) for w in range(wheel_count)]
    momenta = [wheel_symbols(w)["momentum"] + DT * wheel_rate_rhs(
        w, torques[w]) for w in range(wheel_count)]
    power = sp.Integer(0)
    for w in range(wheel_count):
        # the rotor's speed is linear over the step (constant motor
        # torque): its mean is the mid-step speed
        speed = (wheel_relative_speed(w)
                 + wheel_relative_speed(w, momenta[w])) / 2
        power += wheel_power_rhs(w, torques[w], speed)
    out = [(f"wheel{w}_momentum_next", momenta[w])
           for w in range(wheel_count)]
    axes = [wheel_symbols(w)["axis"] for w in range(wheel_count)]
    stored = []
    for a in AXES:
        out.append((f"wheel_torque_{a}_next", sum(
            (axes[w][a] * torques[w] for w in range(wheel_count)),
            sp.Integer(0))))
        total = sum((axes[w][a] * momenta[w] for w in range(wheel_count)),
                    sp.Integer(0))
        stored.append(total)
        out.append((f"wheel_momentum_{a}_next", total))
    out.append(("wheel_energy_next", sp.Symbol("wheel_energy") + DT * power))
    out.append(("dt_limit", wheel_gyroscopic_dt_limit_rhs(stored)))
    return tuple(out)


def craft_machine_dt_pieces(craft: CraftMachine, center_count: int,
                            batch: int = 1):
    """The nine pieces (module docstring), in causal order, with labels."""
    thrusters, tank_index = craft.thrusters, craft.tank_index
    n, tanks = craft.thruster_count, len(craft.tanks)
    wheels = len(craft.wheels)
    gimballed = [k for k, t in enumerate(thrusters) if t.gimballed]
    tag = (f"{craft.identity.replace('-', '_')}_t{n}_k{tanks}"
           f"_g{len(gimballed)}")

    def eq(name, rhs):
        return sp.Eq(sp.Symbol(name), rhs, evaluate=False)

    props = equation_piece(f"orbital_machine_mass_properties_{tag}", tuple(
        eq(f"{name}_next", rhs)
        for name, rhs in mass_property_rhs(tanks).items()), batch=batch)
    inverse = equation_piece(
        "orbital_machine_inverse_inertia"
        + (f"_w{wheels}" if wheels else ""), tuple(
            eq(f"{name}_next", rhs)
            for name, rhs in inverse_inertia_rhs(wheels).items()),
        batch=batch)
    slew_equations = [eq(f"thruster{k}_throttle_state_next",
                         throttle_state_rhs(k)) for k in range(n)]
    slew_equations += [eq(f"thruster{k}_delivered_next",
                          delivered_throttle_rhs(k)) for k in range(n)]
    for k in gimballed:
        for actuator in ("a", "b"):
            slew_equations.append(eq(f"thruster{k}_gimbal_{actuator}_next",
                                     gimbal_state_rhs(k, actuator)))
    slew_equations.append(eq("dt_limit", actuator_dt_limit_rhs(thrusters)))
    slew = equation_piece(f"orbital_machine_slew_{tag}",
                          tuple(slew_equations), batch=batch)
    supplies = [tank_supply_rhs(t, thrusters, tank_index)
                for t in range(tanks)]
    supply = equation_piece(f"orbital_machine_supply_{tag}", (
        *(eq(f"tank{t}_supply_next", supplies[t]) for t in range(tanks)),
        eq("propellant_supply_next",
           sp.Min(*[tank_symbols(t)["supply"] for t in range(tanks)])
           if tanks > 1 else (tank_symbols(0)["supply"] if tanks
                              else sp.Integer(1))),
    ), batch=batch)
    raw = {a: sp.Symbol(f"raw_force_{a}") for a in AXES}
    flows = [tank_flow_rhs(t, thrusters, tank_index) for t in range(tanks)]
    actuation = equation_piece(f"orbital_machine_actuation_{tag}", (
        *(eq(f"applied_force_{a}_next",
             machine_force_rhs(a, thrusters, tank_index) + raw[a])
          for a in AXES),
        *(eq(f"torque_{a}_next", machine_torque_rhs(a, thrusters, tank_index))
          for a in AXES),
        *(eq(f"tank{t}_flow_next", flows[t]) for t in range(tanks)),
        eq("propellant_flow_next", sum(flows, sp.Integer(0))),
    ), batch=batch)

    spin = (equation_piece(f"orbital_machine_wheels_w{wheels}", tuple(
        eq(name, rhs) for name, rhs in wheel_piece_equations(wheels)),
        batch=batch) if wheels else None)

    jumper = orbital_jumper_dt_pieces(center_count, 0, batch=batch)
    gravity, position = jumper[2], jumper[4]

    dt = DT
    draws = [tank_symbols(t)["propellant"]
             - sp.Min(tank_symbols(t)["propellant"],
                      dt * tank_symbols(t)["flow"]) for t in range(tanks)]
    remaining = sum(draws, sp.Integer(0))
    rates = euler_rate_tensor_rhs(wheels)
    # the jumper's momentum law: kick by the mean of the adjacent steps,
    # publish translational kinetic energy and |F . v|
    leapfrog = {a: leapfrog_momentum(a) for a in AXES}
    momentum_next = {a: leapfrog[a][0] for a in AXES}
    mass_new = sp.Symbol("dry_mass") + remaining
    momentum = equation_piece(
        f"orbital_machine_momentum_k{tanks}"
        + (f"_w{wheels}" if wheels else "") + f"_c{center_count}", (
        *(eq(f"momentum_{a}_next", momentum_next[a]) for a in AXES),
        *(eq(f"momentum_carry_{a}_next", leapfrog[a][1]) for a in AXES),
        eq("dt_prev_next", dt),
        *exchange_publication(
            center_count,
            sum(momentum_next[a]**2 for a in AXES) / (2 * mass_new),
            sp.Abs(sum((sp.Symbol(f"force_{a}")
                        - sp.Symbol(f"applied_force_{a}"))
                       * momentum_next[a] for a in AXES)) / mass_new),
        eq("dt_limit", propellant_dt_limit_rhs(mass_new)),
        *(eq(f"tank{t}_propellant_next", draws[t]) for t in range(tanks)),
        eq("propellant_mass_next", remaining),
        eq("mass_next", sp.Symbol("dry_mass") + remaining),
        *(eq(f"angular_velocity_{a}_next",
             sp.Symbol(f"angular_velocity_{a}") + dt * rates[a])
          for a in AXES),
    ), batch=batch)

    fuel = sp.Symbol("fuel_impulse")
    impulses = [eq(f"thruster{k}_impulse_next",
                   thruster_symbols(k)["impulse"]
                   + dt * machine_thrust(k, thrusters[k], tank_index))
                for k in range(n)]
    total_rate = (sum((machine_thrust(k, thrusters[k], tank_index)
                       for k in range(n)), sp.Integer(0))
                  + thrust_cost_integrand("raw_force"))
    cost = equation_piece(f"orbital_machine_thrust_cost_{tag}", (
        *impulses, eq("fuel_impulse_next", fuel + dt * total_rate),
    ), batch=batch)
    pieces = (props, inverse, slew, supply, actuation, *(
        (spin,) if spin is not None else ()), gravity, momentum, position,
        cost)
    labels = ("mass properties", "inverse inertia", "slew",
              "propellant supply", "actuation", *(
                  ("wheels",) if spin is not None else ()),
              "N4.1 gravity", "N7.2/N1.6 momentum", "N1.1/N1.3 position",
              "thrust cost")
    return declare_binding(pieces), labels


def craft_machine_columns(craft: CraftMachine) -> dict:
    """The machine's own columns (one lane): reduction coefficients, tank
    contents and the thruster state machines.  The initial mass-property
    outputs are the reduction at the declared fill."""
    columns = {}
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
        for i, a in enumerate(AXES):
            columns[f"tank{t}_{a}"] = entry["position"][i]
        for pair in PAIRS:
            i, j = _pair_index(pair)
            columns[f"tank{t}_unit_second_{pair}"] = entry["unit_second"][i, j]
    rigid = craft.mass_properties()
    tensor = np.asarray(rigid.inertia_tensor_kg_m2, dtype=float)
    inverse = np.linalg.inv(craft.spin_free_inertia())
    columns["impulse_step_max"] = IMPULSE_STEP_N_S
    if craft.wheels:
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
                 attitude_step_rad: float = ATTITUDE_STEP_RAD):
        self.craft = craft
        super().__init__(centers, design=craft, position_m=position_m,
                         velocity_m_s=velocity_m_s,
                         length_scale_m=length_scale_m, window_s=window_s,
                         cfl=cfl, attitude=attitude,
                         angular_velocity_rad_s=angular_velocity_rad_s,
                         attitude_step_rad=attitude_step_rad)

    # ------------------------------------------------- the jumper's hooks
    def _inertia_kg_m2(self) -> np.ndarray:
        return self.craft.principal_inertia()

    def _dt_pieces(self):
        return craft_machine_dt_pieces(self.craft, len(self.centers),
                                       batch=self.batch)

    def _initial_columns(self, position_m, velocity_m_s, attitudes,
                         angular_velocity_rad_s) -> dict:
        columns = super()._initial_columns(position_m, velocity_m_s,
                                           attitudes, angular_velocity_rad_s)
        columns.update(craft_machine_columns(self.craft))
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
        """The propellant aboard: the sum of the tank charges -- the
        ``tank{t}_propellant`` columns the mass-properties law reduces
        (:func:`mass_property_rhs`).  Replaces the jumper's reading of a
        ``propellant_mass`` column: the machine's momentum piece writes
        ``propellant_mass_next`` but no piece reads it, so the dt system
        keeps no such column (a column exists only when a piece reads it)
        and the inherited property raised AttributeError."""
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
        craft feels its reaction).  The wheel piece holds it to the motor's
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

    def spin_free_inertia(self) -> np.ndarray:
        """``I' = I - sum I_w a a^T`` at the tensor the last substep used:
        what the attitude law carries besides the wheels' momenta."""
        return self.inertia_tensor() - wheel_spin_inertia(self.craft.wheels)

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
        minimum of the ``tank{t}_supply`` columns the thrusters read (the
        same ``Min`` the supply piece publishes as ``propellant_supply``,
        which no machine piece reads, so the dt system keeps no such
        column); ``1`` with no tanks."""
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

    # ------------------------------------------------------------ the round
    def actuator_dt_limit(self) -> float:
        """The slew piece's own ``dt_limit`` for the commands now standing:
        the COMPILED slew piece, called on the host at ``dt = 0`` (the
        states where they are, the travel to the new commands still ahead).
        Every later substep is bounded by the piece's publication; this is
        the first one's (``advance`` hands it to the controller)."""
        piece = self.pieces[self.piece_labels.index("slew")]
        arguments = [np.zeros(self.batch) if name == "dt"
                     else np.ascontiguousarray(np.asarray(
                         self._span(name), dtype=np.float64))
                     for name in piece.argument_names]
        outputs = dict(zip(piece.output_names, piece(*arguments)))
        return float(np.min(np.asarray(outputs["dt_limit"], dtype=float)))

    def advance(self, window_s: float | None = None):
        """One round.  A command written since the last round starts a ramp
        the controller's continuation knows nothing of: its first attempt
        is pinned by the controller's own pre-trial bound,
        ``dt_limit_hint`` (``dt_controller.run_superstep``), set to the
        slew piece's ``dt_limit`` for those commands (+inf: no pin)."""
        limit = self.actuator_dt_limit()
        self.dt_state.dt_limit_hint = (limit if math.isfinite(limit)
                                       and limit > 0.0 else None)
        return super().advance(window_s)


def _host_wrench(craft, throttle_states, gimbal_states, centre, attitude):
    from orbital_actuation import machine_wrench
    return machine_wrench(craft.thrusters, throttle_states, gimbal_states,
                          centre, attitude)


__all__ = ["CraftMachine", "MachineCraft", "PropellantTank",
           "IMPULSE_STEP_N_S", "WHEEL_ROLE", "spin_free_tensor",
           "wheel_gyroscopic_dt_limit_rhs", "wheel_piece_equations",
           "ThrusterGeometry", "craft_machine_columns",
           "craft_machine_dt_pieces", "euler_rate_tensor_rhs",
           "mass_property_rhs", "orbital_craft", "THRUSTER_KINDS",
           "feed_symbol", "attitude_symbols", "PROPELLANT_FLOW",
           "CENTRE_OF_MASS"]
