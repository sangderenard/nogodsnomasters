"""A CRAFT'S GRAPH: ENGINES IN THE DT SYSTEM, BODIES ON THEIR OWN.

    python time_trials/craft_graph.py --frames 120

Two layers, joined only where the ABI says they may be.

  THE ENGINE BATCH. Every craft's `EngineCycleSim` is registered as a
  `DtCompatibleEngine` in ONE dt round and advanced together, each
  publishing its own `dt_limit`. That is the global engine batch: the
  interior of a machine -- crank, cylinders, fuel, circuits -- is private,
  stiff, phase-locked and necessarily one time zone, and it is what
  `engine_abi` says batches by topology.

  THE VEHICLE GRAPHS. Each craft runs turing's own
  `abstract_ui_vehicle_step` independently, taking from its engine only
  the EXTERIOR: shaft speed and shaft torque. `engine_abi` is explicit
  that the exterior is "small, and the ONLY place couplings live", so it
  is the only thing that should cross. Nothing else passes between the
  two layers in either direction, except the coupling's reaction going
  back as load.

WHY THIS SHAPE AND NOT THE OTHER ONE. Stepping each engine inside its own
vehicle's tick binds the two clocks together and makes the engines
un-batchable -- four engines become four separate interiors advanced at
whatever their vehicle happened to need. Batched, they share one round and
one substep negotiation, and a vehicle graph is then a consumer of a
published number rather than an owner of an engine.

THE LAW'S OWN ENGINE IS SILENCED, not removed: its displacement is set to
zero so its combustion contributes nothing, its `engine_angular_speed` is
written from the real crank each tick instead of being carried from its
own integration, and the clutch torque it reports is written back to the
real sim as `brake_load_nm`. Without that last step the sim free-revs and
the car coasts, which looks like it is working.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ENGINE_TOY = Path(__file__).resolve().parent.parent
if str(_ENGINE_TOY) not in sys.path:
    sys.path.insert(0, str(_ENGINE_TOY))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np                                                    # noqa: E402

import engines                                                        # noqa: E402
from engine_cycle_sim import EngineCycleSim, FIXED_PHYSICS_DT_S       # noqa: E402
from dt_benchmark import CycleEngine, use_numpy_backend               # noqa: E402
from time_allocator import derive_dt_limit_s                          # noqa: E402
from src.common.dt_system.dt_controller import STController, Targets  # noqa: E402
from src.common.dt_system.dt_graph import GraphBuilder, MetaLoopRunner  # noqa: E402
from src.common.dt_system.engine_api import EngineRegistration        # noqa: E402
from src.common.dt_system.state_table import StateTable               # noqa: E402
from src.common.dt_system.realtime import RealtimeConfig              # noqa: E402

from src.common.tensors import AbstractTensor as AT                   # noqa: E402
from src.compiler.abstract_ui_vehicles import (                       # noqa: E402
    compile_symbolic_vehicle_physics, compile_wheel_contact_ssa,
    load_default_car_configuration, basic_craft_defaults,
    BASIC_CRAFT_ROLLING_RADIUS_M, WHEEL_NAMES)
from native_law import build_law                                      # noqa: E402

RAD_PER_S_PER_RPM = math.pi / 30.0
FLOOR_S = FIXED_PHYSICS_DT_S


# ---------------------------------------------------------------------
# layer one: every engine, batched in the dt system
# ---------------------------------------------------------------------

@dataclass
class EngineBatch:
    """Every craft's engine in one dt round.

    The exterior each one publishes is two numbers: what its output shaft
    is turning at, and what it is delivering. That is all a vehicle graph
    is allowed to see of it.
    """

    sims: dict[str, EngineCycleSim]
    registrations: list = field(default_factory=list)
    table: StateTable = field(default_factory=StateTable)
    builder: GraphBuilder = field(init=False)
    runner: MetaLoopRunner = field(init=False)
    config: RealtimeConfig = field(init=False)
    #: the declared flat span -- see `compile_contract.BATCH_RECORD`
    state_span: object = None

    @classmethod
    def build(cls, identities: dict[str, str]) -> "EngineBatch":
        use_numpy_backend()
        sims = {}
        for name, identity in identities.items():
            sim = EngineCycleSim(engine=engines.get(identity))
            sim.start()
            sim.gear_index = 1
            sims[name] = sim
        batch = cls(sims=sims)
        targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
        batch.builder = GraphBuilder(ctrl=STController(dt_min=1e-7),
                                     targets=targets, dx=0.1)
        # the realtime lane: no `_RoundTransaction`, no deepcopy checkpoint
        # per attempt, no rollback. A game does not re-run a frame.
        batch.config = RealtimeConfig(budget_ms=1000.0, slack=1.0)
        batch.runner = MetaLoopRunner(realtime_config=batch.config,
                                      realtime=True, state_table=batch.table)
        for name, sim in sims.items():
            batch.registrations.append(EngineRegistration(
                name=f"{name}.engine", engine=CycleEngine(sim), targets=targets,
                dx=0.1, localize=True))
        return batch

    def step(self, dt: float) -> None:
        node = self.builder.round(dt=dt, engines=self.registrations,
                                  realtime_config=self.config,
                                  state_table=self.table)
        self.runner.run_round(node, dt=dt, state_table=self.table)

    # -- the declared span: every engine's interior, end to end --------
    def sync_state_span(self):
        """Concatenate every engine's declared span into one array.

        An `EngineBatch` is opaque to the compiler until it has one, and
        the measured consequence is exact: lowering the game refuses with
        `blockers=('opaque-state-effect',) batch.step(dt)`.
        """
        spans = [sim.sync_state_span() for sim in self.sims.values()]
        self.state_span = np.concatenate(spans) if spans else np.zeros(0)
        return self.state_span

    def load_state_span(self, span=None) -> None:
        source = self.state_span if span is None else span
        if source is None:
            return
        cursor = 0
        for sim in self.sims.values():
            width = sim.state_span_length()
            sim.load_state_span(np.asarray(source[cursor:cursor + width]))
            cursor += width

    def exterior(self, name: str) -> tuple[float, float]:
        """(shaft omega rad/s, shaft torque Nm) -- the whole crossing."""
        sim = self.sims[name]
        return (float(sim.rpm or 0.0) * RAD_PER_S_PER_RPM,
                float(sim.state.torque_rms_nm or 0.0))

    def dt_limits(self) -> dict[str, float]:
        return {name: derive_dt_limit_s(sim.engine) or 1.0 / 240.0
                for name, sim in self.sims.items()}


# ---------------------------------------------------------------------
# layer two: one vehicle graph per craft
# ---------------------------------------------------------------------

def craft_parameters(configuration=None) -> dict[str, float]:
    """Every figure the body needs, COMPUTED by the configuration.

    None of these are chosen here. `VehicleConfiguration.mass_properties()`
    sums 118 declared components -- engine, transmission, differentials,
    fuel, wheels, tires, calipers, harness, body shell -- into a total
    mass, a centre of mass and a roll/pitch/yaw inertia tensor, and
    `wheel_rotational_inertia()` does the same for a corner. They were
    literals in this file until it was pointed out that the system
    already derives them, and the invented ones were wrong: a chassis
    mass of 520 kg against a computed sprung mass of 524 kg, and yaw
    inertia of 380 against a computed 728.

    The one thing still stated rather than derived is that the law's own
    combustion is silenced, because the real `EngineCycleSim` is supplying
    crank torque and two engines would double-count.
    """
    configuration = configuration or load_default_car_configuration()
    values = dict(configuration.parameter_defaults())
    mass = configuration.mass_properties()
    inertia = mass["inertia_kg_m2"]
    total_kg = float(mass["total_mass_kg"])
    values.update({
        "inverse_mass": 1.0 / total_kg,
        "inverse_inertia_roll": 1.0 / float(inertia["roll"]),
        "inverse_inertia_pitch": 1.0 / float(inertia["pitch"]),
        "inverse_inertia_yaw": 1.0 / float(inertia["yaw"]),
        "wheel_inertia": float(configuration.wheel_rotational_inertia()),
        "gravity": -9.81,
        # the real engine is the engine; the law's own combustion is off
        "engine_displacement_m3": 0.0,
        "engine_enabled": 1.0,
        "assembly_alpha_drivetrain": 1.0,
        "yaw_cos": 1.0, "yaw_sin": 0.0,
        "drive_direction": 1.0,
    })
    for axis, value in zip("xyz", mass["center_of_mass"]):
        if f"center_of_mass_{axis}" in values:
            values[f"center_of_mass_{axis}"] = float(value)
    for corner in WHEEL_NAMES:
        values[f"assembly_alpha_{corner}"] = 1.0
    values["_total_mass_kg"] = total_kg
    return values


@dataclass
class Craft:
    """One craft: a ROW in the fleet's columns, not an object with a law.

    Everything a craft is lives in slot `row` of every column of the two
    kernels. It owns no arrays and calls nothing.
    """

    name: str
    row: int
    last: dict = field(default_factory=dict)
    static_deflection_m: float = 0.0
    limit_s: float = float("inf")


@dataclass
class CraftFleet:
    """Every craft's body, in one call of each law.

    THE CRAFT STEP TOGETHER, and that is not only an optimisation. They
    share a world, so they share its floor: the substep count is set by
    the TIGHTEST stability limit in the fleet and everybody takes it.
    Letting each craft pick its own would mean two bodies advancing
    different amounts of world time between the same pair of engine
    rounds, which is the drift this whole demo exists to expose rather
    than to create.

    Both laws are batch kernels lowered by the repository
    (`native_law_kernels.law_kernel`, sympy -> AbstractTensor -> SSA) and
    prepared once, so a column is the kernel's own memory. Writing an
    input for every craft is one array assignment; carrying 90 states
    from outputs back to inputs is 90 assignments for the whole fleet
    rather than 90 per craft.
    """

    craft: dict
    vehicle: object          # NativeLaw, or FannedLaw if refused
    contact: object
    #: (output name, input name) for every state the law advances
    feedback: list
    contact_defaults: dict
    mass_kg: float
    values: dict
    substeps: int = 1
    limit_s: float = float("inf")
    rolling_radius_m: float = BASIC_CRAFT_ROLLING_RADIUS_M
    #: the radial effective mass the contact law is given, so the mode it
    #: forms can be read back out of what it publishes
    RADIAL_MASS: float = 6.0
    calls: int = 0
    wall_s: float = 0.0
    world_s: float = 0.0
    state_span: object = None

    # -- addressing ----------------------------------------------------
    def column(self, name: str):
        return self.vehicle.column(name)

    def put(self, name: str, value, row=None) -> None:
        self.vehicle.put(name, value, row)

    def got(self, name: str, row: int = 0) -> float:
        return self.vehicle.get(name, row)

    def out(self, name: str):
        return self.vehicle.out(name)

    def reset_ledger(self) -> None:
        self.calls = 0
        self.wall_s = self.world_s = 0.0
        self.vehicle.reset_ledger()
        self.contact.reset_ledger()

    @property
    def tau(self) -> float:
        """World seconds these bodies advanced per wall second spent."""
        return self.world_s / self.wall_s if self.wall_s > 0.0 else 0.0

    # -- the declared span ---------------------------------------------
    def sync_state_span(self):
        """Every craft's law inputs, concatenated.

        The columns already ARE the bodies' state -- one value per port
        per craft -- so the span is that, not a new representation
        invented for the compiler.
        """
        self.state_span = np.concatenate(
            [np.asarray(self.vehicle.column(n)) for n in self.vehicle.input_names])
        return self.state_span

    def load_state_span(self, span=None) -> None:
        source = self.state_span if span is None else span
        if source is None:
            return
        flat = np.asarray(source, dtype=np.float64).reshape(-1)
        width = self.vehicle.batch
        for index, name in enumerate(self.vehicle.input_names):
            self.vehicle.column(name)[:] = flat[index * width:(index + 1) * width]

    # -- the contact law, all wheels of all craft in one call ----------
    def contact_rows(self, craft_row: int):
        """The slice of contact rows belonging to one craft."""
        start = craft_row * len(WHEEL_NAMES)
        return slice(start, start + len(WHEEL_NAMES))

    def load_contact_defaults(self) -> None:
        for key, value in self.contact_defaults.items():
            self.contact.put(key, float(value))

    def probe_normal_force(self, compression) -> object:
        """Every craft's corner through the contact law at stated depths."""
        self.load_contact_defaults()
        self.contact.put("dt", 1.0e-4)   # well inside the law's radial mode
        self.contact.put("tire_radial_compression", compression)
        self.contact.run()
        return np.asarray(self.contact.out("chassis_force_y")).copy()

    def settle_on_its_tires(self) -> None:
        """Find the compression that carries a corner, BY ASKING THE LAW.

        A craft whose tires start at zero compression makes no contact
        force, so it is never supported, so nothing ever compresses a
        tire. It has to begin where a parked craft actually sits.

        That depth is bisected out of the contact law itself rather than
        divided out of a stiffness. There is no radial k*x tire spring to
        divide by -- the law says so -- so a stiffness picked to stand in
        for one is a number with no referent.

        Bisecting the whole fleet at once costs 28 kernel calls for every
        craft there is, instead of 28 per craft.
        """
        corner_load = self.mass_kg * 9.81 / len(WHEEL_NAMES)
        rows = self.contact.batch
        low = np.zeros(rows)
        high = np.full(rows, self.contact_defaults["tire_section_radius"] * 1.6)
        for _ in range(28):
            mid = 0.5 * (low + high)
            carried = self.probe_normal_force(mid) >= corner_load
            high = np.where(carried, mid, high)
            low = np.where(carried, low, mid)
        settled = 0.5 * (low + high)
        for craft in self.craft.values():
            depth = float(settled[self.contact_rows(craft.row)].min())
            craft.static_deflection_m = depth
            for wheel in WHEEL_NAMES:
                self.put(f"compression_{wheel}", depth, craft.row)

    def stability_limit_s(self) -> float:
        """The contact law's own limit, from what it published.

        The contact response is a real radial mode: the law forms a
        pneumatic stiffness from the patch it is standing on and rings it
        down through implicit midpoint stages. An explicit step cannot
        outrun that mode, so the limit is `2/sqrt(k/m)`.

        Taken from the law's OUTPUTS -- the normal force it produced and
        the compression that produced it are a secant stiffness -- rather
        than by recomputing its patch geometry and gas law in Python. A
        second copy of one law is two things to keep in step, and they do
        not stay in step.

        THE FLEET TAKES THE TIGHTEST. A shared world has one floor.
        """
        tightest = float("inf")
        for craft in self.craft.values():
            force = abs(craft.last.get("contact_force_y", 0.0))
            depth = craft.last.get("compression_m", 0.0)
            craft.limit_s = (float("inf") if force <= 0.0 or depth <= 0.0
                             else 2.0 / math.sqrt((force / depth) / self.RADIAL_MASS))
            tightest = min(tightest, craft.limit_s)
        return tightest

    def publish_slip(self) -> None:
        """Wheel speed against ground speed, for every wheel of every craft.

        Longitudinal slip ratio in the usual sense: the surface speed of
        the tire minus the speed the hub is actually travelling, over
        whichever is larger. This is the quantity the law consumes and
        cannot form, because it does not know how big its wheels are.
        """
        body = np.abs(np.asarray(self.column("velocity_x")))
        for wheel in WHEEL_NAMES:
            surface = (np.asarray(self.column(f"wheel_omega_{wheel}"))
                       * self.rolling_radius_m)
            reference = np.maximum(np.maximum(np.abs(surface), body), 0.5)
            self.column(f"slip_longitudinal_{wheel}")[:] = (
                (surface - np.asarray(self.column("velocity_x"))) / reference)

    def contact_wrench(self, dt: float):
        """Every wheel of every craft through the contact law, in ONE call.

        Returns the force, torque and patch area summed per craft.
        """
        self.load_contact_defaults()
        self.contact.put("dt", dt)
        floor = np.array([self.craft_by_row[r // len(WHEEL_NAMES)].static_deflection_m
                          for r in range(self.contact.batch)])
        for index, wheel in enumerate(WHEEL_NAMES):
            rows = slice(index, None, len(WHEEL_NAMES))
            depth = np.asarray(self.out(f"compression_{wheel}_next")).copy()
            self.contact.column("tire_radial_compression")[rows] = np.where(
                depth > 0.0, depth, floor[rows])
            self.contact.column("slip_longitudinal")[rows] = np.asarray(
                self.out(f"slip_longitudinal_{wheel}_next"))
            self.contact.column("tire_radial_velocity")[rows] = np.asarray(
                self.out(f"compression_velocity_{wheel}_next"))
            for port, produced in (
                    ("sidewall_deformation_longitudinal",
                     f"tire_deformation_longitudinal_{wheel}_next"),
                    ("sidewall_deformation_lateral",
                     f"tire_deformation_lateral_{wheel}_next"),
                    ("sidewall_deformation_velocity_longitudinal",
                     f"tire_deformation_velocity_longitudinal_{wheel}_next"),
                    ("sidewall_deformation_velocity_lateral",
                     f"tire_deformation_velocity_lateral_{wheel}_next")):
                column = self.contact.column(port)
                if column is not None:
                    column[rows] = np.asarray(self.out(produced))
        self.contact.run()

        width = len(WHEEL_NAMES)
        force = np.empty((len(self.craft), 3))
        torque = np.empty((len(self.craft), 3))
        area = np.empty(len(self.craft))
        axes = ("x", "y", "z")
        for craft in self.craft.values():
            rows = self.contact_rows(craft.row)
            for axis, letter in enumerate(axes):
                force[craft.row, axis] = float(
                    np.asarray(self.contact.out(f"chassis_force_{letter}"))[rows].sum())
                torque[craft.row, axis] = float(
                    np.asarray(self.contact.out(f"chassis_torque_{letter}"))[rows].sum())
            area[craft.row] = float(
                np.asarray(self.contact.out("contact_area"))[rows].sum())
        _ = width
        return force, torque, area

    def carry(self) -> None:
        """Advance every state the law published, for the whole fleet.

        One array assignment per carried state, not one per state per
        craft. The two exclusions are the same two they always were.
        """
        for produced, into in self.feedback:
            self.vehicle.column(into)[:] = np.asarray(self.vehicle.out(produced))

    def lowest_compression(self, craft: Craft) -> float:
        depths = [self.got(f"compression_{w}_next", craft.row) for w in WHEEL_NAMES]
        depths = [d if d > 0.0 else craft.static_deflection_m for d in depths]
        return min(depths)

    # -- one tick of every body ----------------------------------------
    def step(self, dt: float, *, shaft_omega: dict, throttle: dict,
             brake: dict | None = None) -> dict:
        started = time.perf_counter()
        brake = brake or {}
        for name, craft in self.craft.items():
            self.put("engine_angular_speed", float(shaft_omega.get(name, 0.0)),
                     craft.row)
            self.put("throttle", float(throttle.get(name, 0.0)), craft.row)
            self.put("brake", float(brake.get(name, 0.0)), craft.row)
        self.put("drive_direction", 1.0)

        # THE STABILITY LIMIT DECIDES THE STEP. The contact law publishes
        # a mode it cannot be stepped slower than; honour it by dividing
        # the window until every piece of it is inside the limit.
        limit = self.stability_limit_s()
        self.substeps = 1 if limit >= dt else max(1, int(math.ceil(dt / limit)))
        self.limit_s = limit
        step_dt = dt / self.substeps
        self.put("dt", step_dt)

        for _ in range(self.substeps):
            self.publish_slip()
            self.vehicle.run()
            force, torque, area = self.contact_wrench(step_dt)
            for axis, letter in enumerate("xyz"):
                self.column(f"contact_wrench_force_{letter}")[:] = force[:, axis]
                self.column(f"contact_wrench_torque_{letter}")[:] = torque[:, axis]
            self.vehicle.run()
            self.carry()
            for wheel in WHEEL_NAMES:
                self.column(f"previous_slip_longitudinal_{wheel}")[:] = np.asarray(
                    self.out(f"slip_longitudinal_{wheel}_next"))
            for craft in self.craft.values():
                craft.last["contact_force_y"] = float(force[craft.row, 1])
                craft.last["compression_m"] = self.lowest_compression(craft)

        self.vehicle.run()
        force, torque, area = self.contact_wrench(step_dt)
        for name, craft in self.craft.items():
            craft.last = {
                "clutch_torque_nm": self.got("clutch_torque", craft.row),
                "driveline_torque_nm": self.got("driveline_torque", craft.row),
                "x": self.got("position_x_next", craft.row),
                "z": self.got("position_z_next", craft.row),
                "speed_x": self.got("velocity_x_next", craft.row),
                "speed_z": self.got("velocity_z_next", craft.row),
                "wheel_omega": self.got("wheel_omega_rear_left_next", craft.row),
                "contact_force_x": float(force[craft.row, 0]),
                "contact_force_y": float(force[craft.row, 1]),
                "patch_cm2": float(area[craft.row]) * 1e4,
                "compression_m": self.lowest_compression(craft),
                "limit_ms": craft.limit_s * 1e3,
                "substeps": float(self.substeps),
            }
        self.wall_s += time.perf_counter() - started
        self.world_s += dt
        self.calls += 1
        return {name: craft.last for name, craft in self.craft.items()}

    def last(self, name: str) -> dict:
        return self.craft[name].last


def build_graphs(names: list[str]):
    """One kernel per law for the WHOLE fleet: the law is the same law.

    The batch axis is the fleet -- one row per craft for the body, one
    per wheel of per craft for the contact law -- so a tick is two kernel
    calls however many craft there are, instead of two per craft plus one
    per wheel.

    The producers are passed UNCALLED. Calling one builds the thing the
    kernel cache exists to avoid building: even a HIT on the repository's
    symbolic cache costs 19.6 s for the contact law and 95.2 s for the
    body, of cloudpickle alone.
    """
    fleet_size = len(names)
    vehicle = build_law(compile_symbolic_vehicle_physics,
                        "abstract_ui_vehicle_step", batch=fleet_size)
    contact = build_law(compile_wheel_contact_ssa,
                        "abstract_ui_wheel_contact",
                        batch=fleet_size * len(WHEEL_NAMES))

    produced = set(vehicle.output_names)
    feedback = [(name, name[:-5]) for name in vehicle.output_names
                if name.endswith("_next") and name[:-5] in vehicle.columns
                # the real engine owns its speed, and a filter's output is
                # not its input: `slip_longitudinal_next` is the FILTERED
                # measurement and belongs on `previous_slip_longitudinal`.
                # Carried onto the raw port it closes a loop with no
                # source, decays to zero, and reads exactly like a tire
                # with no grip.
                and name[:-5] != "engine_angular_speed"
                and not name[:-5].startswith("slip_longitudinal_")]
    _ = produced

    parameters = craft_parameters()
    mass_kg = float(parameters["_total_mass_kg"])
    defaults = basic_craft_defaults(load_default_car_configuration())

    fleet = CraftFleet(
        craft={name: Craft(name=name, row=row) for row, name in enumerate(names)},
        vehicle=vehicle, contact=contact, feedback=feedback,
        contact_defaults=defaults, mass_kg=mass_kg, values=dict(parameters))
    fleet.craft_by_row = {craft.row: craft for craft in fleet.craft.values()}

    for key, value in parameters.items():
        if key in vehicle.columns:
            vehicle.put(key, float(value))
    # EACH CRAFT IN ITS OWN LANE. They start superimposed otherwise, and
    # a craft with no steering hardware -- no rack, no knuckles, no tie
    # rods -- can only ever separate by speed, so for a long stretch
    # there is visibly one craft on screen and the other underneath it.
    for craft in fleet.craft.values():
        fleet.put("position_z", 4.0 * craft.row, craft.row)
    fleet.settle_on_its_tires()
    return (fleet, len(vehicle.input_names), len(contact.input_names),
            len(feedback))
