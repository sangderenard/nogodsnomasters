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
from native_law import NativeLaw                                      # noqa: E402

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
class VehicleGraph:
    """One craft's body: turing's law, its contact law, and nothing of mine.

    Both laws are COMPILED. `columns` is a plain vector in the vehicle
    law's own argument order and `result` a vector in its output order,
    so a tick writes numbers into the kernel's buffers and reads numbers
    back out -- no `AbstractTensor` is built anywhere on this path.
    """

    name: str
    mass_kg: float
    values: dict[str, float]
    columns: object                      # ndarray, one slot per law input
    feedback: list[tuple[int, int]]
    inputs: dict[str, int]
    outputs: dict[str, int]
    vehicle: NativeLaw
    contact: NativeLaw
    contact_names: list[str]
    last: dict[str, float] = field(default_factory=dict)
    #: the craft's tire, declared. The vehicle configuration carries no
    #: tire radius, stiffness or pressure at all -- its only tire
    #: parameters are sidewall deformation frequencies -- so the radial
    #: tire has to be stated here or invented silently.
    contact_defaults: dict = field(default_factory=dict)
    substeps: int = 1
    limit_s: float = float("inf")
    #: the declared flat span -- see `compile_contract.GRAPH_RECORD`
    state_span: object = None
    #: THE WHEEL'S SIZE. The vehicle law has no radius input of any kind --
    #: not wheel, not tire, not rolling -- so it cannot turn wheel speed
    #: into ground speed by itself. It takes `slip_longitudinal` as an
    #: INPUT and expects something outside to compute it; in the validator
    #: that is the roller fixture and the balloon-tire material. Without
    #: it a wheel spinning at 77 rad/s under a stationary craft reports a
    #: slip of exactly zero, so there is no traction and the craft never
    #: moves however much torque reaches the hubs.
    rolling_radius_m: float = BASIC_CRAFT_ROLLING_RADIUS_M
    #: one contact row, reused. The declared tire is written into it once
    #: at build; a wheel then overwrites only the eight ports that differ
    #: between corners, instead of rebuilding forty values per wheel.
    contact_row: object = None
    contact_rest: object = None
    contact_slot: dict = field(default_factory=dict)
    result: object = None
    contact_out: object = None

    # -- the declared span: the law's whole input vector ----------------
    def sync_state_span(self):
        """The 341 law inputs as one contiguous array.

        `columns` already IS the body's state -- one value per port of the
        vehicle law -- so the span is that vector rather than a new
        representation invented for the compiler.
        """
        self.state_span = np.array(self.columns, dtype=np.float64)
        return self.state_span

    def load_state_span(self, span=None) -> None:
        source = self.state_span if span is None else span
        if source is None:
            return
        self.columns[:] = np.asarray(source, dtype=np.float64).reshape(-1)

    def put(self, name: str, value: float) -> None:
        slot = self.inputs.get(name)
        if slot is not None:
            self.columns[slot] = value

    def scalar(self, name: str) -> float:
        slot = self.inputs.get(name)
        return 0.0 if slot is None else float(self.columns[slot])

    def got(self, result, name: str) -> float:
        slot = self.outputs.get(name)
        return 0.0 if slot is None else float(result[slot])

    def run_vehicle(self):
        """The body law, once, on whatever is in `columns`."""
        self.result = self.vehicle.call(self.columns, self.result)
        return self.result

    def settle_on_its_tires(self) -> None:
        """Find the compression that carries a corner, BY ASKING THE LAW.

        A craft whose tires start at zero compression makes no contact
        force, so it is never supported, so nothing ever compresses a
        tire. It has to begin where a parked craft actually sits.

        That depth is bisected out of the contact law itself rather than
        divided out of a stiffness. There is no radial k*x tire spring to
        divide by -- the law says so: "there is no radial k*x tire
        spring" -- so a stiffness picked to stand in for one is a number
        with no referent, which is what was here before.
        """
        corner_load = self.mass_kg * 9.81 / len(WHEEL_NAMES)
        low, high = 0.0, self.contact_defaults["tire_section_radius"] * 1.6
        for _ in range(28):
            mid = 0.5 * (low + high)
            if self.probe_normal_force(mid) < corner_load:
                low = mid
            else:
                high = mid
        self.static_deflection_m = 0.5 * (low + high)
        for wheel in WHEEL_NAMES:
            self.put(f"compression_{wheel}", self.static_deflection_m)
            self.values[f"compression_{wheel}"] = self.static_deflection_m

    def reset_contact_row(self) -> None:
        self.contact_row[:] = self.contact_rest

    def contact_put(self, name: str, value: float) -> None:
        slot = self.contact_slot.get(name)
        if slot is not None:
            self.contact_row[slot] = value

    def probe_normal_force(self, compression: float) -> float:
        """One wheel through the contact law at a stated compression."""
        self.reset_contact_row()
        self.contact_put("dt", 1.0e-4)   # well inside the law's radial mode
        self.contact_put("tire_radial_compression", max(compression, 0.0))
        out = self.contact.call(self.contact_row, self.contact_out)
        return float(out[1])

    #: the radial effective mass the contact law is given, so the mode it
    #: forms can be read back out of what it publishes
    RADIAL_MASS = 6.0

    def stability_limit_s(self) -> float:
        """The contact law's own stability limit, from what it published.

        The contact response is a real radial mode: the law forms a
        pneumatic stiffness from the patch it is standing on and rings it
        down through implicit midpoint stages. An explicit step cannot
        outrun that mode, so the limit is `2/sqrt(k/m)`.

        Taken from the law's OUTPUTS -- the normal force it produced and
        the compression that produced it are a secant stiffness -- rather
        than by recomputing its patch geometry and gas law in Python. A
        second copy of one law is two things to keep in step, and they do
        not stay in step.

        Measured, and sharp: below this the law returns 32-44 kN across
        the working range; above it, exactly 0 N, with nothing in
        between. A craft stepped past it has no ground at all and its
        wheels spin freely, which is what this is here to prevent.
        """
        force = abs(self.last.get("contact_force_y", 0.0))
        depth = self.last.get("compression_m", 0.0)
        if force <= 0.0 or depth <= 0.0:
            return float("inf")
        return 2.0 / math.sqrt((force / depth) / self.RADIAL_MASS)

    def publish_slip(self) -> None:
        """Wheel speed against ground speed, per wheel.

        Longitudinal slip ratio in the usual sense: the surface speed of
        the tire minus the speed the hub is actually travelling, over
        whichever is larger. This is the quantity the law consumes and
        cannot form, because it does not know how big its wheels are.
        """
        body = self.scalar("velocity_x")
        for wheel in WHEEL_NAMES:
            surface = self.scalar(f"wheel_omega_{wheel}") * self.rolling_radius_m
            reference = max(abs(surface), abs(body), 0.5)
            self.put(f"slip_longitudinal_{wheel}", (surface - body) / reference)

    def wheel_compressions(self, result) -> list[float]:
        out = []
        for wheel in WHEEL_NAMES:
            value = self.got(result, f"compression_{wheel}_next")
            out.append(value if value > 0.0
                       else getattr(self, "static_deflection_m", 0.01))
        return out

    def contact_wrench(self, result, dt: float):
        """Every wheel through the real contact law, summed to one wrench."""
        force = [0.0, 0.0, 0.0]
        torque = [0.0, 0.0, 0.0]
        area = 0.0
        for wheel in WHEEL_NAMES:
            self.reset_contact_row()
            compression = self.got(result, f"compression_{wheel}_next")
            if compression <= 0.0:
                compression = getattr(self, "static_deflection_m", 0.01)
            self.contact_put("dt", dt)
            self.contact_put("tire_radial_compression", compression)
            self.contact_put("slip_longitudinal", self.got(
                result, f"slip_longitudinal_{wheel}_next"))
            self.contact_put("tire_radial_velocity", self.got(
                result, f"compression_velocity_{wheel}_next"))
            self.contact_put("sidewall_deformation_longitudinal", self.got(
                result, f"tire_deformation_longitudinal_{wheel}_next"))
            self.contact_put("sidewall_deformation_lateral", self.got(
                result, f"tire_deformation_lateral_{wheel}_next"))
            self.contact_put(
                "sidewall_deformation_velocity_longitudinal", self.got(
                    result,
                    f"tire_deformation_velocity_longitudinal_{wheel}_next"))
            self.contact_put("sidewall_deformation_velocity_lateral", self.got(
                result, f"tire_deformation_velocity_lateral_{wheel}_next"))
            out = self.contact.call(self.contact_row, self.contact_out)
            for axis in range(3):
                force[axis] += float(out[axis])
                torque[axis] += float(out[3 + axis])
            area += float(out[6])
        return force, torque, area

    def step(self, dt: float, *, shaft_omega: float, throttle: float,
             steer: float, brake: float = 0.0) -> dict[str, float]:
        """One tick of this craft's body, given its engine's exterior."""
        self.put("engine_angular_speed", shaft_omega)
        self.put("dt", dt)
        self.put("throttle", throttle)
        self.put("brake", brake)
        self.put("drive_direction", 1.0)

        # THE STABILITY LIMIT DECIDES THE STEP. The contact law publishes
        # a mode it cannot be stepped slower than; honour it by dividing
        # the window until every piece of it is inside the limit.
        limit = self.stability_limit_s()
        self.substeps = 1 if limit >= dt else max(1, int(math.ceil(dt / limit)))
        self.limit_s = limit
        step_dt = dt / self.substeps
        self.put("dt", step_dt)

        raw_slip = {self.inputs.get(f"slip_longitudinal_{w}")
                    for w in WHEEL_NAMES}
        engine_slot = self.inputs.get("engine_angular_speed")
        for _ in range(self.substeps):
            self.publish_slip()
            result = self.run_vehicle()
            force, torque, area = self.contact_wrench(result, step_dt)
            for axis, axis_name in enumerate("xyz"):
                self.put(f"contact_wrench_force_{axis_name}", force[axis])
                self.put(f"contact_wrench_torque_{axis_name}", torque[axis])
            result = self.run_vehicle()
            for out_slot, in_slot in self.feedback:
                if in_slot == engine_slot:
                    continue              # the real engine owns its speed
                if in_slot in raw_slip:
                    # THE FILTER'S OUTPUT IS NOT ITS INPUT.
                    # `slip_longitudinal` is a raw MEASUREMENT the law
                    # runs through a second-order sensor;
                    # `slip_longitudinal_next` is the filtered result and
                    # `previous_slip_longitudinal` is where it belongs.
                    # Carried back onto the raw port it closes a loop with
                    # no source, decays to zero and stays there -- which
                    # reads exactly like a tire with no grip.
                    continue
                self.columns[in_slot] = result[out_slot]
            for wheel in WHEEL_NAMES:
                self.put(f"previous_slip_longitudinal_{wheel}",
                         self.got(result, f"slip_longitudinal_{wheel}_next"))
            self.last["contact_force_y"] = force[1]
            self.last["compression_m"] = min(self.wheel_compressions(result))
        result = self.run_vehicle()
        force, torque, area = self.contact_wrench(result, step_dt)
        self.last = {
            "clutch_torque_nm": self.got(result, "clutch_torque"),
            "driveline_torque_nm": self.got(result, "driveline_torque"),
            "x": self.got(result, "position_x_next"),
            "z": self.got(result, "position_z_next"),
            "speed_x": self.got(result, "velocity_x_next"),
            "speed_z": self.got(result, "velocity_z_next"),
            "wheel_omega": self.got(result, "wheel_omega_rear_left_next"),
            "contact_force_x": force[0],
            "contact_force_y": force[1],
            "patch_cm2": area * 1e4,
            "compression_m": min(self.wheel_compressions(result)),
            "limit_ms": self.limit_s * 1e3,
            "substeps": float(self.substeps),
        }
        return self.last


def build_graphs(names: list[str]):
    """One compile, shared by every craft: the law is the same law.

    Shared all the way down to the kernel's own buffers, which is safe
    only because a call writes every input before it runs -- see
    `NativeLaw.write`. Two craft taking turns in one set of buffers is
    otherwise exactly the kind of quiet cross-talk that reads as physics.
    """
    vehicle = NativeLaw.build(compile_symbolic_vehicle_physics(),
                              "abstract_ui_vehicle_step")
    contact = NativeLaw.build(compile_wheel_contact_ssa(),
                              "abstract_ui_wheel_contact")
    vehicle_names = list(vehicle.input_names)
    vehicle_outputs = list(vehicle.output_names)
    contact_names = list(contact.input_names)
    inputs = dict(vehicle.input_slot)
    outputs = dict(vehicle.output_slot)
    feedback = [(slot, inputs[name[:-5]])
                for slot, name in enumerate(vehicle_outputs)
                if name.endswith("_next") and name[:-5] in inputs]

    parameters = craft_parameters()
    mass_kg = float(parameters['_total_mass_kg'])
    defaults = basic_craft_defaults(load_default_car_configuration())
    contact_slot = dict(contact.input_slot)
    rest = np.zeros(len(contact_names))
    for key, value in defaults.items():
        if key in contact_slot:
            rest[contact_slot[key]] = float(value)

    graphs = {}
    for lane, name in enumerate(names):
        values = {key: 0.0 for key in vehicle_names}
        for key, value in parameters.items():
            if key in values:
                values[key] = float(value)
        graph = VehicleGraph(
            name=name, mass_kg=mass_kg, values=values,
            columns=np.array([values[key] for key in vehicle_names]),
            feedback=feedback, inputs=inputs, outputs=outputs,
            vehicle=vehicle, contact=contact, contact_names=contact_names,
            contact_defaults=defaults, contact_row=rest.copy(),
            contact_rest=rest, contact_slot=contact_slot,
            contact_out=np.empty(len(contact.output_names)))
        # EACH CRAFT IN ITS OWN LANE. They start superimposed otherwise,
        # and since the body law has no steering port they can only ever
        # separate by speed -- so for a long stretch there is visibly one
        # craft on screen and the other is underneath it.
        graph.put("position_z", 4.0 * lane)
        graph.values["position_z"] = 4.0 * lane
        graph.settle_on_its_tires()
        graphs[name] = graph
    return graphs, len(vehicle_names), len(contact_names), len(feedback)
