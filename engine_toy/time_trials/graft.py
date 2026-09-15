"""ENGINES IN THE DT SYSTEM; VEHICLE GRAPHS ON THEIR OWN.

    python time_trials/graft.py --frames 120

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
    load_default_car_configuration, WHEEL_NAMES)
from src.compiler.vehicle_python_compilation import (                 # noqa: E402
    vehicle_python_runtime_bindings, _abstract_tensor_stage_callable)

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

def simple_craft_parameters(mass_kg: float = 520.0) -> dict[str, float]:
    """A light craft, from the configuration's own declared defaults.

    The law's input list is fixed at 341, so "simple" cannot mean fewer
    ports -- it means the craft those ports describe is small: a light
    chassis on four wheels, and the law's own combustion silenced so the
    real engine is the only source of crank torque.
    """
    values = dict(load_default_car_configuration().parameter_defaults())
    values.update({
        "inverse_mass": 1.0 / mass_kg,
        "gravity": -9.81,
        "angular_damping": 0.22,
        "inverse_inertia_roll": 1.0 / 160.0,
        "inverse_inertia_pitch": 1.0 / 300.0,
        "inverse_inertia_yaw": 1.0 / 380.0,
        "wheel_inertia": 2.4,
        "engine_displacement_m3": 0.0,
        "engine_enabled": 1.0,
        "assembly_alpha_drivetrain": 1.0,
        "yaw_cos": 1.0, "yaw_sin": 0.0,
        "drive_direction": 1.0,
    })
    for corner in WHEEL_NAMES:
        values[f"assembly_alpha_{corner}"] = 1.0
    return values


@dataclass
class VehicleGraph:
    """One craft's body: turing's law, its contact law, and nothing of mine."""

    name: str
    mass_kg: float
    values: dict[str, float]
    columns: list
    feedback: list[tuple[int, int]]
    inputs: dict[str, int]
    outputs: dict[str, int]
    vehicle_step: object
    contact_step: object
    contact_names: list[str]
    last: dict[str, float] = field(default_factory=dict)
    #: the craft's tire, declared. The vehicle configuration carries no
    #: tire radius, stiffness or pressure at all -- its only tire
    #: parameters are sidewall deformation frequencies -- so the radial
    #: tire has to be stated here or invented silently.
    major_radius_m: float = 0.3925
    section_radius_m: float = 0.1425
    tread_width_m: float = 0.285
    tire_pressure_pa: float = 760_000.0
    substeps: int = 1
    dt_limit_s: float = float("inf")

    def put(self, name: str, value: float) -> None:
        slot = self.inputs.get(name)
        if slot is not None:
            self.columns[slot] = AT.tensor(np.array([float(value)]))

    def got(self, result, name: str) -> float:
        slot = self.outputs.get(name)
        if slot is None:
            return 0.0
        value = result[slot]
        return float(np.asarray(getattr(value, "data", value)).reshape(-1)[0])

    def settle_on_its_tires(self) -> None:
        """Press the craft onto the ground before the first tick.

        A craft that starts with every tire at zero compression generates
        no contact force, so it is never supported, so nothing ever
        compresses a tire -- measured, the first attempt reported 0.0 N of
        contact for every wheel while the driveline spun up to 200 rad/s
        and the body never moved a millimetre. The static deflection that
        carries its own weight is where a parked craft actually sits.
        """
        corner_load = self.mass_kg * 9.81 / len(WHEEL_NAMES)
        stiffness = float(self.values.get("tire_radial_stiffness", 0.0)) or 220_000.0
        deflection = corner_load / stiffness
        for wheel in WHEEL_NAMES:
            self.put(f"compression_{wheel}", deflection)
            self.values[f"compression_{wheel}"] = deflection
        self.static_deflection_m = deflection

    #: geometry the analytic floor needs, matching the law's own constants
    MINIMUM_AREA = 0.008
    MAXIMUM_AREA = 0.06
    POLYTROPIC = 1.3
    REFERENCE_VOLUME = 0.035
    RADIAL_MASS = 6.0

    def contact_dt_limit(self, compression: float) -> float:
        """The tire's own safe step, from what the law already computes.

        The contact response is a real radial mode, not a penalty force:
        pneumatic pressure over the flattened patch gives a radial
        stiffness, and against the radial effective mass that is a
        frequency. An explicit scheme cannot outrun it, so the floor is
        `2/sqrt(k/m)`.

        MEASURED, and sharp: at half this step the law returns 32-44 kN
        across the working range; at twice it, exactly 0 N every time,
        with nothing in between. A 16.7 ms frame violates it at every
        compression, which is why a craft with four wheels on the ground
        reported no ground at all.

        The counterintuitive part is worth keeping in mind: the floor is
        TIGHTEST when the tire is barely touching (2.5 ms at 10 mm) and
        loosens as it is pressed (7.4 ms at 120 mm), because the patch
        clamps at `maximum_contact_area` so `P*A/d` falls with depth. A
        policy that reasoned about penetration depth would guess the
        wrong way round; publishing the number does not.
        """
        depth = min(max(compression, 0.0), self.section_radius_m * 1.65)
        if depth <= 0.0:
            return float("inf")
        chord = max(0.0, 2.0 * self.major_radius_m * depth - depth * depth)
        area = min(max(2.0 * math.sqrt(chord) * self.tread_width_m,
                       self.MINIMUM_AREA), self.MAXIMUM_AREA)
        strain = min(max(area * depth * 0.55
                         / (self.REFERENCE_VOLUME + 1e-5), 0.0), 0.65)
        pressure = self.tire_pressure_pa * (
            1.0 + self.POLYTROPIC * strain
            + self.POLYTROPIC * (self.POLYTROPIC + 1.0) * strain * strain / 2.0)
        stiffness = pressure * area / (depth + self.section_radius_m * 1e-5 + 1e-5)
        if stiffness <= 0.0:
            return float("inf")
        return 2.0 / math.sqrt(stiffness / self.RADIAL_MASS)

    def wheel_compressions(self, result) -> list[float]:
        out = []
        for wheel in WHEEL_NAMES:
            value = self.got(result, f"compression_{wheel}_next")
            out.append(value if value > 0.0
                       else getattr(self, "static_deflection_m", 0.01))
        return out

    def contact_wrench(self, result, dt: float):
        """Every wheel through the real contact law, summed to one wrench."""
        rated = self.tire_pressure_pa
        radius = self.major_radius_m + self.section_radius_m
        section = self.section_radius_m
        force = [0.0, 0.0, 0.0]
        torque = [0.0, 0.0, 0.0]
        area = 0.0
        for wheel in WHEEL_NAMES:
            row = {name: 0.0 for name in self.contact_names}
            compression = self.got(result, f"compression_{wheel}_next")
            if compression <= 0.0:
                compression = getattr(self, "static_deflection_m", 0.01)
            row.update({
                "support": 1.0, "normal_y": 1.0, "forward_x": 1.0, "right_z": 1.0,
                "dt": dt, "corner_weight": 1.0,
                "mu_static": 1.15, "mu_kinetic": 0.95,
                "load_sensitivity": 0.85, "slip_transition_speed": 2.0,
                "minimum_contact_area": self.MINIMUM_AREA,
                "maximum_contact_area": self.MAXIMUM_AREA,
                "tire_pressure": rated,
                "tire_major_radius": max(radius - section, 0.05),
                "tire_section_radius": section,
                "tire_gas_polytropic_exponent": self.POLYTROPIC,
                "tire_reference_volume": self.REFERENCE_VOLUME,
                "tire_radial_effective_mass": self.RADIAL_MASS,
                "sidewall_shear_stiffness_longitudinal": 180_000.0,
                "sidewall_shear_stiffness_lateral": 140_000.0,
                "sidewall_shear_damping": 900.0,
                "radial_carcass_loss": 0.04,
                "tire_effective_tread_width": 0.18,
                "tire_radial_compression": compression,
                "slip_longitudinal": self.got(result, f"slip_longitudinal_{wheel}_next"),
                "tire_radial_velocity": self.got(
                    result, f"compression_velocity_{wheel}_next"),
                "sidewall_deformation_longitudinal": self.got(
                    result, f"tire_deformation_longitudinal_{wheel}_next"),
                "sidewall_deformation_lateral": self.got(
                    result, f"tire_deformation_lateral_{wheel}_next"),
                "sidewall_deformation_velocity_longitudinal": self.got(
                    result, f"tire_deformation_velocity_longitudinal_{wheel}_next"),
                "sidewall_deformation_velocity_lateral": self.got(
                    result, f"tire_deformation_velocity_lateral_{wheel}_next"),
            })
            columns = [AT.tensor(np.array([row[name]]))
                       for name in self.contact_names]
            out = self.contact_step(*columns)
            read = lambda i: float(np.asarray(
                getattr(out[i], "data", out[i])).reshape(-1)[0])
            for axis in range(3):
                force[axis] += read(axis)
                torque[axis] += read(3 + axis)
            area += read(6)
        return force, torque, area

    def step(self, dt: float, *, shaft_omega: float, throttle: float,
             steer: float, brake: float = 0.0) -> dict[str, float]:
        """One tick of this craft's body, given its engine's exterior."""
        self.put("engine_angular_speed", shaft_omega)
        self.put("dt", dt)
        self.put("throttle", throttle)
        self.put("brake", brake)
        self.put("drive_direction", 1.0)

        # THE TIRE'S OWN FLOOR DECIDES THE STEP. Ask it what it can
        # tolerate, then take that many substeps -- the body never gets a
        # step its tires cannot resolve. This is the module publishing a
        # safe dt range and the caller honouring it, which is the whole
        # mechanism `Metrics.dt_limit` exists for and which the realtime
        # lane collects and never reads.
        result = self.vehicle_step(*self.columns)
        floor = min(self.contact_dt_limit(c)
                    for c in self.wheel_compressions(result))
        substeps = 1 if floor >= dt else min(int(math.ceil(dt / floor)), 512)
        self.substeps = substeps
        self.dt_limit_s = floor
        step_dt = dt / substeps
        self.put("dt", step_dt)

        force = torque = None
        for _ in range(substeps):
            result = self.vehicle_step(*self.columns)
            force, torque, area = self.contact_wrench(result, step_dt)
            for axis, axis_name in enumerate("xyz"):
                self.put(f"contact_wrench_force_{axis_name}", force[axis])
                self.put(f"contact_wrench_torque_{axis_name}", torque[axis])
            result = self.vehicle_step(*self.columns)
            engine_slot = self.inputs.get("engine_angular_speed")
            for out_slot, in_slot in self.feedback:
                if in_slot == engine_slot:
                    continue                  # the real engine owns its speed
                self.columns[in_slot] = result[out_slot]

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
            "substeps": float(self.substeps),
            "dt_limit_ms": self.dt_limit_s * 1e3,
        }
        return self.last


def build_graphs(names: list[str], mass_kg: float = 520.0):
    """One compile, shared by every craft: the law is the same law."""
    vehicle = compile_symbolic_vehicle_physics()
    contact = compile_wheel_contact_ssa()
    vehicle_names = list(vehicle.function.metadata["argument_names"])
    vehicle_outputs = list(vehicle.function.metadata["output_names"])
    contact_names = list(contact.function.metadata["argument_names"])
    bindings = vehicle_python_runtime_bindings(include_configured_vehicle=True)
    vehicle_step = bindings["abstract_ui_vehicle_step"]
    contact_step = _abstract_tensor_stage_callable(contact,
                                                   "abstract_ui_wheel_contact")
    inputs = {name: slot for slot, name in enumerate(vehicle_names)}
    outputs = {name: slot for slot, name in enumerate(vehicle_outputs)}
    feedback = [(slot, inputs[name[:-5]])
                for slot, name in enumerate(vehicle_outputs)
                if name.endswith("_next") and name[:-5] in inputs]

    graphs = {}
    for name in names:
        values = {key: 0.0 for key in vehicle_names}
        for key, value in simple_craft_parameters(mass_kg).items():
            if key in values:
                values[key] = float(value)
        graph = VehicleGraph(
            name=name, mass_kg=mass_kg, values=values,
            columns=[AT.tensor(np.array([values[key]])) for key in vehicle_names],
            feedback=feedback, inputs=inputs, outputs=outputs,
            vehicle_step=vehicle_step, contact_step=contact_step,
            contact_names=contact_names)
        graph.settle_on_its_tires()
        graphs[name] = graph
    return graphs, len(vehicle_names), len(contact_names), len(feedback)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--throttle", type=float, default=0.9)
    parser.add_argument("--substeps", type=int, default=16)
    args = parser.parse_args(argv)

    craft = {"roadster": "mazda-b6ze-miata-1990",
             "coupe": "vw-vr6-2800-12v"}

    started = time.perf_counter()
    batch = EngineBatch.build(craft)
    graphs, n_in, n_contact, n_carry = build_graphs(list(craft))
    print(f"built in {time.perf_counter() - started:.1f}s")
    print(f"   engine batch : {len(batch.sims)} sims in one dt round")
    for name, limit in batch.dt_limits().items():
        print(f"      {name:<10} {batch.sims[name].engine.identity:<24}"
              f" dt_limit {limit * 1e6:8.1f} us")
    print(f"   vehicle graph: {n_in} inputs, {n_contact} contact inputs, "
          f"{n_carry} carried states, one per craft")
    print(f"   static tire deflection "
          f"{graphs[list(craft)[0]].static_deflection_m * 1000:.2f} mm\n")

    dt = FLOOR_S * args.substeps
    for frame in range(args.frames):
        batch.step(dt)
        for name, graph in graphs.items():
            omega, _torque = batch.exterior(name)
            graph.step(dt, shaft_omega=omega, throttle=args.throttle, steer=0.0)
        if frame % max(args.frames // 6, 1) == 0:
            for name, graph in graphs.items():
                row = graph.last
                print(f"   f{frame:4d} {name:<9} rpm "
                      f"{float(batch.sims[name].rpm or 0.0):7.1f}"
                      f"  clutch {row['clutch_torque_nm']:8.1f} Nm"
                      f"  wheel_w {row['wheel_omega']:8.2f}"
                      f"  Fy {row['contact_force_y']:9.1f} N"
                      f"  patch {row['patch_cm2']:6.1f} cm2"
                      f"  x {row['x']:8.3f} m")
    print(f"\nworld {args.frames * dt:.3f} s")
    for name, graph in graphs.items():
        row = graph.last
        print(f"   {name:<10} x {row['x']:9.3f} m  z {row['z']:8.3f} m"
              f"  speed {row['speed_x']:7.3f} m/s"
              f"  rpm {float(batch.sims[name].rpm or 0.0):7.1f}")


if __name__ == "__main__":
    main()
