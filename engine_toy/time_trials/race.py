"""A car and a plane, both running on the repository's own dt cascade.

    python time_trials/race.py --frames 120

Built the way `dt_benchmark.py` builds one, because that is how this
project constructs a dt system out of a series of sims:

    STController + Targets -> GraphBuilder.round(engines=...) -> MetaLoopRunner

Every simulator is a `DtCompatibleEngine` registered into ONE round. Each
returns `Metrics.dt_limit` -- what it can tolerate -- and the controller
takes the minimum instead of each sim substepping privately. Nothing here
schedules anything; the cascade does that.

The engines registered, per vehicle:

    <name>.engine       `dt_benchmark.CycleEngine` around a real
                        `EngineCycleSim` -- crank, cylinders, fuel, heat.
                        Its floor is two degrees of crank.
                        It substeps its OWN stability internally -- fixed
                        1 ms ticks, its drivetrain solver to its own stable
                        sub-dt, and the clutch junction again -- so the
                        drivetrain is NOT registered separately. Doing that
                        advances the same `DrivetrainSolver` twice a frame,
                        because `_step_once` already steps it.
    <name>.chassis      where the vehicle is on the mat. Its floor is the
                        contact patch: a step may not carry the tire
                        further than the patch it is standing on.

The car is a Mazda B6ZE, the plane a Pratt & Whitney R-1340 Wasp. Tire
normal load comes from turing's reduced contact law, integrated over the
tire's own ring stations.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ENGINE_TOY = str(Path(__file__).resolve().parent.parent)
if _ENGINE_TOY not in sys.path:
    sys.path.insert(0, _ENGINE_TOY)

import engines                                                      # noqa: E402
from engine_cycle_sim import (EngineCycleSim, FIXED_PHYSICS_DT_S,    # noqa: E402
                              MAX_CATCHUP_STEPS)
from dt_benchmark import (CycleEngine, use_numpy_backend,            # noqa: E402
                          _metrics)
from src.common.dt_system.dt_controller import STController, Targets  # noqa: E402
from src.common.dt_system.dt_graph import GraphBuilder, MetaLoopRunner  # noqa: E402
from src.common.dt_system.engine_api import (DtCompatibleEngine,     # noqa: E402
                                             EngineRegistration)
from src.common.dt_system.state_table import StateTable              # noqa: E402
from src.common.dt_system.realtime import RealtimeConfig, RealtimeState  # noqa: E402
from time_field import TimeField, TimeFieldConfig                    # noqa: E402

from tires import KART, AIRCRAFT, TireSection                        # noqa: E402

import numpy as np                                                   # noqa: E402
from src.common.tensors import AbstractTensor as AT                  # noqa: E402

RAD_PER_S_PER_RPM = math.pi / 30.0


@dataclass
class Gearing:
    ratios: tuple[float, ...]
    final_drive: float
    wheel_radius_m: float
    efficiency: float = 0.92

    def ratio(self, gear_index: int) -> float:
        if gear_index <= 0 or gear_index > len(self.ratios):
            return 0.0
        return self.ratios[gear_index - 1] * self.final_drive

    def wheel_force_n(self, crank_torque_nm: float, gear_index: int) -> float:
        ratio = self.ratio(gear_index)
        if ratio <= 0.0:
            return 0.0
        return crank_torque_nm * ratio * self.efficiency / self.wheel_radius_m

    def crank_load_nm(self, wheel_force_n: float, gear_index: int) -> float:
        """The road, reflected back to the crank, so the engine answers to
        the track instead of free-revving."""
        ratio = self.ratio(gear_index)
        if ratio <= 0.0:
            return 0.0
        return abs(wheel_force_n) * self.wheel_radius_m / (ratio * self.efficiency)


@dataclass
class Machine:
    """One vehicle: its sim, its tire, and where it is."""

    name: str
    sim: EngineCycleSim
    gearing: Gearing
    tire: TireSection
    mass_kg: float
    drag_area_m2: float
    corners: int = 4
    is_aircraft: bool = False

    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed_mps: float = 0.0
    yaw_rate: float = 0.0
    bank: float = 0.0
    gear_index: int = 1
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0

    normal_load_n: float = 0.0
    patch_area_m2: float = 0.0
    grip_limit_n: float = 0.0
    drive_force_n: float = 0.0
    distance_m: float = 0.0
    world_s: float = 0.0

    def settle_tire(self) -> None:
        """Compression that carries this corner's share, then the load and
        patch the contact law gives at that compression. Bisected, because
        the law is a clamped quadrature with no clean inverse."""
        target = self.mass_kg * 9.81 / max(1, self.corners)
        low, high = 0.0, self.tire.max_compression_m
        for _ in range(24):
            mid = 0.5 * (low + high)
            _area, force = self.tire.contact(mid)
            low, high = (mid, high) if force < target else (low, mid)
        self.patch_area_m2, self.normal_load_n = self.tire.contact(0.5 * (low + high))

    @property
    def patch_length_m(self) -> float:
        return math.sqrt(max(self.patch_area_m2, 1e-9))

    def apply_throttle(self) -> None:
        self.sim.throttle = self.throttle


class MeasuredCycleEngine(CycleEngine):
    """`CycleEngine`, reporting what it ACTUALLY advanced.

    TAU IS NOT A LEVER. Nothing here asks for a rate and nothing sets one.
    A zone is offered a window; its substeppers do what their own stability
    requires; and the world time that came out the other side is a FACT
    about what happened. Tau is that fact divided by the reference. It is
    read, never written.

    `EngineCycleSim` makes the difference measurable exactly. It banks the
    window into `_physics_accum_s`, clamps the bank at
    `FIXED_PHYSICS_DT_S * MAX_CATCHUP_STEPS`, then runs whole 1 ms ticks.
    So for one call:

        banked    = min(accumulator + dt, cap)
        dropped   = (accumulator + dt) - banked      world time DESTROYED
        advanced  = banked - accumulator_after       world time that RAN

    `dropped` is the part that matters and the part nothing reported
    before. When the cap binds, the sim quietly advances less than it was
    asked for and says nothing -- which under this model is not a missing
    log line, it is tau changing with no record of it. A silent cap is the
    field losing track of the truth.
    """

    CAP_S = FIXED_PHYSICS_DT_S * MAX_CATCHUP_STEPS

    def __init__(self, sim, label: str = "engine"):
        super().__init__(sim, label)
        self.asked_s = 0.0
        self.advanced_s = 0.0
        self.dropped_s = 0.0

    def step(self, dt: float, state=None, state_table=None):
        dt = float(dt)
        before = self.sim._physics_accum_s
        banked = min(before + dt, self.CAP_S)
        self.dropped_s += max(0.0, (before + dt) - banked)
        ok, metrics, state = super().step(dt, state, state_table)
        self.advanced_s += banked - self.sim._physics_accum_s
        self.asked_s += dt
        return ok, metrics, state


def build_car() -> Machine:
    sim = EngineCycleSim(engine=engines.get("mazda-b6ze-miata-1990"))
    sim.start()
    machine = Machine(
        name="car", sim=sim,
        gearing=Gearing(ratios=(3.14, 1.89, 1.33, 1.00, 0.81),
                        final_drive=4.10, wheel_radius_m=0.23),
        tire=KART, mass_kg=520.0, drag_area_m2=0.62, corners=4,
        y=-7.0)
    machine.settle_tire()
    return machine


def build_plane() -> Machine:
    sim = EngineCycleSim(engine=engines.get("pw-r1340-wasp"))
    sim.start()
    machine = Machine(
        name="plane", sim=sim,
        gearing=Gearing(ratios=(1.0,), final_drive=0.667, wheel_radius_m=0.26),
        tire=AIRCRAFT, mass_kg=1430.0, drag_area_m2=0.95, corners=3,
        is_aircraft=True, y=7.0)
    machine.settle_tire()
    return machine


@dataclass
class Race:
    """Every sim of every vehicle, in one dt round."""

    machines: list[Machine] = field(default_factory=list)
    #: THE WORLD TIME FLOOR. Substeps come off until what is left is
    #: irreducible -- one substep of the most expensive sim -- and the
    #: world time that single substep covers is the smallest grain the
    #: shared world can advance in. `EngineCycleSim` runs whole
    #: `FIXED_PHYSICS_DT_S` ticks and cannot be asked for less, so for
    #: these machines the floor is 1 ms. Real time is then just real
    #: time: not managed, not targeted, whatever the floor costs.
    dt: float = FIXED_PHYSICS_DT_S
    registrations: list = field(default_factory=list, init=False)
    table: StateTable = field(default_factory=StateTable, init=False)
    builder: GraphBuilder = field(init=False)
    runner: MetaLoopRunner = field(init=False)
    realtime_config: RealtimeConfig = field(init=False)
    realtime_state: object = field(init=False)
    #: The field as a RECORD, not a control signal. Each frame the zones
    #: report what they actually advanced and that ratio is written here.
    #: Nothing reads it back to steer anything.
    field_: TimeField = field(init=False)
    zones: dict = field(default_factory=dict, init=False)
    _last_advanced: dict = field(default_factory=dict, init=False)
    frames: int = field(default=0, init=False)

    @classmethod
    def build(cls, *, dt: float = FIXED_PHYSICS_DT_S) -> "Race":
        use_numpy_backend()
        race = cls(machines=[build_car(), build_plane()], dt=dt)
        targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
        controller = STController(dt_min=1e-7)
        race.builder = GraphBuilder(ctrl=controller, targets=targets, dx=0.1)
        # THE REALTIME LANE, not the scientific one. `run_round` has two
        # paths and they are not a tuning difference. The scientific path
        # wraps every attempt in a `_RoundTransaction`, checkpoints with
        # `copy_shallow()` and restores on reject, so it can roll back --
        # correct for a study, and measured here at 52.6 s for ONE round,
        # of which 44.8 s was `copy.deepcopy` (8.4 million calls, 2653
        # checkpoints, the whole StateTable and the Integrator archive
        # copied per attempt). The physics in that round was about 65 ms.
        #
        # The realtime path takes no checkpoint and never restores: each
        # engine is advanced once per frame on the window
        # `compile_allocations` gave it. Rollback is the capability we are
        # declining, deliberately, because a game does not re-run a frame
        # it did not like.
        # A config is passed to the BUILDER because that is what makes it
        # lay the engines out flat instead of wrapping each in a nested
        # localized round. No `realtime_state` is passed to the RUNNER, and
        # that omission is the point: with no state there are no
        # allocations, so nothing divides a wall-clock budget into
        # per-engine windows and nothing converts milliseconds into
        # seconds. Every engine is handed the same `dt` -- the floor -- and
        # advances it. Whatever rate results is measured afterwards.
        race.realtime_config = RealtimeConfig(budget_ms=1000.0, slack=1.0)
        race.realtime_state = None
        race.runner = MetaLoopRunner(
            realtime_config=race.realtime_config,
            realtime=True, state_table=race.table)
        for machine in race.machines:
            for label, engine in (
                (f"{machine.name}.engine", MeasuredCycleEngine(machine.sim)),
            ):
                race.registrations.append(EngineRegistration(
                    name=label, engine=engine, targets=targets, dx=0.1,
                    localize=True))
                race.zones[label] = engine
                race._last_advanced[label] = 0.0
        race.field_ = TimeField.flat([*race.zones, "reference"],
                                     TimeFieldConfig())
        return race

    def control(self, name: str, *, throttle=None, steer=None, brake=None) -> None:
        for machine in self.machines:
            if machine.name != name:
                continue
            if throttle is not None:
                machine.throttle = max(0.0, min(1.0, float(throttle)))
                machine.apply_throttle()
            if steer is not None:
                machine.steer = max(-1.0, min(1.0, float(steer)))
            if brake is not None:
                machine.brake = max(0.0, min(1.0, float(brake)))

    def frame(self) -> None:
        """One round of the cascade. The controller owns every subdivision."""
        node = self.builder.round(dt=self.dt, engines=self.registrations,
                                  realtime_config=self.realtime_config,
                                  state_table=self.table)
        self.runner.run_round(node, dt=self.dt, state_table=self.table)
        self.frames += 1

        # TAU, MEASURED. Not a target anybody set and not rate limited on
        # the way in: the ramp limit is a statement about what a joint can
        # STAND, so it belongs on the reaction as a diagnostic, not on the
        # measurement as a clamp. `math.inf` here is the same choice
        # `time_diffusion.chain_response` makes for its interior nodes --
        # something other than this call already decided how fast it moved.
        for label, engine in self.zones.items():
            advanced = float(getattr(engine, "advanced_s", 0.0))
            realised = advanced - self._last_advanced[label]
            self._last_advanced[label] = advanced
            tau = realised / self.dt if self.dt > 0.0 else 0.0
            self.field_.set_target(
                label, math.log(max(tau, 1e-9)), self.dt, math.inf)

    def zone_rows(self) -> list[dict]:
        """What each zone was asked for, what it advanced, and what it lost.

        `tau` is the realised rate: world seconds that actually ran, over
        reference seconds that elapsed. `dropped_s` is world time a capped
        substepper destroyed -- asked for and never run, and never
        previously reported by anything.
        """
        reference_s = self.frames * self.dt
        rows = []
        for label, engine in self.zones.items():
            advanced = float(getattr(engine, "advanced_s", 0.0))
            rows.append({
                "zone": label,
                "asked_s": round(float(getattr(engine, "asked_s", 0.0)), 4),
                "advanced_s": round(advanced, 4),
                "dropped_s": round(float(getattr(engine, "dropped_s", 0.0)), 4),
                "tau": round(self.field_.velocity(label), 4),
                "kept_up": round(advanced / reference_s, 4) if reference_s else 0.0,
                "dlog_tau_dt": round(
                    self.field_.dlog_tau_dt[self.field_._index[label]], 4),
            })
        return rows

    def state(self) -> dict:
        return {
            "frame": self.frames,
            "zones": self.zone_rows(),
            "machines": [{
                "name": m.name,
                "x": round(m.x, 3), "y": round(m.y, 3),
                "heading": round(m.heading, 4), "bank": round(m.bank, 4),
                "speed_mps": round(m.speed_mps, 3),
                "gear": m.gear_index,
                "rpm": round(float(m.sim.rpm or 0.0), 1),
                "torque_nm": round(float(m.sim.state.torque_rms_nm or 0.0), 2),
                "power_kw": round(float(m.sim.state.power_rms_kw or 0.0), 2),
                "coolant_k": round(float(m.sim.state.coolant_temp_k or 0.0), 1),
                "drive_force_n": round(m.drive_force_n, 1),
                "grip_limit_n": round(m.grip_limit_n, 1),
                "normal_load_n": round(m.normal_load_n, 1),
                "patch_cm2": round(m.patch_area_m2 * 1e4, 1),
                "distance_m": round(m.distance_m, 2),
                "world_s": round(m.world_s, 4),
            } for m in self.machines],
        }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--dt", type=float, default=FIXED_PHYSICS_DT_S,
                        help="world time floor; one irreducible substep")
    args = parser.parse_args(argv)

    import time
    race = Race.build(dt=args.dt)
    race.control("car", throttle=0.85, steer=0.12)
    race.control("plane", throttle=1.0, steer=0.0)
    for machine in race.machines:
        print(f"   {machine.name:<6} {machine.sim.engine.identity:<24} "
              f"patch {machine.patch_area_m2*1e4:6.1f} cm2 at "
              f"{machine.normal_load_n:7.1f} N/corner")
    print(f"   {len(race.registrations)} sims in one round\n")

    started = time.perf_counter()
    for _ in range(args.frames):
        race.frame()
    elapsed = time.perf_counter() - started

    world_s = args.frames * race.dt
    print(f"{args.frames} rounds in {elapsed:.2f} s wall "
          f"({elapsed / max(args.frames, 1) * 1000:.1f} ms/round), "
          f"floor {race.dt*1000:.2f} ms of world per round")
    print(f"world {world_s:.3f} s against wall {elapsed:.3f} s -- "
          f"behind together at {world_s/elapsed*100:.1f}% of real time")
    for row in race.state()["machines"]:
        print(f"   {row['name']:<6} {row['speed_mps']:6.2f} m/s  gear {row['gear']}"
              f"  rpm {row['rpm']:7.1f}  torque {row['torque_nm']:7.1f} Nm"
              f"  travelled {row['distance_m']:8.2f} m")
    print()
    print(f"   {'zone':<16}{'asked_s':>9}{'advanced_s':>12}{'dropped_s':>11}"
          f"{'tau':>8}{'kept_up':>9}")
    for row in race.zone_rows():
        print(f"   {row['zone']:<16}{row['asked_s']:>9.3f}{row['advanced_s']:>12.3f}"
              f"{row['dropped_s']:>11.3f}{row['tau']:>8.3f}{row['kept_up']:>9.3f}")


if __name__ == "__main__":
    main()
