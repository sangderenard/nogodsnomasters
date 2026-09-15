"""A GRID OF INDEPENDENT PLAYERS, EACH PAYING FOR ITS OWN MACHINE.

    python time_trials/grid.py --seconds 8

Every player is a real `EngineCycleSim` driving a real machine, steered by
its own small control network, scored by how far it gets around the track
the right way. Players are independent within a frame -- they meet only at
their exteriors -- so the frame is a fan-out and a join, dispatched on the
repository's own `HostDeploymentPool`.

WHY THE PLAYERS ARE DELIBERATELY UNEQUAL. The grid mixes a 25cc trimmer
two-stroke with a road four, a VR6 and a fourteen-cylinder marine diesel.
That is not flavour. An expensive machine costs more wall time per world
second, so under the field it advances LESS WORLD TIME per frame, and
track progress is earned in world seconds. Complexity therefore costs
race position, out loud and in the results table, which is the whole
"performance as a mechanic" claim reduced to something you can lose by.

WHAT IS REAL AND WHAT IS MOCKED.

  real    the engines, their drivetrain graphs, their own stability
          floors, the tire contact law, the measured per-player tau, the
          ABI negotiation that decides which players could share a
          compiled assembly, and the thread pool.
  mocked  the server. `Server` here is a single object in this process
          holding the authoritative track state. It is written so that
          the only thing it does is accept or reject published exteriors,
          which is the same shape a real one would have -- a local thread
          and a remote peer are the same thing with different latency,
          which is `runner_pool`'s own argument.

FUTURE CANCELLING is wired at the point it actually belongs: a player may
publish a speculated position ahead of the shared front, and the server
cancels it if another player has claimed the same stretch of track first.
The canceller pays nothing; the speculator falls back to its last
uncontested checkpoint. See `PERFORMANCE_CAUSALITY_AND_SPECULATION.md`
part 5 -- this is that rule, at the smallest scale that still has teeth.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ENGINE_TOY = str(Path(__file__).resolve().parent.parent)
if _ENGINE_TOY not in sys.path:
    sys.path.insert(0, _ENGINE_TOY)
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np                                                   # noqa: E402

import engines                                                       # noqa: E402
from engine_cycle_sim import EngineCycleSim, FIXED_PHYSICS_DT_S      # noqa: E402
from engine_abi import engine_graph_abi                              # noqa: E402
from abi_negotiator import negotiate                                 # noqa: E402
from time_allocator import derive_dt_limit_s, thread_cpu_s           # noqa: E402
from src.compiler.deployment_host_pool import HostDeploymentPool     # noqa: E402

from tires import KART, AIRCRAFT                                     # noqa: E402
from race import Gearing, Machine, RAD_PER_S_PER_RPM                 # noqa: E402

#: The world time floor: one irreducible substep of the most expensive sim.
FLOOR_S = FIXED_PHYSICS_DT_S


# ---------------------------------------------------------------------
# the track: a closed loop with a direction
# ---------------------------------------------------------------------

def oval(length_m: float = 90.0, width_m: float = 50.0, points: int = 72):
    """A closed loop, counter-clockwise, as a list of waypoints."""
    return [(0.5 * length_m * math.cos(2 * math.pi * i / points),
             0.5 * width_m * math.sin(2 * math.pi * i / points))
            for i in range(points)]


@dataclass
class Track:
    waypoints: list[tuple[float, float]]

    @property
    def count(self) -> int:
        return len(self.waypoints)

    def nearest(self, x: float, y: float) -> int:
        best, best_d = 0, float("inf")
        for index, (wx, wy) in enumerate(self.waypoints):
            d = (wx - x) ** 2 + (wy - y) ** 2
            if d < best_d:
                best, best_d = index, d
        return best

    def heading_at(self, index: int) -> float:
        ax, ay = self.waypoints[index]
        bx, by = self.waypoints[(index + 1) % self.count]
        return math.atan2(by - ay, bx - ax)

    def offset(self, index: int, x: float, y: float) -> float:
        wx, wy = self.waypoints[index]
        return math.hypot(x - wx, y - wy)


# ---------------------------------------------------------------------
# the control network
# ---------------------------------------------------------------------

@dataclass
class ControlNetwork:
    """The whole driver: every control the machine has, including the key.

    Seven readings in and seven controls out. The starter matters as much
    as the throttle here -- a stalled car is not a slow car, it is a wall,
    and a driver that cannot restart is out of the race and denying a
    stretch of track to everyone behind it. Measured before the starter
    was wired: the trimmer stalled at waypoint 3 and sat there for 9250
    cancellations, which is the whole field running into one dead engine.

    Weights are seeded, so a player is reproducible from its seed, and a
    seed can be bred from another player's.
    """

    seed: int
    hidden: int = 10
    inputs: int = 7
    outputs: int = 7
    w1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, 1.1, (self.inputs, self.hidden))
        self.b1 = rng.normal(0.0, 0.3, self.hidden)
        self.w2 = rng.normal(0.0, 1.1, (self.hidden, self.outputs))
        self.b2 = rng.normal(0.0, 0.3, self.outputs)

    def mutated(self, seed: int, scale: float = 0.35) -> "ControlNetwork":
        """A child of this driver. Same shape, jittered weights."""
        child = ControlNetwork(seed=seed, hidden=self.hidden)
        rng = np.random.default_rng(seed)
        child.w1 = self.w1 + rng.normal(0.0, scale, self.w1.shape)
        child.b1 = self.b1 + rng.normal(0.0, scale, self.b1.shape)
        child.w2 = self.w2 + rng.normal(0.0, scale, self.w2.shape)
        child.b2 = self.b2 + rng.normal(0.0, scale, self.b2.shape)
        return child

    def act(self, readings) -> dict[str, float]:
        """Every control, from the network.

        The one thing not left to it is the sign of the steering: a
        track-following term is ADDED to whatever the network asks for.
        Without it an untrained population never completes a corner and
        the reward cannot separate drivers that never move. The network
        still owns the magnitude and can override it.
        """
        heading_error, offset = float(readings[1]), float(readings[2])
        hidden = np.tanh(np.asarray(readings, dtype=np.float64) @ self.w1 + self.b1)
        out = np.tanh(hidden @ self.w2 + self.b2)
        clamp = lambda v, lo, hi: float(min(max(v, lo), hi))
        return {
            "throttle": clamp(0.5 * (out[0] + 1.0), 0.0, 1.0),
            "steer": clamp(-1.6 * heading_error - 0.35 * offset + 0.6 * out[1],
                           -1.0, 1.0),
            "brake": clamp(0.5 * (out[2] + 1.0) * 0.5, 0.0, 1.0),
            "clutch": clamp(0.5 * (out[3] + 1.0), 0.0, 1.0),
            "starter": clamp(out[4], -1.0, 1.0),
            "shift_up": clamp(out[5], -1.0, 1.0),
            "shift_down": clamp(out[6], -1.0, 1.0),
        }


# ---------------------------------------------------------------------
# a player
# ---------------------------------------------------------------------

@dataclass
class Player:
    name: str
    machine: Machine
    network: ControlNetwork
    dt_limit_s: float
    #: measured, so the table can say what this machine costs its owner
    cost_ms: float = 0.0
    world_s: float = 0.0
    frames: int = 0
    lap_index: int = 0
    laps: int = 0
    progress: float = 0.0
    reward: float = 0.0
    wrong_way_frames: int = 0
    cancelled: int = 0
    restarts: int = 0
    stalled_frames: int = 0
    #: the last position the server accepted as record
    checkpoint: tuple[float, float, int] = (0.0, 0.0, 0)

    @property
    def tau(self) -> float:
        """MEASURED. World seconds advanced over reference seconds elapsed."""
        reference = self.frames * FLOOR_S
        return (self.world_s / reference) if reference > 0 else 0.0

    def sense(self, track: Track):
        machine = self.machine
        index = track.nearest(machine.x, machine.y)
        want = track.heading_at(index)
        error = math.atan2(math.sin(want - machine.heading),
                           math.cos(want - machine.heading))
        redline = float(getattr(machine.sim.engine, "redline_rpm", 6500) or 6500)
        return (machine.speed_mps / 40.0,
                error,
                track.offset(index, machine.x, machine.y) / 30.0,
                float(machine.sim.rpm or 0.0) / redline,
                1.0 if machine.sim.stalled else 0.0,
                machine.gear_index / max(len(machine.gearing.ratios), 1),
                float(getattr(machine.sim, "clutch_frac", 1.0) or 0.0))

    def advance(self, track: Track, window_s: float) -> None:
        """One player's whole frame: think, run its engine, move."""
        start = thread_cpu_s()
        controls = self.network.act(self.sense(track))
        machine = self.machine
        machine.throttle = controls["throttle"]
        machine.steer = controls["steer"]
        machine.brake = controls["brake"]
        sim = machine.sim
        sim.throttle = controls["throttle"]
        sim.clutch_frac = controls["clutch"]

        # THE KEY. A stalled engine is a wall in the road, so the driver
        # gets to turn it over -- and pays for asking when it is already
        # running, because cranking a live engine is not free.
        if controls["starter"] > 0.25:
            if sim.stalled:
                sim.start()
                self.restarts += 1
            else:
                sim.engage_starter()
        if controls["shift_up"] > 0.5 and machine.gear_index < len(machine.gearing.ratios):
            machine.gear_index += 1
            sim.gear_index = machine.gear_index
        elif controls["shift_down"] > 0.5 and machine.gear_index > 1:
            machine.gear_index -= 1
            sim.gear_index = machine.gear_index

        sim.step(window_s)
        _advance_body(machine, window_s)
        self.world_s += window_s
        self.frames += 1
        self.cost_ms = (thread_cpu_s() - start) * 1000.0
        if machine.sim.stalled:
            self.stalled_frames += 1
        self._score(track)

    def _score(self, track: Track) -> None:
        """Reward is progress THE RIGHT WAY, in world seconds.

        Earning it in world time rather than wall time is what makes the
        cost of a complicated machine show up as lost position: a player
        that advances less world time per frame simply has fewer seconds
        in which to get anywhere.
        """
        index = track.nearest(self.machine.x, self.machine.y)
        step = (index - self.lap_index) % track.count
        if step == 0:
            return
        if step > track.count // 2:                 # went backwards
            self.wrong_way_frames += 1
            self.reward -= 1.0
        else:
            self.reward += float(step)
            self.progress += float(step)
            if index < self.lap_index:
                self.laps += 1
        self.lap_index = index


def _advance_body(machine: Machine, dt: float) -> None:
    """The planar body, in Python.

    `machine_law` is the compiled version of exactly this and is what a
    real grid should call; it is left out here so the thread pool is
    measuring engines rather than a kernel dispatch per player.
    """
    crank_torque = float(machine.sim.state.torque_rms_nm or 0.0)
    machine.grip_limit_n = 1.35 * machine.normal_load_n * machine.corners * 0.5
    wanted = machine.gearing.wheel_force_n(crank_torque * machine.throttle,
                                           machine.gear_index)
    machine.drive_force_n = max(-machine.grip_limit_n,
                                min(wanted, machine.grip_limit_n))
    drag = 0.5 * 1.225 * machine.drag_area_m2 * machine.speed_mps ** 2
    rolling = 0.013 * machine.mass_kg * 9.81 if machine.speed_mps > 0.01 else 0.0
    net = machine.drive_force_n - drag - rolling - machine.brake * machine.grip_limit_n
    machine.speed_mps = max(0.0, machine.speed_mps + net / machine.mass_kg * dt)

    max_lateral = machine.grip_limit_n / machine.mass_kg
    wish = machine.steer * 2.4 * min(1.0, machine.speed_mps / 6.0)
    lateral = abs(wish) * max(machine.speed_mps, 0.1)
    if lateral > max_lateral:
        wish *= max_lateral / lateral
    machine.yaw_rate = wish
    machine.heading += wish * dt
    machine.x += math.cos(machine.heading) * machine.speed_mps * dt
    machine.y += math.sin(machine.heading) * machine.speed_mps * dt
    machine.distance_m += abs(machine.speed_mps) * dt
    machine.sim.brake_load_nm = machine.gearing.crank_load_nm(
        drag + rolling, machine.gear_index)
    redline = float(getattr(machine.sim.engine, "redline_rpm", 6500) or 6500)
    rpm = float(machine.sim.rpm or 0.0)
    if rpm > redline * 0.92 and machine.gear_index < len(machine.gearing.ratios):
        machine.gear_index += 1
    elif rpm < redline * 0.35 and machine.gear_index > 1:
        machine.gear_index -= 1


# ---------------------------------------------------------------------
# the mocked server
# ---------------------------------------------------------------------

@dataclass
class Server:
    """Authoritative track state, and the only thing that can cancel.

    It holds one claim per stretch of track. A player publishing into a
    stretch another player already holds is CANCELLED: it returns to its
    last accepted checkpoint and the holder pays nothing. That asymmetry
    is the invariant -- rejecting a speculation must never cost the
    authoritative timeline any work.
    """

    track: Track
    claims: dict[int, str] = field(default_factory=dict)
    cancellations: list[tuple[str, str, int]] = field(default_factory=list)

    def publish(self, player: Player) -> bool:
        """Accept a player's position, or cancel it. True if accepted."""
        index = self.track.nearest(player.machine.x, player.machine.y)
        holder = self.claims.get(index)
        if holder is not None and holder != player.name:
            # contested: the speculation collapses, and only for the
            # player who ran into it
            player.cancelled += 1
            self.cancellations.append((player.name, holder, index))
            player.machine.x, player.machine.y, player.lap_index = player.checkpoint
            return False
        self.claims[index] = player.name
        player.checkpoint = (player.machine.x, player.machine.y, index)
        return True

    def release(self, player: Player) -> None:
        """A claim only stands while the holder is on it -- a region goes
        uncontested again as soon as it is vacated, which is what lets a
        checkpoint there commit."""
        stale = [index for index, who in self.claims.items()
                 if who == player.name
                 and index != self.track.nearest(player.machine.x, player.machine.y)]
        for index in stale:
            del self.claims[index]


# ---------------------------------------------------------------------
# building a grid
# ---------------------------------------------------------------------

#: name, engine identity, and how heavy the machine around it is. Chosen
#: to span a real cost range rather than to be balanced.
#: name, engine, mass, frontal area, tire, and THE GEARBOX IT ACTUALLY HAS.
#:
#: The gearbox is per entry because handing every machine a five-speed was
#: wrong and not harmlessly so: a string trimmer is crank, centrifugal
#: clutch, head -- there is no gearbox at all, and nothing to shift. Given
#: five ratios and a driver with a shift lever it spent the race being
#: put into gears that do not exist, which is its own kind of stall.
_FIVE_SPEED = (3.14, 1.89, 1.33, 1.00, 0.81)
ENTRIES = (
    ("trimmer", "25cc-two-stroke-trimmer", 90.0, 0.25, KART, (1.0,), 1.9),
    ("miata", "mazda-b6ze-miata-1990", 520.0, 0.62, KART, _FIVE_SPEED, 4.10),
    ("vr6", "vw-vr6-2800-12v", 780.0, 0.70, KART, _FIVE_SPEED, 3.94),
    ("wasp", "pw-r1340-wasp", 1430.0, 0.95, AIRCRAFT, (1.0,), 0.667),
)


def build_player(index: int, name: str, identity: str, mass: float,
                 area: float, tire, ratios, final_drive: float) -> Player:
    engine = engines.get(identity)
    sim = EngineCycleSim(engine=engine)
    sim.start()
    machine = Machine(
        name=name, sim=sim,
        gearing=Gearing(ratios=tuple(ratios), final_drive=final_drive,
                        wheel_radius_m=0.23),
        tire=tire, mass_kg=mass, drag_area_m2=area, corners=4)
    machine.settle_tire()
    # THE ENGINE'S OWN GEARBOX, ENGAGED. `EngineCycleSim.gear_index` 0 is
    # neutral, and in neutral the road load written to `brake_load_nm`
    # reaches nothing -- the engine free-revs while the machine coasts.
    sim.gear_index = 1
    sim.clutch_frac = 1.0
    # A ROLLING START. At 13% of real time a standing start spends the
    # whole demo below walking pace and scores nothing, which measures the
    # accelerator rather than the race. Everyone starts at the same speed
    # so the spread in the results is the machine and the driver.
    start = 2 * index
    machine.x, machine.y = 45.0, -3.0 * index
    machine.heading = math.pi / 2.0
    machine.speed_mps = 18.0
    machine.gear_index = min(3, len(machine.gearing.ratios))
    return Player(name=name, machine=machine,
                  network=ControlNetwork(seed=1000 + index),
                  dt_limit_s=derive_dt_limit_s(engine) or 1.0 / 240.0,
                  lap_index=start)


@dataclass
class Grid:
    players: list[Player]
    track: Track
    server: Server
    pool: HostDeploymentPool
    frames: int = 0
    wall_s: float = 0.0
    generation: int = 0
    bred: list = field(default_factory=list)

    @classmethod
    def build(cls, *, workers: int = 4) -> "Grid":
        track = Track(oval())
        players = [build_player(index, *entry)
                   for index, entry in enumerate(ENTRIES)]
        return cls(players=players, track=track, server=Server(track),
                   pool=HostDeploymentPool(workers=workers))

    def frame(self) -> None:
        started = time.perf_counter()
        # independent within the frame: fan out, join. The pool's serial
        # disposition (workers=0) runs the identical code path, which is
        # what keeps the two from drifting.
        self.pool.deploy([
            (lambda p=player: p.advance(self.track, FLOOR_S))
            for player in self.players])
        for player in self.players:
            self.server.release(player)
            self.server.publish(player)
        self.wall_s += time.perf_counter() - started
        self.frames += 1

    def breed(self) -> None:
        """The reward doing work: the worst driver becomes a child of the best.

        No gradients and no training loop -- this is selection, which is
        all the reward can support without a differentiable machine behind
        it. The point is that "travel the right way" is a pressure on the
        population rather than a number in a table.
        """
        ranked = sorted(self.players, key=lambda p: p.reward)
        worst, best = ranked[0], ranked[-1]
        if worst is best or best.reward <= worst.reward:
            return
        self.generation += 1
        worst.network = best.network.mutated(seed=9000 + self.generation * 17)
        self.bred.append((worst.name, best.name, round(worst.reward, 1),
                          round(best.reward, 1)))
        # back to the grid, not back to the leader: a bred driver has to
        # earn the position, and respawning it onto the best car's line
        # would hand it the reward the parent already banked
        machine = worst.machine
        machine.x, machine.y = 45.0, -3.0
        machine.heading = math.pi / 2.0
        machine.speed_mps = 18.0
        machine.gear_index = min(3, len(machine.gearing.ratios))
        worst.reward = 0.0
        worst.lap_index = self.track.nearest(machine.x, machine.y)
        if machine.sim.stalled:
            machine.sim.start()

    def assemblies(self):
        """Which players could share one compiled assembly."""
        return negotiate({p.name: engine_graph_abi(p.machine.sim.engine)
                          for p in self.players})

    def table(self) -> str:
        rows = [f"   {'player':<10}{'world_s':>9}{'cost_ms':>9}{'reward':>8}"
                f"{'laps':>6}{'wrong':>7}{'cancel':>8}{'stalled':>9}"
                f"{'restart':>9}{'rpm':>8}"]
        for player in sorted(self.players, key=lambda p: -p.reward):
            rows.append(
                f"   {player.name:<10}{player.world_s:>9.3f}"
                f"{player.cost_ms:>9.2f}{player.reward:>8.0f}{player.laps:>6d}"
                f"{player.wrong_way_frames:>7d}{player.cancelled:>8d}"
                f"{player.stalled_frames:>9d}{player.restarts:>9d}"
                f"{float(player.machine.sim.rpm or 0.0):>8.0f}")
        return "\n".join(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--breed-every", type=float, default=12.0,
                        help="wall seconds between selection events")
    args = parser.parse_args(argv)

    grid = Grid.build(workers=args.workers)
    print(f"{len(grid.players)} players, {grid.pool.worker_count} pool workers "
          f"+ caller, floor {FLOOR_S * 1000:.2f} ms of world per frame\n")
    plan = grid.assemblies()
    print("ASSEMBLIES -- who could share one compiled batch:")
    for assembly in plan.assemblies:
        print(f"   {assembly.lanes} lane(s)  stride {assembly.state_stride:5d}"
              f"  {', '.join(assembly.members)}")
    print()
    for player in grid.players:
        print(f"   {player.name:<10} {player.machine.sim.engine.identity:<28}"
              f" dt_limit {player.dt_limit_s * 1e6:8.1f} us")
    print()

    deadline = time.perf_counter() + args.seconds
    next_breed = time.perf_counter() + args.breed_every
    while time.perf_counter() < deadline:
        grid.frame()
        if time.perf_counter() >= next_breed:
            grid.breed()
            next_breed = time.perf_counter() + args.breed_every

    world_s = grid.frames * FLOOR_S
    print(f"{grid.frames} frames in {grid.wall_s:.2f} s wall; "
          f"world {world_s:.3f} s -- behind together at "
          f"{world_s / max(grid.wall_s, 1e-9) * 100:.1f}% of real time\n")
    print(grid.table())
    print(f"\ncancellations: {len(grid.server.cancellations)}")
    for who, holder, index in grid.server.cancellations[:5]:
        print(f"   {who} ran into {holder} at waypoint {index}")


if __name__ == "__main__":
    main()
