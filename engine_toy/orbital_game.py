"""Orbital craft, build step 6: the live game.

Decisions: ``turing/docs/ORBITAL_CRAFT_SOLVER_DESIGN_2026-10-02.md``,
decision 6: "clicking on a mass transfers the craft to it, with the
thruster activations animated".

    python orbital_game.py                        interactive window
    python orbital_game.py --click 2 --frames 6 --every 40 --out shot
                                                  headless: select target 2,
                                                  save 6 PNGs 40 frames apart
    python orbital_game.py --click 0 --frames 24 --every 60 --burn-shots
           --out shots/orbital_game_machine      the default click, with a
                                                  shot while an engine burns

Nothing here is new machinery; it is the existing pieces wired together:

    the craft      ``orbital_craft_machine.MachineCraft`` of
                   ``orbital_craft()``: the engine_toy machine (gimballed
                   main engine, two retro "brake" engines, sixteen RCS
                   "navigation" nozzles, three propellant tanks) in its own
                   dt state; it applies its own allocation (throttles AND
                   gimbal commands) when the tracker asks for a wrench
    the targets    ONE batched ``orbital_jumper.OrbitalJumper(batch=N)``
                   of plain massless (``mass_kg``) jumpers on circular orbits
                   about the same ``GravityCenter``: one dt state, one shared
                   dt, advanced over the same windows as the craft, so craft
                   and targets agree at every frame boundary (lockstep).
                   They do not attract the craft (the plan is two-body).
    the plan       ``orbital_plan.hohmann_plan`` (or, through the one
                   ``planner=`` switch, ``orbital_collocation.plan_transfer``),
                   phased by the laws below using that plan's actual angular
                   sweep so the craft meets the target, not just its radius
    the flight     ``orbital_tracker.fly``, called once per frame over the
                   frame's window (its own per-round read/decide/allocate/
                   advance loop), with the tracker's own
                   ``hohmann_replanner`` injected so its ON/OFF-PLAN switch
                   and re-plan count are live (read back from the
                   ``FlightReport``)
    the viewer     ``turret_demo.py``'s pattern: a pygame OPENGL window, a
                   plan-view Surface drawn through ``gl_text.TextLayer``,
                   and a headless ``--frames`` mode (hidden window +
                   ``glReadPixels`` -> PNG)

Phasing laws (section PH, composed with ``orbital_plan.KEPLER_LAWS``). The
plan starts at the craft's polar angle ``phase`` and arrives on the target
circle at ``phase + sweep`` after ``t_transfer``. The sweep comes from the
actual plan reference (for Hohmann, ``HohmannPlan.legs`` gives pi). With the
target leading the craft by ``phi_0`` now:

    eq_PH1_1  n_1 = eq_KE1_6 at a = r_1        craft mean motion
    eq_PH1_2  n_2 = eq_KE1_6 at a = r_2        target mean motion
    eq_PH1_3  lead_angle = sweep - n_2 t_transfer
              (the lead the target must have at burn 1 so it is at
              ``phase + sweep`` when the craft is)
    eq_PH1_4  t_wait = Mod((phi_0 - lead_angle) sgn(n_1 - n_2), 2 pi)
                       / |n_1 - n_2|
              (the lead closes at n_1 - n_2; the first non-negative wait)
    eq_PH1_5  phase = theta_0 + n_1 t_wait     the craft's angle at burn 1

Valid while ``r_1 != r_2`` (``PHASING_SCALE``): co-orbital targets have no
synodic period and no Hohmann transfer.

Drawing.  Only the craft's own readings: ``thruster_geometry()`` (each
mount relative to the CURRENT centre of mass, its exhaust at the CURRENT
gimbal state, its delivered throttle), ``centre_of_mass()``,
``tank_propellant_kg()``, ``gimbal_states()`` and the machine's prism.
The live plumes are the delivered throttles at the frame's end; the
group readout also shows what each group fired over the frame, from the
thrust-cost piece's per-thruster impulse (``dI_k / (window * T_k)``).
"""
from __future__ import annotations

import functools
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from honorary_engine_equation_catalogue import LawScale, equation_piece
from machine_package import measure_prism
from orbital_craft_machine import CraftMachine, MachineCraft, orbital_craft
from orbital_jumper import GravityCenter, OrbitalJumper
from orbital_plan import KEPLER_LAWS, hohmann_plan, reference
from orbital_tracker import (TrackingGains, fly, hohmann_replanner,
                             plan_impulses, plan_reference, tracker_mode,
                             tracking_error)
from solution_service import SolutionService

MU_EARTH = 3.986004418e14
R_EARTH = 6.371e6
CRAFT_RADIUS_M = 7.0e6
#: Stations are plain thrusterless jumpers (``mass_kg``); the mass only
#: names the lanes (they do not attract the craft).
TARGET_MASS_KG = 1.0e3
#: Attitude response used by the machine tracker.  The dt system supplies the
#: integration span; the game does not impose a frame-sized timestep.
MACHINE_GAINS = TrackingGains(attitude_frequency_rad_s=0.2)
LENGTH_SCALE_M = 5.0e4
#: The thruster groups by declared ``Thruster.role``, in display order.
THRUSTER_GROUPS = (("main", "main"), ("brake", "retro"),
                   ("navigation", "RCS"))
#: The stations' controller dx (CFL dt <= 0.5 dx / v).  The stations are
#: the truth the craft must meet and nothing corrects them, so they run a
#: finer CFL bound than the tracked craft.
STATION_LENGTH_SCALE_M = 5.0e3
#: The rendezvous is measured this long after the plan's arrival time (six
#: time constants of the default TrackingGains, 1/w = 50 s): the
#: tracker turns the impulsive burn 2 into a finite saturated burn that
#: starts at ``t_burn2``.
ARRIVAL_SETTLE_S = 300.0

# Presentation owns only how quickly accepted simulation time is shown.  These
# are simulation-seconds per wall-second and never select a physics window.
PRESENTATION_FPS = 30.0
TIME_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
                    64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
                    4096.0)
DEFAULT_TIME_MULTIPLIER = 1.0
GUIDANCE_PRESENTATION_MULTIPLIER = 1.0


@dataclass(frozen=True)
class TargetSpec:
    """A station on a circular, prograde orbit in the plan's xy plane."""

    name: str
    radius_m: float
    angle_rad: float


DEFAULT_TARGETS = (
    TargetSpec("Kestrel", 9.0e6, 2.2),
    TargetSpec("Halyard", 1.3e7, -1.0),
    TargetSpec("Tern", 2.0e7, 0.6),
    TargetSpec("Moonhook", 3.0e7, 3.6),
)


# ------------------------------------------------------------ the laws
def phasing_laws() -> dict:
    """Section PH (see the module docstring), composed on eq_KE1_6."""
    mean_motion = KEPLER_LAWS["eq_KE1_6"]
    by_name = {symbol.name: symbol for symbol in mean_motion.rhs.free_symbols}
    a = by_name["a"]
    r_1, r_2 = sp.symbols("r_1 r_2", positive=True)
    n_1, n_2 = sp.symbols("n_1 n_2", positive=True)
    t_transfer = sp.Symbol("t_transfer", positive=True)
    sweep, phi_0, theta_0 = sp.symbols("sweep phi_0 theta_0", real=True)
    lead = sp.Symbol("lead_angle", real=True)
    t_wait = sp.Symbol("t_wait", nonnegative=True)
    closing = n_1 - n_2
    return {
        "eq_PH1_1": sp.Eq(n_1, mean_motion.rhs.xreplace({a: r_1})),
        "eq_PH1_2": sp.Eq(n_2, mean_motion.rhs.xreplace({a: r_2})),
        "eq_PH1_3": sp.Eq(lead, sweep - n_2 * t_transfer),
        "eq_PH1_4": sp.Eq(t_wait, sp.Mod((phi_0 - lead) * sp.sign(closing),
                                         2 * sp.pi) / sp.Abs(closing)),
        "eq_PH1_5": sp.Eq(sp.Symbol("phase", real=True),
                          theta_0 + n_1 * t_wait),
    }


PHASING_LAWS = phasing_laws()
globals().update(PHASING_LAWS)

PHASING_SCALE = LawScale(
    "coplanar circular phasing for a two-burn transfer",
    tuple(PHASING_LAWS),
    (sp.Ne(sp.Symbol("r_1"), sp.Symbol("r_2")),),
    source="synodic phasing of two circular orbits; undefined when the "
           "mean motions are equal (co-orbital)",
)


def _closed(expression):
    """``expression`` with the PH laws' left sides replaced by their right
    sides until none remains (composition, not new physics)."""
    replacements = {law.lhs: law.rhs for law in PHASING_LAWS.values()}
    while True:
        composed = expression.xreplace(replacements)
        if composed == expression:
            return composed
        expression = composed


@functools.lru_cache(maxsize=None)
def _phasing_piece():
    return equation_piece("orbital_game_phasing", tuple(
        sp.Eq(sp.Symbol(name), _closed(PHASING_LAWS[law].rhs),
              evaluate=False)
        for name, law in (("lead_angle", "eq_PH1_3"),
                          ("t_wait", "eq_PH1_4"),
                          ("phase", "eq_PH1_5"))), batch=1)


def _call(piece, columns: dict) -> dict:
    outputs = piece(*(np.ascontiguousarray(np.atleast_1d(columns[name]),
                                           dtype=np.float64)
                      for name in piece.argument_names))
    return {name: float(np.asarray(value).reshape(-1)[0])
            for name, value in zip(piece.output_names, outputs)}


@dataclass(frozen=True)
class Phasing:
    lead_angle: float
    t_wait: float
    phase: float


def phasing(mu: float, r1: float, r2: float, t_transfer: float,
            sweep: float, phi_0: float, theta_0: float) -> Phasing:
    """Evaluate eq_PH1_3..5 (one compiled piece)."""
    if not all(PHASING_SCALE.validity({"r_1": float(r1), "r_2": float(r2)})):
        raise ValueError("co-orbital target: no phasing (r_1 == r_2)")
    values = _call(_phasing_piece(), {
        "mu": mu, "r_1": r1, "r_2": r2, "t_transfer": t_transfer,
        "sweep": sweep, "phi_0": phi_0, "theta_0": theta_0})
    return Phasing(values["lead_angle"], values["t_wait"], values["phase"])


# ------------------------------------------------------------ the planner
def _collocation_planner():
    """Use the collocation module's problem-based planner API."""
    import orbital_collocation

    def planner(problem, t0, position, velocity, *, propellant_kg=None,
                previous=None):
        plan, _hohmann = orbital_collocation.plan_transfer(
            problem, t0, position, velocity, propellant_kg=propellant_kg,
            previous=previous)
        return plan

    return planner


PLANNERS = {
    "hohmann": lambda: hohmann_plan,
    "collocation": _collocation_planner,
}
def _plan_times(plan) -> tuple[float, float, float, float]:
    """Return start, arrival, and first/last thrust times for a plan."""
    impulses = plan_impulses(plan)
    if impulses:
        first_burn, last_burn = (float(impulses[0].time_s),
                                 float(impulses[-1].time_s))
    else:
        start = float(plan.t_start if hasattr(plan, "t_start") else
                      plan.t_burn1)
        arrival = float(plan.t_arrive if hasattr(plan, "t_arrive") else
                        plan.t_burn2)
        first_burn = float(plan.t_burn1 if hasattr(plan, "t_burn1") else
                            start)
        last_burn = float(plan.t_burn2 if hasattr(plan, "t_burn2") else
                          arrival)
    if impulses:
        start = float(plan.t_start if hasattr(plan, "t_start") else
                      first_burn)
        arrival = float(plan.t_arrive if hasattr(plan, "t_arrive") else
                        last_burn)
    return start, arrival, first_burn, last_burn


def _plan_sweep(plan, samples: int = 240) -> float:
    """Unwrapped in-plane angle traversed by the actual planned reference."""
    legs = getattr(plan, "legs", None)
    if callable(legs):
        declared = legs()
        return float(declared[-1][3] - declared[0][3])
    if hasattr(plan, "positions"):
        positions = np.asarray(plan.positions, dtype=float)
    else:
        start, arrival, _first_burn, _last_burn = _plan_times(plan)
        times = np.linspace(start, arrival, max(2, int(samples)))
        positions = reference(plan, times)[0]
    angles = np.unwrap(np.arctan2(positions[:, 1], positions[:, 0]))
    return float(angles[-1] - angles[0])


def _plan_positions(plan, times) -> np.ndarray:
    """Sample positions through each plan's own reference interface."""
    times = np.asarray(times, dtype=float).reshape(-1)
    if hasattr(plan, "positions"):
        return np.stack([plan.reference(float(t))[0] for t in times])
    return reference(plan, times)[0]


@dataclass(frozen=True)
class TransferOrder:
    """A phased transfer to one target, decided at ``ordered_s``."""

    target: int
    ordered_s: float
    plan: object
    phasing: Phasing
    phi_0: float


@dataclass(frozen=True)
class Rendezvous:
    """The craft against the target at the plan's arrival time."""

    target: int
    time_s: float
    distance_m: float
    relative_speed_m_s: float


@dataclass(frozen=True)
class PresentationEndpoint:
    """One accepted endpoint retained for rendering between dt windows."""

    time_s: float
    craft_position_m: np.ndarray
    craft_velocity_m_s: np.ndarray
    station_position_m: np.ndarray
    station_velocity_m_s: np.ndarray
    craft_attitude: np.ndarray
    craft_drawing: object


@dataclass(frozen=True)
class GameView:
    """One immutable publication containing every mutable renderer read."""

    endpoint: PresentationEndpoint
    order_target: int | None
    plan_curve: np.ndarray
    plan_times: tuple | None
    plan_delta_v: str | None
    on_plan: bool
    replans: int
    planner_name: str
    phase_name: str
    plan_deviation_m: float | None
    rendezvous: tuple
    path: tuple
    propellant_kg: float
    propellant_fraction: float
    mass_kg: float
    tanks: tuple
    main_gimbal: tuple | None
    com_shift_m: float
    thruster_groups: tuple
    wheel_actuators: tuple
    dt_proposal_s: float
    dt_blocker: str
    propagation_mode: str
    guidance_signature: tuple
    guidance_active: bool
    announcements: tuple


@dataclass(frozen=True)
class StepRequest:
    generation: int
    outer_window_s: float | None
    selection: int | None = None


@dataclass(frozen=True)
class StepPublication:
    request: StepRequest
    view: GameView | None = None
    error: str | None = None
    wall_s: float = 0.0


def _command_signature(command) -> tuple:
    if command is None:
        return ("idle",)

    def values(value):
        if value is None:
            return ()
        return tuple(np.round(np.asarray(value, dtype=float).reshape(-1), 9))

    return (command.phase, values(command.force_demand_n),
            values(command.torque_demand_n_m), values(command.throttles),
            values(command.gimbal_rad), values(command.wheel_torque_n_m))


def _command_active(command) -> bool:
    signature = _command_signature(command)
    return (command is not None
            and (command.phase != "coast"
                 or any(abs(value) > 1.0e-9 for group in signature[1:]
                        for value in group)))


def _guidance_state(game) -> tuple[tuple, bool]:
    mode = tracker_mode(game.craft)
    command = None if mode is None else (mode.round_command or mode.last)
    signature = _command_signature(command)
    active = _command_active(command)
    return signature, active


def _game_view(game) -> GameView:
    plan = game.flown_plan()
    plan_times = None
    plan_delta_v = None
    curve = np.zeros((0, 3), dtype=float)
    if game.order is not None and plan is not None:
        plan_times = _plan_times(plan)
        curve = np.asarray(game.plan_curve(), dtype=float).copy()
        if hasattr(plan, "dv1"):
            plan_delta_v = f"plan dv {plan.dv1:+.1f} / {plan.dv2:+.1f} m/s"
        else:
            plan_delta_v = f"plan delta-v {plan.ideal_delta_v:.1f} m/s"
    contents = game.craft.tank_propellant_kg()
    tanks = tuple((tank.identity, tank.fluid, float(contents[tank.identity]),
                   float(game.loaded_tank_kg[tank.identity]))
                  for tank in game.craft.craft.tanks)
    guidance_signature, guidance_active = _guidance_state(game)
    return GameView(
        endpoint=_presentation_endpoint(game),
        order_target=(None if game.order is None else game.order.target),
        plan_curve=curve, plan_times=plan_times,
        plan_delta_v=plan_delta_v, on_plan=game.on_plan(),
        replans=game.replans(), planner_name=game.planner_name,
        phase_name=game.phase_name(),
        plan_deviation_m=game.plan_deviation_m,
        rendezvous=tuple(game.rendezvous), path=tuple(game.path),
        propellant_kg=game.propellant_kg,
        propellant_fraction=game.propellant_fraction,
        mass_kg=float(game.craft.mass_kg), tanks=tanks,
        main_gimbal=game.main_gimbal(),
        com_shift_m=float(np.linalg.norm(
            game.craft.centre_of_mass() - game.loaded_centre_of_mass)),
        thruster_groups=tuple(game.thruster_groups()),
        wheel_actuators=tuple(game.wheel_actuators()),
        dt_proposal_s=game.craft.managed_dt_proposal_s,
        dt_blocker=game.craft.managed_dt_blocker,
        propagation_mode=game.craft.propagation_mode,
        guidance_signature=guidance_signature,
        guidance_active=guidance_active,
        announcements=tuple(game.announcements()))


class OrbitalStepService:
    """Single owner for mutable game state; publishes complete snapshots."""

    def __init__(self, game):
        self.game = game
        self.view = _game_view(game)
        self.request: StepRequest | None = None
        self.completed: StepPublication | None = None
        self._generation = 0
        self._solved_generation = 0
        self._wake = threading.Event()
        self._done = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="orbital-step", daemon=True)

    def start(self):
        self._thread.start()
        return self

    @property
    def busy(self) -> bool:
        return (self.request is not None
                and self.request.generation > self._solved_generation)

    def submit(self, outer_window_s: float | None,
               selection: int | None = None) -> StepRequest:
        if self.busy:
            raise RuntimeError("orbital step already in flight")
        self._generation += 1
        request = StepRequest(self._generation, outer_window_s, selection)
        self.request = request
        self._done.clear()
        self._wake.set()
        return request

    def latest_result(self) -> StepPublication | None:
        return self.completed

    def wait(self):
        self._done.wait()

    def stop(self) -> bool:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
        return not self._thread.is_alive()

    def _run(self):
        try:
            while not self._stop.is_set():
                request = self.request
                if (request is None
                        or request.generation == self._solved_generation):
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue
                began = time.perf_counter()
                try:
                    if request.selection is not None:
                        self.game.select(request.selection)
                    if request.outer_window_s is not None:
                        self.game.step(request.outer_window_s)
                    view = _game_view(self.game)
                    publication = StepPublication(
                        request, view=view,
                        wall_s=time.perf_counter() - began)
                except Exception as error:
                    publication = StepPublication(
                        request, error=f"{type(error).__name__}: {error}",
                        wall_s=time.perf_counter() - began)
                if not self._stop.is_set() and self.request is request:
                    self.view = publication.view or self.view
                    self.completed = publication
                    self._solved_generation = request.generation
                    self._done.set()
        finally:
            self.game.close()


def _presentation_endpoint(game) -> PresentationEndpoint:
    craft_position, craft_velocity = game.craft.r()
    station_position, station_velocity = game.station_states()
    return PresentationEndpoint(
        float(game.time_s),
        np.asarray(craft_position, dtype=float).copy(),
        np.asarray(craft_velocity, dtype=float).copy(),
        np.asarray(station_position, dtype=float).copy(),
        np.asarray(station_velocity, dtype=float).copy(),
        np.asarray(game.craft.attitude(), dtype=float).copy(),
        craft_drawing(game.craft, game.loaded_centre_of_mass))


def _hermite_position(p0, v0, p1, v1, span_s: float, fraction: float):
    """Interpolate an accepted trajectory segment using endpoint velocity."""
    u = min(1.0, max(0.0, float(fraction)))
    u2, u3 = u * u, u * u * u
    return ((2.0 * u3 - 3.0 * u2 + 1.0) * p0
            + (u3 - 2.0 * u2 + u) * span_s * v0
            + (-2.0 * u3 + 3.0 * u2) * p1
            + (u3 - u2) * span_s * v1)


class PresentationPacer:
    """Wall-clock pacing and tweening around accepted dt-system endpoints.

    Only one accepted endpoint may be ahead of the displayed state.  If a
    physics window costs more wall time than its display duration, excess
    demand is backpressured rather than accumulated into a later large jump.
    """

    def __init__(self, game, multiplier: float = DEFAULT_TIME_MULTIPLIER):
        multiplier = float(multiplier)
        if (not math.isfinite(multiplier)
                or not TIME_MULTIPLIERS[0] <= multiplier <= TIME_MULTIPLIERS[-1]):
            raise ValueError(
                f"time multiplier must be between {TIME_MULTIPLIERS[0]:g}x "
                f"and {TIME_MULTIPLIERS[-1]:g}x")
        self._multiplier = multiplier
        endpoint = _presentation_endpoint(game)
        self.previous = endpoint
        self.current = endpoint
        self.display_time_s = endpoint.time_s
        self.paused = False
        self.backpressure_s = 0.0

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @property
    def ready(self) -> bool:
        return self.display_time_s >= self.current.time_s - 1.0e-12

    def faster(self):
        self._multiplier = next(
            (rate for rate in TIME_MULTIPLIERS if rate > self.multiplier),
            TIME_MULTIPLIERS[-1])

    def slower(self):
        self._multiplier = next(
            (rate for rate in reversed(TIME_MULTIPLIERS)
             if rate < self.multiplier), TIME_MULTIPLIERS[0])

    def limit_for_guidance(self) -> bool:
        limited = min(self.multiplier, GUIDANCE_PRESENTATION_MULTIPLIER)
        changed = limited < self.multiplier
        self._multiplier = limited
        return changed

    def accept(self, endpoint: PresentationEndpoint):
        self.previous, self.current = self.current, endpoint
        self.display_time_s = max(self.display_time_s, self.previous.time_s)

    def wall_demand(self, elapsed_s: float) -> float:
        return (0.0 if self.paused else
                max(0.0, float(elapsed_s)) * self.multiplier)

    def consume(self, demand_s: float) -> float:
        """Move the display toward its accepted endpoint; return excess."""
        available = max(0.0, self.current.time_s - self.display_time_s)
        shown = min(max(0.0, float(demand_s)), available)
        self.display_time_s += shown
        return max(0.0, float(demand_s) - shown)

    def sample(self) -> PresentationEndpoint:
        span = self.current.time_s - self.previous.time_s
        if span <= 1.0e-12:
            return self.current
        fraction = min(1.0, max(
            0.0, (self.display_time_s - self.previous.time_s) / span))
        return PresentationEndpoint(
            self.display_time_s,
            _hermite_position(self.previous.craft_position_m,
                              self.previous.craft_velocity_m_s,
                              self.current.craft_position_m,
                              self.current.craft_velocity_m_s,
                              span, fraction),
            (1.0 - fraction) * self.previous.craft_velocity_m_s
            + fraction * self.current.craft_velocity_m_s,
            _hermite_position(self.previous.station_position_m,
                              self.previous.station_velocity_m_s,
                              self.current.station_position_m,
                              self.current.station_velocity_m_s,
                              span, fraction),
            (1.0 - fraction) * self.previous.station_velocity_m_s
            + fraction * self.current.station_velocity_m_s,
            _blend_rotation(self.previous.craft_attitude,
                            self.current.craft_attitude, fraction),
            _blend_craft_drawing(self.previous.craft_drawing,
                                 self.current.craft_drawing,
                                 self.previous.craft_attitude,
                                 self.current.craft_attitude, fraction))


def _circular_state(mu: float, radius: float, angle: float):
    speed = math.sqrt(mu / radius)
    return ((radius * math.cos(angle), radius * math.sin(angle), 0.0),
            (-speed * math.sin(angle), speed * math.cos(angle), 0.0))


def _polar(position) -> float:
    return math.atan2(float(position[1]), float(position[0]))


@dataclass(frozen=True)
class PlanningInputs:
    """Main-thread capture: workers never read the craft or station columns."""
    target: int
    target_spec: TargetSpec
    order_id: int
    epoch: int
    time_s: float
    mu: float
    position: tuple
    velocity: tuple
    target_position: tuple
    target_velocity: tuple
    target_time_s: float
    propellant_kg: float
    planner: object
    problem: object = None
    previous: object = None
    previous_owner: object = None
    previous_reference: tuple | None = None
    replan: bool = False


def _solve_planning(request: PlanningInputs):
    """Solve only captured inputs, using the existing two planner APIs."""
    if request.replan:
        if request.problem is None:
            return hohmann_replanner(request.previous, request.time_s,
                                     request.position, request.velocity)
        plan = request.planner(
            request.problem, request.time_s, request.position, request.velocity,
            propellant_kg=request.propellant_kg, previous=request.previous)
    else:
        r1 = float(np.linalg.norm(request.position))
        r2 = float(np.linalg.norm(request.target_position))
        theta = _polar(request.position)
        phi = _polar(request.target_position) - theta
        if request.problem is None:
            probe = request.planner(request.mu, r1, r2)
        else:
            probe = request.planner(request.problem, request.time_s,
                                    request.position, request.velocity,
                                    propellant_kg=request.propellant_kg)
        timing = phasing(request.mu, r1, r2, probe.transfer_time,
                         _plan_sweep(probe), phi, theta)
        if request.problem is None:
            plan = request.planner(request.mu, r1, r2,
                                   t_burn1=request.time_s + timing.t_wait,
                                   phase=timing.phase)
        else:
            position, velocity = _circular_state(request.mu, r1, timing.phase)
            plan = request.planner(request.problem,
                                   request.time_s + timing.t_wait,
                                   position, velocity,
                                   propellant_kg=request.propellant_kg)
        if not getattr(plan, "converged", True):
            raise RuntimeError(f"planner did not converge: {plan.message}")
        return TransferOrder(request.target, request.time_s, plan, timing, phi)
    if not getattr(plan, "converged", True):
        raise RuntimeError(f"planner did not converge: {plan.message}")
    return plan


class _ServiceReplanner:
    """The existing tracker callback, with nonblocking service publication."""
    def __init__(self, game):
        self.game = game

    def __call__(self, previous, time_s, position, velocity):
        self.game._request_replan(previous)
        return None

    def poll(self, previous, time_s, position, velocity):
        return self.game._poll_replan(previous)


class OrbitalGame:
    """The non-graphical game: craft, targets, orders, frames."""

    def __init__(self, *, mu: float = MU_EARTH,
                 craft_radius_m: float = CRAFT_RADIUS_M,
                 craft_angle_rad: float = 0.0,
                 targets=DEFAULT_TARGETS,
                 machine: CraftMachine | None = None,
                 gains: TrackingGains | None = None,
                 round_s: float | None = None,
                 length_scale_m: float = LENGTH_SCALE_M,
                 station_length_scale_m: float = STATION_LENGTH_SCALE_M,
                 planner: str = "hohmann"):
        if planner not in PLANNERS:
            raise ValueError(f"unknown planner {planner!r}; "
                             f"one of {sorted(PLANNERS)}")
        self.mu = float(mu)
        self.planner_name = planner
        self.planner = PLANNERS[planner]()
        self.gains = MACHINE_GAINS if gains is None else gains
        center = GravityCenter((0.0, 0.0, 0.0), self.mu)
        self.center = center
        machine = orbital_craft() if machine is None else machine
        position, velocity = _circular_state(self.mu, craft_radius_m,
                                             craft_angle_rad)
        # An explicit round_s remains a caller-owned outer-window override.
        # Otherwise the opening window is the dt controller's own stability
        # ceiling, dx / max_vel.  After that, step() uses dt_next verbatim.
        initial_window = (float(round_s) if round_s is not None else
                          float(length_scale_m)
                          / float(np.linalg.norm(velocity)))
        self.round_s = initial_window
        self.craft = MachineCraft(
            [center], machine, position_m=position, velocity_m_s=velocity,
            length_scale_m=length_scale_m, window_s=initial_window,
            best_effort_metrics=("orbital_attitude_orthogonality",),
            green_coast=True)
        self.loaded_tank_kg = self.craft.tank_propellant_kg()
        self.loaded_centre_of_mass = self.craft.centre_of_mass()
        self.specs = tuple(targets)
        states = [_circular_state(self.mu, spec.radius_m, spec.angle_rad)
                  for spec in self.specs]
        # ONE dt state: every station is a lane of one batched jumper
        self.stations = OrbitalJumper(
            [center], mass_kg=TARGET_MASS_KG,
            position_m=np.asarray([p for p, _v in states], dtype=float),
            velocity_m_s=np.asarray([v for _p, v in states], dtype=float),
            length_scale_m=station_length_scale_m,
            window_s=initial_window, batch=len(self.specs))
        self.max_thrust = np.asarray(
            [t.max_thrust_n for t in machine.thrusters], dtype=float)
        self.order: TransferOrder | None = None
        self.report = None          # the last frame's FlightReport
        self.rendezvous: list[Rendezvous] = []
        self.plume_throttles = np.zeros(machine.thruster_count)
        self.plan_deviation_m: float | None = None
        self.path: list[tuple[float, float]] = []
        self._record_path()
        self._init_planning()
        self.prepare_planning()
        self._planning_service.start()

    def _init_planning(self):
        self._planning_service = SolutionService(solver=_solve_planning)
        self._pending_plan = None
        self._order_id = 0
        self._planning_epoch = 0
        self._planning_time = self.time_s
        self._planning_closed = False
        self.planning_error = None
        self._announcements = []
        self.replanner = _ServiceReplanner(self)

    def prepare_planning(self):
        """Warm native helpers during setup, before the planning worker starts."""
        from orbital_plan import _plan_piece, _reference_pieces
        from orbital_actuation import throttle_delivery
        _plan_piece()
        _reference_pieces()
        _phasing_piece()
        count = self.craft.design.thruster_count
        throttle_delivery(self.craft.design.thrusters, np.zeros(count),
                          np.zeros(count), 1.0)
        if self.planner_name == "collocation":
            import orbital_collocation as collocation
            problem = collocation.CollocationProblem(
                collocation.pointing_proxy(self.craft.design), (self.center,),
                self.specs[0].radius_m, fuel_budget=self.craft.propellant_kg)
            collocation.prepare_rows(problem)

    def close(self):
        """Stop publication; an in-flight solve cannot start a second worker."""
        self._pending_plan = None
        self._planning_closed = True
        self._planning_service.stop()

    def announcements(self):
        """Drain main-thread acceptance/failure messages exactly once."""
        messages, self._announcements = tuple(self._announcements), []
        return messages

    # ------------------------------------------------------------- reads
    @property
    def time_s(self) -> float:
        return self.craft.time_s

    @property
    def propellant_kg(self) -> float:
        """Propellant left in the craft's tanks (the dt state's columns)."""
        return float(sum(self.craft.tank_propellant_kg().values()))

    @property
    def propellant_fraction(self) -> float:
        loaded = float(sum(self.loaded_tank_kg.values()))
        return 1.0 if loaded <= 0.0 else self.propellant_kg / loaded

    def station_states(self):
        """``(N, 3)`` positions and velocities of every station lane."""
        position, velocity = self.stations.r()
        return (np.asarray(position, dtype=float).reshape(-1, 3),
                np.asarray(velocity, dtype=float).reshape(-1, 3))

    def target_state(self, index: int):
        position, velocity = self.station_states()
        return position[index], velocity[index]

    def flown_plan(self):
        """The plan the tracker flies now: the order's, or its re-plan."""
        if self.order is None:
            return None
        mode = tracker_mode(self.craft)
        if mode is not None and mode.origin is self.order.plan:
            return mode.plan
        return self.order.plan

    def on_plan(self) -> bool:
        """The tracker's ON/OFF-PLAN switch (the last FlightReport)."""
        return True if self.report is None else bool(self.report.on_plan)

    def replans(self) -> int:
        """Full-trip re-plans since the order (the last FlightReport)."""
        return 0 if self.report is None else int(self.report.replans)

    def arrival_s(self) -> float | None:
        """When the active order's rendezvous is measured."""
        if self.order is None:
            return None
        _start, arrival, _first, _last = _plan_times(self.flown_plan())
        return arrival + ARRIVAL_SETTLE_S

    def busy(self) -> bool:
        """An order is still on its way (its rendezvous is ahead)."""
        return (self.order is not None
                and self.time_s < self.arrival_s() - 1.0e-9)

    def phase_name(self) -> str:
        if self.order is None:
            return "planning" if self._pending_plan is not None else "coasting"
        plan, t = self.flown_plan(), self.time_s
        _start, _arrival, first_burn, last_burn = _plan_times(plan)
        if t < first_burn:
            return "phasing wait"
        if t < last_burn:
            return "transfer"
        if t < self.arrival_s():
            return "arrival burn"
        return "on station orbit"

    def plan_curve(self, samples: int = 240) -> np.ndarray:
        """The flown plan's reference from burn 1 to arrival, (samples, 3)."""
        if self.order is None:
            return np.zeros((0, 3))
        plan = self.flown_plan()
        start, arrival, _first_burn, _last_burn = _plan_times(plan)
        times = np.linspace(start, arrival, samples)
        return _plan_positions(plan, times)

    def main_gimbal(self) -> tuple[float, float] | None:
        """The main engine's tilt off its axis (rad): from the craft's
        gimbal STATE, and from the tracker's last command
        (``TrackingCommand.gimbal_rad``; ``nan`` before any command)."""
        mains = self.craft.craft.thrusters_by_role("main")
        if not mains:
            return None
        k = mains[0]
        a, b = self.craft.gimbal_states()[k]
        state = math.acos(min(1.0, math.cos(a) * math.cos(b)))
        command = math.nan
        mode = tracker_mode(self.craft)
        if (mode is not None and mode.last is not None
                and mode.last.gimbal_rad is not None):
            ca, cb = np.asarray(mode.last.gimbal_rad, dtype=float)[k]
            command = math.acos(min(1.0, math.cos(ca) * math.cos(cb)))
        return state, command

    def thruster_groups(self) -> list[tuple[str, int, int, float, float]]:
        """Per declared role: (label, thrusters, firing now, largest live
        delivered throttle, largest throttle fired over the last frame)."""
        live = np.asarray([g.throttle
                           for g in self.craft.thruster_geometry()])
        out = []
        for role, label in THRUSTER_GROUPS:
            ks = self.craft.craft.thrusters_by_role(role)
            if not ks:
                continue
            out.append((label, len(ks),
                        int(np.count_nonzero(live[ks] > 1.0e-3)),
                        float(live[ks].max()),
                        float(np.max(self.plume_throttles[ks]))))
        return out

    def wheel_actuators(self):
        """Live reaction-wheel actuator rows for the flight display."""
        commands = self.craft.wheel_commands()
        momenta = self.craft.wheel_momenta()
        speeds = self.craft.wheel_speeds()
        out = []
        for wheel, command, momentum, speed in zip(
                self.craft.craft.wheels, commands, momenta, speeds):
            dumping = abs(speed) >= wheel.dump_fraction * wheel.max_speed_rad_s
            out.append((wheel, float(command), float(momentum), float(speed),
                        float(wheel.electrical_power_w(command, speed)),
                        dumping))
        return out

    # ------------------------------------------------------------ orders
    def select(self, index: int):
        """Submit a captured transfer request. Flight never waits for its solve."""
        if self.busy():
            raise RuntimeError("transfer in progress")
        if self._planning_closed:
            raise RuntimeError("planning service is stopped")
        if not 0 <= index < len(self.specs):
            raise ValueError("unknown target")
        self._update_planning_epoch()
        self._order_id += 1
        self.planning_error = None
        return self._submit_plan(index)

    def _capture_plan(self, index, previous=None):
        position, velocity = self.craft.r()
        target_position, target_velocity = self.target_state(index)
        propellant = float(self.craft.propellant_kg)
        previous_owner = previous
        previous_reference = (None if previous is None else
                              tuple(tuple(map(float, value)) for value in
                                    plan_reference(previous, self.time_s)))
        problem = None
        if self.planner_name == "collocation":
            import orbital_collocation as collocation
            problem = collocation.CollocationProblem(
                collocation.pointing_proxy(self.craft.design), (self.center,),
                float(np.linalg.norm(target_position)), fuel_budget=propellant)
            previous = collocation.capture_remainder(previous)
        return PlanningInputs(
            index, self.specs[index], self._order_id, self._planning_epoch,
            self.time_s, self.mu, tuple(map(float, position)),
            tuple(map(float, velocity)), tuple(map(float, target_position)),
            tuple(map(float, target_velocity)), float(self.stations.time_s),
            propellant, self.planner, problem=problem, previous=previous,
            previous_owner=previous_owner, previous_reference=previous_reference,
            replan=previous_owner is not None)

    def _submit_plan(self, index, previous=None):
        self._pending_plan = self._planning_service.submit(
            self._capture_plan(index, previous))
        return self._pending_plan

    def _request_replan(self, previous):
        if (not self._planning_closed and self._pending_plan is None
                and self.order is not None):
            self.planning_error = None
            self._submit_plan(self.order.target, previous)

    def _update_planning_epoch(self):
        if self.time_s < self._planning_time:
            self._planning_epoch += 1
        self._planning_time = self.time_s

    def _completed_plan(self):
        self._update_planning_epoch()
        pending = self._pending_plan
        if pending is None:
            return None
        captured = pending.payload
        if (captured.epoch != self._planning_epoch
                or captured.order_id != self._order_id
                or not 0 <= captured.target < len(self.specs)
                or captured.target_spec != self.specs[captured.target]):
            self._pending_plan = None
            self.planning_error = "request changed"
            self._announcements.append("planning result discarded: request changed")
            return None
        done = self._planning_service.latest_result()
        if done is None or done.request.generation != pending.generation:
            return None
        self._pending_plan = None
        if done.error is not None:
            self.planning_error = done.error
            self._announcements.append(
                f"re-plan failed: {done.error}; continuing previous plan"
                if captured.replan else f"planning failed: {done.error}")
            return None
        return captured, done.result

    def _fresh_plan(self, captured, plan):
        now = self.time_s
        start, arrival, first, _last = _plan_times(plan)
        if captured.replan and now >= arrival + ARRIVAL_SETTLE_S:
            return "arrival passed during solve"
        if not captured.replan and now > first:
            return "first burn passed during solve"
        if captured.replan:
            # Transport the captured discrepancy on the previous reference.
            # This tests unchanged assumptions; it is not physical propagation.
            old_r, old_v = plan_reference(captured.previous_owner, now)
            then_r, then_v = captured.previous_reference
            r_ref = np.asarray(captured.position) + old_r - np.asarray(then_r)
            v_ref = np.asarray(captured.velocity) + old_v - np.asarray(then_v)
        elif now < start:
            r1 = float(np.linalg.norm(captured.position))
            angle = _polar(captured.position) + math.sqrt(self.mu / r1**3) * (
                now - captured.time_s)
            r_ref, v_ref = _circular_state(self.mu, r1, angle)
        else:
            r_ref, v_ref = plan_reference(plan, now)
        r, v = self.craft.r()
        if tracking_error(r, v, r_ref, v_ref) >= self.gains.on_plan_threshold:
            return "craft moved outside the ON PLAN band during solve"
        target_radius = float(np.linalg.norm(captured.target_position))
        target_angle = _polar(captured.target_position) + math.sqrt(
            self.mu / target_radius**3) * (
                self.stations.time_s - captured.target_time_s)
        target_ref = _circular_state(self.mu, target_radius, target_angle)
        target_now = self.target_state(captured.target)
        if tracking_error(*target_now, *target_ref) >= self.gains.on_plan_threshold:
            return "target moved outside its captured circular orbit"
        if (captured.problem is not None and captured.problem.burns_propellant
                and self.craft.propellant_kg < float(plan.fuel)):
            return "remaining propellant does not cover the candidate's fuel use"
        return None

    def poll_planning(self):
        """Accept an initial result on the main thread; never waits."""
        if self._pending_plan is None or self._pending_plan.payload.replan:
            return None
        completed = self._completed_plan()
        if completed is None:
            return None
        captured, order = completed
        stale = self._fresh_plan(captured, order.plan)
        if stale:
            self._announcements.append(f"planning result stale: {stale}; retrying")
            self._submit_plan(captured.target)
            return None
        self.order, self.report = order, None
        self._announcements.append(
            f"-> {self.specs[order.target].name}: wait "
            f"{_fmt_time(max(0.0, _plan_times(order.plan)[0] - self.time_s))}, "
            f"transfer {_fmt_time(order.plan.transfer_time)}")
        return order

    def _poll_replan(self, previous):
        if self._pending_plan is None or not self._pending_plan.payload.replan:
            return None
        completed = self._completed_plan()
        if completed is None:
            return None
        captured, plan = completed
        if (self.order is None or captured.target != self.order.target
                or captured.previous_owner is not previous):
            self._announcements.append("re-plan discarded: flown plan changed")
            return None
        stale = self._fresh_plan(captured, plan)
        if stale:
            self._announcements.append(f"re-plan stale: {stale}; retrying")
            self._submit_plan(self.order.target, previous)
            return None
        self._announcements.append("OFF PLAN: completed re-plan accepted")
        return plan

    # ------------------------------------------------------------ steps
    def step(self, outer_window_s: float | None = None) -> float:
        """Advance one requested outer window through the dt coordinator.

        With no override, the request is the dt system's published ``dt_next``.
        The dt system remains responsible for every internal grow, shrink,
        rejection, retry, and accepted substep spanning the requested window.
        """
        self.poll_planning()
        yield_on_guidance_change = outer_window_s is not None
        window = (self.craft.managed_dt_proposal_s
                  if outer_window_s is None else float(outer_window_s))
        if not math.isfinite(window) or window <= 0.0:
            raise ValueError("outer window must be finite and positive")
        arrival = None
        if self.busy():
            arrival = self.arrival_s()
            window = min(window, arrival - self.time_s)
        start = self.time_s
        frame_start = start
        impulses = self.craft.thruster_impulses_n_s
        if self.order is not None:
            plan = self.flown_plan()
            plan_start = (float(plan.t_start)
                          if hasattr(plan, "t_start") else start)
            wait = min(window, max(0.0, plan_start - start))
            if wait > 1.0e-9:
                count = self.craft.design.thruster_count
                self.craft.throttle(np.zeros(count))
                gimbal = getattr(self.craft, "gimbal", None)
                if callable(gimbal):
                    gimbal(np.zeros((count, 2)))
                wheel_torque = getattr(self.craft, "wheel_torque", None)
                if callable(wheel_torque):
                    wheel_torque(np.zeros_like(self.craft.wheel_commands()))
            remaining = wait
            while remaining > 1.0e-9:
                chunk = min(self.craft.managed_dt_proposal_s, remaining)
                advanced, _dt_next, _telemetry = self.craft.advance(chunk)
                advanced = float(advanced)
                reduced = remaining - advanced
                if (not math.isfinite(advanced) or advanced <= 0.0
                        or reduced == remaining):
                    raise RuntimeError(
                        "dt system made no representable progress during coast")
                remaining = max(0.0, reduced)
            window -= wait
            start = self.time_s
            if window > 1.0e-9:
                frame_end = start + window
                deadline = ((lambda: min(frame_end, self.arrival_s()))
                            if arrival is not None else frame_end)
                guidance_yield = None
                if (yield_on_guidance_change
                        and window > (GUIDANCE_PRESENTATION_MULTIPLIER
                                      / PRESENTATION_FPS + 1.0e-12)):
                    guidance_yield = lambda previous, current: (
                        _command_active(current)
                        and _command_signature(current)
                        != _command_signature(previous))
                self.report = fly(self.craft, self.order.plan, self.gains,
                                  until_s=deadline,
                                  round_s=window,
                                  replanner=self.replanner,
                                  yield_on_guidance=guidance_yield)
                self.plan_deviation_m = self.report.final_position_error_m
        else:
            self.craft.advance(window)
        advanced = self.craft.time_s - frame_start
        if advanced > 0.0:
            # every station lane in one round of the one batched dt state
            self.stations.advance(advanced)
            fired = self.craft.thruster_impulses_n_s - impulses
            self.plume_throttles = fired / (advanced * self.max_thrust)
        self._record_path()
        if arrival is not None and self.time_s >= self.arrival_s() - 1.0e-9:
            self._measure_rendezvous()
        self.poll_planning()
        return advanced

    def run_until_arrival(self, window_s: float | None = None) -> Rendezvous:
        """Step until the active order's arrival is measured (no graphics)."""
        if self._planning_closed and self.order is None:
            raise RuntimeError("initial transfer request was cancelled")
        if not self.busy() and self._pending_plan is None:
            raise RuntimeError("no transfer on its way")
        count = len(self.rendezvous)
        stalled = 0
        while len(self.rendezvous) == count:
            before = self.time_s
            self.step(window_s)
            if self.time_s <= before:
                stalled += 1
                # One zero-time return is the intentional guidance publication
                # handoff.  A second one means the requested window cannot move
                # the managed clock and must not become an unbounded wait.
                if stalled >= 2:
                    raise RuntimeError(
                        "dt system made no progress across consecutive "
                        "arrival steps")
            else:
                stalled = 0
            if (len(self.rendezvous) == count and not self.busy()
                    and self._pending_plan is None):
                if self.planning_error is not None:
                    raise RuntimeError(f"planning failed: {self.planning_error}")
                raise RuntimeError("transfer request was cancelled")
        return self.rendezvous[-1]

    def _record_path(self):
        position, _ = self.craft.r()
        self.path.append((float(position[0]), float(position[1])))
        del self.path[:-6000]

    def _measure_rendezvous(self):
        craft_r, craft_v = self.craft.r()
        target_r, target_v = self.target_state(self.order.target)
        self.rendezvous.append(Rendezvous(
            target=self.order.target, time_s=self.time_s,
            distance_m=float(np.linalg.norm(craft_r - target_r)),
            relative_speed_m_s=float(np.linalg.norm(craft_v - target_v))))


# ------------------------------------------------- the craft's declared parts
@dataclass(frozen=True)
class CraftDrawing:
    """The machine craft as the drawing needs it, world-oriented, about a
    point FIXED in the machine (the centre of its prism), so the centre of
    mass visibly moves as the tanks drain.

    ``parts``: ``MachineCraft.thruster_geometry()`` (mount relative to the
    current centre of mass, exhaust at the current gimbal, delivered
    throttle); ``centre_of_mass_m``: where the current centre of mass is
    (add it to a part's mount); ``loaded_centre_of_mass_m``: where it was at
    the declared fill; ``body``: the prism's 8 corners; ``tanks``:
    ``(identity, position, fill fraction of capacity)``."""

    parts: tuple
    centre_of_mass_m: np.ndarray
    loaded_centre_of_mass_m: np.ndarray
    body: np.ndarray
    tanks: tuple


def craft_drawing(craft: MachineCraft,
                  loaded_centre_of_mass=None) -> CraftDrawing:
    """Read the craft's declared geometry and live state (see
    :class:`CraftDrawing`)."""
    rotation = craft.attitude()
    prism = measure_prism(craft.craft.document["nodes"])
    fixed = np.asarray(prism.centre, dtype=float)
    centre = craft.centre_of_mass()
    loaded = centre if loaded_centre_of_mass is None else np.asarray(
        loaded_centre_of_mass, dtype=float)
    size = np.asarray(prism.size_m, dtype=float)
    signs = np.asarray([(x, y, z) for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)], dtype=float)
    contents = craft.tank_propellant_kg()
    tanks = tuple(
        (tank.identity,
         rotation @ (np.asarray(tank.position_m, dtype=float) - fixed),
         0.0 if tank.capacity_kg <= 0.0
         else float(contents[tank.identity]) / tank.capacity_kg)
        for tank in craft.craft.tanks)
    return CraftDrawing(
        parts=tuple(craft.thruster_geometry()),
        centre_of_mass_m=rotation @ (centre - fixed),
        loaded_centre_of_mass_m=rotation @ (loaded - fixed),
        body=(signs * (0.5 * size)) @ rotation.T,
        tanks=tanks)


def _blend_rotation(previous, current, fraction: float) -> np.ndarray:
    """Interpolate two attitudes and project the result back onto SO(3)."""
    u = min(1.0, max(0.0, float(fraction)))
    mixed = ((1.0 - u) * np.asarray(previous, dtype=float)
             + u * np.asarray(current, dtype=float))
    left, _singular, right = np.linalg.svd(mixed)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _blend_craft_drawing(previous: CraftDrawing, current: CraftDrawing,
                         previous_attitude, current_attitude,
                         fraction: float) -> CraftDrawing:
    """Tween all visible craft state carried by two accepted endpoints."""
    u = min(1.0, max(0.0, float(fraction)))
    before = np.asarray(previous_attitude, dtype=float)
    after = np.asarray(current_attitude, dtype=float)
    rotation = _blend_rotation(before, after, u)

    def blend(a, b):
        return (1.0 - u) * np.asarray(a, dtype=float) + u * np.asarray(
            b, dtype=float)

    def local_blend(left, right):
        return blend(before.T @ np.asarray(left, dtype=float),
                     after.T @ np.asarray(right, dtype=float))

    parts = []
    for left, right in zip(previous.parts, current.parts):
        exhaust = rotation @ local_blend(left.exhaust, right.exhaust)
        norm = float(np.linalg.norm(exhaust))
        if norm > 1.0e-12:
            exhaust = exhaust / norm
        else:
            exhaust = np.asarray(right.exhaust, dtype=float).copy()
        parts.append(type(right)(
            identity=right.identity, role=right.role,
            mount_m=rotation @ local_blend(left.mount_m, right.mount_m),
            exhaust=exhaust,
            throttle=(1.0 - u) * left.throttle + u * right.throttle))
    tanks = tuple(
        (right[0], rotation @ local_blend(left[1], right[1]),
         (1.0 - u) * left[2] + u * right[2])
        for left, right in zip(previous.tanks, current.tanks))
    body_local = blend(np.asarray(previous.body) @ before,
                       np.asarray(current.body) @ after)
    return CraftDrawing(
        parts=tuple(parts),
        centre_of_mass_m=rotation @ local_blend(
            previous.centre_of_mass_m, current.centre_of_mass_m),
        loaded_centre_of_mass_m=rotation @ local_blend(
            previous.loaded_centre_of_mass_m,
            current.loaded_centre_of_mass_m),
        body=body_local @ rotation.T, tanks=tanks)


# ================================================================ viewer
WINDOW = (1400, 860)
PLAN_SIZE = 840
PLAN_X, PLAN_Y = 10, 10
PANEL_X = PLAN_SIZE + 30
BG = (14, 15, 18)
INK = (214, 220, 228)
DIM = (120, 128, 138)
AMBER = (232, 176, 72)
GREEN = (120, 210, 150)
RED = (226, 96, 84)
BLUE = (70, 120, 200)
TARGET_COLOURS = ((120, 190, 240), (220, 150, 230), (240, 210, 120),
                  (150, 230, 200), (240, 140, 120), (180, 180, 250))
INSET_SIZE = 230
#: Plume length at full throttle, in body extents.
PLUME_LENGTH_BODY = 1.6


def _hull(points):
    """Convex hull (monotone chain) of 2-D points, for the body outline."""
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) < 3:
        return pts

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


#: Plume length by declared role (fraction of the full plume length).
PLUME_ROLE_SCALE = {"main": 1.0, "brake": 0.6, "navigation": 0.35}
#: The centre of mass moves millimetres (~130 mm drained); the inset draws
#: its shift from the declared fill this many times larger.
COM_SHIFT_MAGNIFICATION = 10.0
TANK_COLOURS = ((120, 200, 255), (240, 170, 90), (180, 240, 140))


def _draw_craft(surface, centre_px, metres_to_px, drawing: CraftDrawing,
                frame, detail: bool = True):
    """Top view (world x right, y up) of the machine craft about a point
    fixed in it: the prism, each thruster mount, a plume from each mount
    along its CURRENT exhaust direction (the main engine's tilts with its
    gimbal) scaled by its delivered throttle; with ``detail`` also the
    tanks (filled to their contents), the centre of mass (moving as the
    tanks drain; its shift from the declared fill drawn
    ``COM_SHIFT_MAGNIFICATION`` times) and where it was at the declared
    fill."""
    import pygame
    cx, cy = centre_px

    def px(vector):
        return (cx + float(vector[0]) * metres_to_px,
                cy - float(vector[1]) * metres_to_px)

    body = drawing.body
    com = drawing.centre_of_mass_m
    extent = max(float(np.max(np.abs(body))) * 2.0, 1.0e-9)
    outline = _hull([px(corner) for corner in body])
    if len(outline) >= 3:
        pygame.draw.polygon(surface, (150, 158, 170) if detail
                            else (205, 212, 222), outline)
        pygame.draw.polygon(surface, (90, 96, 108), outline, 1)
    if detail:
        radius = max(3, int(0.09 * extent * metres_to_px))
        for t, (_identity, position, fill) in enumerate(drawing.tanks):
            centre = px(position)
            point = (int(centre[0]), int(centre[1]))
            pygame.draw.circle(surface, (40, 44, 52), point, radius)
            inner = int(radius * math.sqrt(min(max(fill, 0.0), 1.0)))
            if inner > 0:
                pygame.draw.circle(surface,
                                   TANK_COLOURS[t % len(TANK_COLOURS)],
                                   point, inner)
            pygame.draw.circle(surface, (20, 22, 26), point, radius, 1)
    for part in drawing.parts:
        mount = px(com + part.mount_m)
        pygame.draw.circle(surface, (60, 64, 72) if part.role == "navigation"
                           else (230, 230, 240),
                           (int(mount[0]), int(mount[1])),
                           max(1, int((0.06 if part.role == "navigation"
                                       else 0.1) * extent * metres_to_px)))
    for k, part in enumerate(drawing.parts):
        u = min(max(part.throttle, 0.0), 1.0)
        if u <= 1.0e-3:
            continue
        mount = com + part.mount_m
        flicker = 0.8 + 0.2 * math.sin(frame * 1.9 + 2.3 * k)
        length = (PLUME_LENGTH_BODY * extent * (0.25 + 0.75 * u) * flicker
                  * PLUME_ROLE_SCALE.get(part.role, 0.5))
        in_plane = np.asarray(part.exhaust[:2], dtype=float)
        base = px(mount)
        if float(np.linalg.norm(in_plane)) < 1.0e-6:   # toward/away from view
            pygame.draw.circle(surface, (255, 170, 60),
                               (int(base[0]), int(base[1])),
                               max(2, int(0.3 * length * metres_to_px)), 2)
            continue
        tip = px(mount + part.exhaust * length)
        across = np.asarray((-in_plane[1], in_plane[0]))
        across *= ((0.12 + 0.12 * u) * extent * metres_to_px
                   * PLUME_ROLE_SCALE.get(part.role, 0.5)
                   / max(float(np.linalg.norm(across)), 1.0e-12))
        half = (across[0], -across[1])
        pygame.draw.polygon(surface, (255, 150, 50), [
            (base[0] + half[0], base[1] + half[1]), tip,
            (base[0] - half[0], base[1] - half[1])])
        core = px(mount + part.exhaust * 0.55 * length)
        pygame.draw.polygon(surface, (255, 240, 170), [
            (base[0] + 0.5 * half[0], base[1] + 0.5 * half[1]), core,
            (base[0] - 0.5 * half[0], base[1] - 0.5 * half[1])])
    if detail:
        # the shift is millimetres on a metres-sized craft: drawn magnified
        loaded = px(drawing.loaded_centre_of_mass_m)
        here = px(drawing.loaded_centre_of_mass_m + COM_SHIFT_MAGNIFICATION
                  * (com - drawing.loaded_centre_of_mass_m))
        r = max(4, int(0.05 * extent * metres_to_px))
        pygame.draw.circle(surface, (120, 128, 138),
                           (int(loaded[0]), int(loaded[1])), r, 1)
        pygame.draw.line(surface, (255, 80, 200), loaded, here, 1)
        # the centre-of-mass symbol: a quartered disc
        point = (int(here[0]), int(here[1]))
        pygame.draw.circle(surface, (255, 255, 255), point, r)
        pygame.draw.rect(surface, (20, 20, 20),
                         pygame.Rect(point[0] - r, point[1] - r, r, r))
        pygame.draw.rect(surface, (20, 20, 20),
                         pygame.Rect(point[0], point[1], r, r))
        pygame.draw.circle(surface, (255, 80, 200), point, r, 1)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return f"{int(hours):d}h{int(minutes):02d}m{secs:04.1f}s"


def main(frames: int = 0, out_prefix: str = "orbital_game", every: int = 1,
         click: int | None = None,
         planner: str = "hohmann", burn_shots: bool = False,
         station_dx_m: float = STATION_LENGTH_SCALE_M,
         time_multiplier: float = DEFAULT_TIME_MULTIPLIER) -> int:
    """``frames`` > 0 runs headless, saving a PNG every ``every`` frames
    (turret_demo's check-without-a-person mode); ``burn_shots`` also saves
    a frame while plumes are lit (at most one per 8 frames)."""
    import pygame
    from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_RGB, GL_UNSIGNED_BYTE,
                           glClear, glClearColor, glReadPixels)

    game = OrbitalGame(planner=planner, station_length_scale_m=station_dx_m)
    text = None
    step_service = None
    try:
        pygame.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.HIDDEN if frames else 0)
        pygame.display.set_mode(WINDOW, flags)
        pygame.display.set_caption("orbital game -- click a station to go there")
        font = pygame.font.SysFont("consolas", 15)
        small = pygame.font.SysFont("consolas", 12)

        from gl_text import TextLayer
        text = TextLayer(font, *WINDOW)

        extent = 1.08 * max([s.radius_m for s in game.specs]
                            + [CRAFT_RADIUS_M])
        scale = (PLAN_SIZE / 2 - 12) / extent
        centre = PLAN_SIZE / 2

        def to_px(x, y):
            return (centre + x * scale, centre - y * scale)

        messages: list[str] = []

        def say(message: str):
            messages.append(message)
            del messages[:-4]
            print(message, flush=True)

        pending_selections = []

        def order(index: int):
            pending_selections.append(index)
            say(f"planning -> {game.specs[index].name}")

        clock = pygame.time.Clock()
        pacer = PresentationPacer(game, time_multiplier)
        step_service = OrbitalStepService(game).start()
        snapshot = step_service.view
        display = pacer.sample()
        if click is not None:
            order(int(click))
        initial_selection = (pending_selections.pop(0)
                             if pending_selections else None)
        step_service.submit(snapshot.dt_proposal_s, initial_selection)
        running, frame, captured, last_burn_shot = True, 0, 0, -100
        plan_points = [to_px(p[0], p[1]) for p in snapshot.plan_curve]
        completed_generation = 0
        pending_publication = None
        guidance_rate_pending = False
        reported = 0
        frame_wall_s = 0.0
        while running:
            wall_elapsed_s = ((1.0 / PRESENTATION_FPS) if frames else
                              clock.tick(int(PRESENTATION_FPS)) / 1000.0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS,
                                      pygame.K_KP_PLUS, pygame.K_RIGHTBRACKET):
                        pacer.faster()
                        say(f"time rate {pacer.multiplier:g}x")
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS,
                                      pygame.K_LEFTBRACKET):
                        pacer.slower()
                        say(f"time rate {pacer.multiplier:g}x")
                    elif event.key == pygame.K_SPACE:
                        pacer.paused = not pacer.paused
                        say("time paused" if pacer.paused else
                            f"time rate {pacer.multiplier:g}x")
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        index = event.key - pygame.K_1
                        if index < len(game.specs):
                            order(index)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos[0] - PLAN_X, event.pos[1] - PLAN_Y
                    best, best_d = None, 28.0
                    for index, position in enumerate(display.station_position_m):
                        px, py = to_px(position[0], position[1])
                        distance = math.hypot(px - mx, py - my)
                        if distance < best_d:
                            best, best_d = index, distance
                    if best is not None:
                        order(best)

            if frames and step_service.busy:
                step_service.wait()
            publication = step_service.latest_result()
            if (publication is not None
                    and publication.request.generation > completed_generation):
                completed_generation = publication.request.generation
                frame_wall_s = publication.wall_s
                if publication.error is not None:
                    selected = publication.request.selection
                    prefix = ("simulation" if selected is None else
                              game.specs[selected].name)
                    pacer.paused = True
                    say(f"{prefix}: request refused ({publication.error}); "
                        "simulation paused")
                else:
                    pending_publication = publication
                    candidate = publication.view
                    if (candidate.guidance_active
                            and candidate.guidance_signature
                            != snapshot.guidance_signature
                            and pacer.multiplier
                            > GUIDANCE_PRESENTATION_MULTIPLIER):
                        # This endpoint includes coast time already accepted at
                        # the old presentation rate.  Present that interval at
                        # the rate under which it was requested, then reduce
                        # the rate before asking physics for a guided window.
                        guidance_rate_pending = True

            if pending_publication is not None and pacer.ready:
                snapshot = pending_publication.view
                pacer.accept(snapshot.endpoint)
                pending_publication = None
                plan_points = [to_px(p[0], p[1])
                               for p in snapshot.plan_curve]
                for message in snapshot.announcements:
                    say(message)
                while reported < len(snapshot.rendezvous):
                    meet = snapshot.rendezvous[reported]
                    say(f"arrived at {game.specs[meet.target].name}: "
                        f"{meet.distance_m / 1000.0:.3f} km, "
                        f"{meet.relative_speed_m_s:.2f} m/s")
                    reported += 1

            demand_s = pacer.wall_demand(wall_elapsed_s)
            demand_s = pacer.consume(demand_s)
            # The worker may be negotiating a long burn window.  The UI keeps
            # this remainder as reported backpressure only; it never creates
            # an unbounded catch-up request.
            pacer.backpressure_s = demand_s
            display = pacer.sample()

            if guidance_rate_pending and pacer.ready:
                pacer.limit_for_guidance()
                guidance_rate_pending = False
                say(f"guidance changed: time rate {pacer.multiplier:g}x")

            if (not step_service.busy and pending_publication is None
                    and not guidance_rate_pending
                    and (pending_selections or not pacer.paused)
                    and (not frames or captured < frames - 1)):
                # The dt system publishes the next requested window.  The
                # display multiplier only controls tweening of accepted time.
                window = None if pacer.paused else snapshot.dt_proposal_s
                selection = (pending_selections.pop(0)
                             if pending_selections else None)
                step_service.submit(window, selection)


            # ---- draw the plan view ----
            text.begin_frame()
            glClearColor(BG[0] / 255, BG[1] / 255, BG[2] / 255, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            view = pygame.Surface((PLAN_SIZE, PLAN_SIZE), pygame.SRCALPHA)
            view.fill((18, 20, 26, 255))
            pygame.draw.circle(view, BLUE, (int(centre), int(centre)),
                               max(3, int(R_EARTH * scale)))
            stations = display.station_position_m
            for index, (spec, position) in enumerate(zip(game.specs, stations)):
                colour = TARGET_COLOURS[index % len(TARGET_COLOURS)]
                pygame.draw.circle(view, (38, 44, 54), (int(centre), int(centre)),
                                   int(spec.radius_m * scale), 1)
                px, py = to_px(position[0], position[1])
                chosen = snapshot.order_target == index
                pygame.draw.circle(view, colour, (int(px), int(py)),
                                   7 if chosen else 5)
                if chosen:
                    pygame.draw.circle(view, colour, (int(px), int(py)), 13, 1)
                view.blit(small.render(f"{index + 1} {spec.name}", True, colour),
                          (px + 9, py - 7))
            if len(plan_points) > 1:
                pygame.draw.lines(view, AMBER if snapshot.on_plan else RED, False,
                                  plan_points, 1)
            if len(snapshot.path) > 1:
                trail = [to_px(x, y) for x, y in snapshot.path[-3000:]]
                pygame.draw.lines(view, GREEN, False, trail, 2)
            position = display.craft_position_m
            drawing = display.craft_drawing
            # on the map the craft is an icon: its declared geometry at a fixed
            # icon scale (the body's largest extent ~14 px)
            icon_scale = 14.0 / max(float(np.max(np.abs(drawing.body))) * 2.0,
                                    1.0e-9)
            _draw_craft(view, to_px(position[0], position[1]), icon_scale,
                        drawing, frame, detail=False)
            text.draw_surface(view, PLAN_X, PLAN_Y, cache_key=("view", frame))
            text.draw(f"dt limiting metric: {snapshot.dt_blocker}",
                      PLAN_X + 8, PLAN_Y + 8, AMBER)

            # ---- the panel ----
            rows = [
                ("ORBITAL GAME -- machine craft", AMBER),
                (f"t = {_fmt_time(display.time_s)}   dt proposal "
                 f"{snapshot.dt_proposal_s:.6g} s   "
                 f"wall {frame_wall_s:5.2f} s", INK),
                (("PAUSED" if pacer.paused else
                  f"rate {pacer.multiplier:g}x")
                 + f"   backpressure {pacer.backpressure_s:.3g} s"
                 + ("   calculating" if step_service.busy else ""), AMBER),
                (f"propagator: {snapshot.propagation_mode}",
                 GREEN if snapshot.propagation_mode.startswith("GREEN")
                 else AMBER),
                (f"planner: {snapshot.planner_name}   status: "
                 f"{snapshot.phase_name}",
                 INK),
            ]
            if snapshot.order_target is not None:
                spec = game.specs[snapshot.order_target]
                _start, arrival, first_burn, _last_burn = snapshot.plan_times
                rows += [
                    (f"tracker: {'ON PLAN' if snapshot.on_plan else 'OFF PLAN'}"
                     f"   re-plans {snapshot.replans}",
                     GREEN if snapshot.on_plan else RED),
                    (f"target: {spec.name}", INK),
                    ((f"burn 1 in {_fmt_time(first_burn - display.time_s)}"
                      if display.time_s < first_burn else "burn 1 done")
                     + "   " +
                     (f"arrive in {_fmt_time(arrival - display.time_s)}"
                      if display.time_s < arrival else "arrived"), INK),
                ]
                rows.append((snapshot.plan_delta_v, INK))
            if snapshot.plan_deviation_m is not None:
                rows.append((f"plan deviation {snapshot.plan_deviation_m:9.1f} m",
                             GREEN))
            if snapshot.rendezvous:
                meet = snapshot.rendezvous[-1]
                rows.append((f"last rendezvous {meet.distance_m:.1f} m "
                             f"@ {meet.relative_speed_m_s:.3f} m/s", AMBER))
            rows.append((f"propellant {snapshot.propellant_kg:7.1f} kg "
                         f"({100.0 * snapshot.propellant_fraction:5.1f}%)  mass "
                         f"{snapshot.mass_kg:7.1f} kg",
                         RED if snapshot.propellant_fraction < 0.1 else INK))
            for t, (identity, fluid, left, loaded) in enumerate(snapshot.tanks):
                share = 0.0 if loaded <= 0.0 else left / loaded
                rows.append((f"  {identity:<15} {fluid:<10} "
                             f"{left:7.2f} kg ({100.0 * share:5.1f}%)",
                             TANK_COLOURS[t % len(TANK_COLOURS)]))
            gimbal = snapshot.main_gimbal
            if gimbal is not None:
                state, command = gimbal
                commanded = ("--" if math.isnan(command)
                             else f"{math.degrees(command):5.2f}")
                rows.append((f"main gimbal {math.degrees(state):5.2f} deg "
                             f"(command {commanded} deg)", INK))
            rows.append((f"centre of mass moved {1000.0 * snapshot.com_shift_m:6.1f}"
                         f" mm since loading", (255, 120, 210)))
            rows.append(("thruster groups (live / fired this frame):", DIM))
            for label, count, lit, live, fired in snapshot.thruster_groups:
                rows.append((f"  {label:<6} {lit:2d}/{count:<2d} "
                             f"{'FIRING' if lit else '  off '}  live {live:4.2f}"
                             f"  frame {fired:4.2f}",
                             AMBER if lit else DIM))
            rows.append(("reaction-wheel actuators (last command / live state):",
                         DIM))
            for wheel, command, momentum, speed, power, dumping in (
                    snapshot.wheel_actuators):
                axis = "".join(f"{v:+.0f}" for v in wheel.axis)
                rows.append((
                    f"  {wheel.identity:<9} axis {axis}  "
                    f"tau {command:+.3f}/{wheel.max_torque_n_m:.1f} Nm  "
                    f"{speed * 60.0 / (2.0 * math.pi):+6.0f}/"
                    f"{wheel.max_speed_rad_s * 60.0 / (2.0 * math.pi):.0f} rpm",
                    AMBER if dumping or abs(command) > 1.0e-6 else DIM))
                rows.append((
                    f"    momentum {momentum:+.1f}/"
                    f"{wheel.max_momentum_n_m_s:.1f} Nms  "
                    f"motor {power:+6.1f} W{'  DUMP' if dumping else ''}",
                    AMBER if dumping or abs(command) > 1.0e-6 else DIM))
            y = 20
            for row, colour in rows:
                text.draw(row, PANEL_X, y, colour)
                y += 19
            y += 4
            inset = pygame.Surface((INSET_SIZE, INSET_SIZE), pygame.SRCALPHA)
            inset.fill((22, 25, 31, 255))
            reach = max(float(np.max(np.abs(drawing.body))),
                        max((float(np.linalg.norm(drawing.centre_of_mass_m
                                                  + part.mount_m))
                             for part in drawing.parts), default=0.0))
            inset_scale = 0.5 * INSET_SIZE / (reach * (1.0 + 0.6 * PLUME_LENGTH_BODY))
            _draw_craft(inset, (INSET_SIZE / 2, INSET_SIZE / 2), inset_scale,
                        drawing, frame)
            inset.blit(small.render(
                f"top view; CoM shift x{COM_SHIFT_MAGNIFICATION:.0f}", True, DIM),
                (6, 4))
            text.draw_surface(inset, PANEL_X, y, cache_key=("inset", frame))
            y += INSET_SIZE + 8
            for row in ("click/1-9 target; +/- time rate; SPACE pause; ESC",
                        *messages):
                text.draw(row, PANEL_X, y, DIM if not row.startswith(
                    ("->", "arr", "OFF")) else AMBER)
                y += 19
            text.end_frame()

            if frames:
                # a main or retro engine lit (live or over the frame)
                lit = any(max(live, fired) > 0.05 for label, _n, _lit, live, fired
                          in snapshot.thruster_groups if label != "RCS")
                burn_shot = burn_shots and lit and frame - last_burn_shot >= 8
                if burn_shot:
                    last_burn_shot = frame
                if frame % max(1, every) == 0 or burn_shot:
                    buf = glReadPixels(0, 0, WINDOW[0], WINDOW[1], GL_RGB,
                                       GL_UNSIGNED_BYTE)
                    image = np.frombuffer(buf, dtype=np.uint8).reshape(
                        WINDOW[1], WINDOW[0], 3)[::-1]
                    surface = pygame.image.frombuffer(
                        np.ascontiguousarray(image).tobytes(), WINDOW, "RGB")
                    Path(f"{out_prefix}_{captured:02d}.png").parent.mkdir(
                        parents=True, exist_ok=True)
                    pygame.image.save(surface, f"{out_prefix}_{captured:02d}.png")
                    captured += 1
                    if captured >= frames:
                        running = False
            else:
                pygame.display.flip()
            frame += 1

        return 0
    finally:
        try:
            if step_service is None:
                game.close()
            else:
                step_service.stop()
        finally:
            try:
                if text is not None:
                    text.close()
            finally:
                pygame.quit()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0,
                    help="headless: save this many PNGs")
    ap.add_argument("--every", type=int, default=1,
                    help="headless: frames between saved PNGs")
    ap.add_argument("--out", default="orbital_game")
    ap.add_argument("--click", type=int, default=None,
                    help="select this target index at start")
    ap.add_argument("--planner", default="hohmann", choices=sorted(PLANNERS))
    ap.add_argument("--burn-shots", action="store_true",
                    help="headless: also save frames while plumes are lit")
    ap.add_argument("--station-dx", type=float,
                    default=STATION_LENGTH_SCALE_M,
                    help="stations' controller dx (m): accuracy vs speed")
    ap.add_argument("--time-multiplier", type=float,
                    default=DEFAULT_TIME_MULTIPLIER,
                    help="initial simulation seconds per wall second")
    a = ap.parse_args()
    raise SystemExit(main(frames=a.frames, out_prefix=a.out, every=a.every,
                          click=a.click, planner=a.planner,
                          burn_shots=a.burn_shots,
                          station_dx_m=a.station_dx,
                          time_multiplier=a.time_multiplier))
