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
                   ``planner=`` switch, the collocation planner), PHASED by
                   the laws below so the craft meets the target, not just
                   its radius
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

Phasing laws (section PH, composed with ``orbital_plan.KEPLER_LAWS``).  The
plan's burn 1 is at the craft's polar angle ``phase`` and it arrives on the
target circle at ``phase + sweep`` (``sweep = pi`` for Hohmann,
``HohmannPlan.legs``) after ``t_transfer``.  With the target leading the
craft by ``phi_0`` now:

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
                             tracker_mode)

MU_EARTH = 3.986004418e14
R_EARTH = 6.371e6
CRAFT_RADIUS_M = 7.0e6
#: Stations are plain thrusterless jumpers (``mass_kg``); the mass only
#: names the lanes (they do not attract the craft).
TARGET_MASS_KG = 1.0e3
#: The machine craft's round and gains: the configuration
#: ``tests/test_orbital_tracker.py::test_machine_craft_transfer_arrives_
#: with_gimballed_burns`` flies (2 s rounds; attitude PD at 0.2 rad/s,
#: w_a * round = 0.4).
ROUND_S = 2.0
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
    # orbital_collocation is being built by another agent; when it exposes
    # a plan with the HohmannPlan interface it plugs in here.
    import orbital_collocation
    return orbital_collocation.collocation_plan


PLANNERS = {
    "hohmann": lambda: hohmann_plan,
    "collocation": _collocation_planner,
}
#: Polar angle the plan sweeps from burn 1 to arrival (HohmannPlan.legs:
#: the arrival circle's periapsis angle is ``phase + pi``).
PLAN_SWEEP_RAD = {"hohmann": math.pi, "collocation": math.pi}


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


def _circular_state(mu: float, radius: float, angle: float):
    speed = math.sqrt(mu / radius)
    return ((radius * math.cos(angle), radius * math.sin(angle), 0.0),
            (-speed * math.sin(angle), speed * math.cos(angle), 0.0))


def _polar(position) -> float:
    return math.atan2(float(position[1]), float(position[0]))


class OrbitalGame:
    """The non-graphical game: craft, targets, orders, frames."""

    def __init__(self, *, mu: float = MU_EARTH,
                 craft_radius_m: float = CRAFT_RADIUS_M,
                 craft_angle_rad: float = 0.0,
                 targets=DEFAULT_TARGETS,
                 machine: CraftMachine | None = None,
                 gains: TrackingGains | None = None,
                 round_s: float = ROUND_S,
                 length_scale_m: float = LENGTH_SCALE_M,
                 station_length_scale_m: float = STATION_LENGTH_SCALE_M,
                 planner: str = "hohmann"):
        if planner not in PLANNERS:
            raise ValueError(f"unknown planner {planner!r}; "
                             f"one of {sorted(PLANNERS)}")
        self.mu = float(mu)
        self.planner_name = planner
        self.planner = PLANNERS[planner]()
        # the tracker's own full-trip re-planner (its ON/OFF-PLAN switch);
        # the collocation planner's re-planner needs a problem object
        # (orbital_collocation.collocation_replanner): not wired here
        self.replanner = hohmann_replanner if planner == "hohmann" else None
        self.gains = MACHINE_GAINS if gains is None else gains
        self.round_s = float(round_s)
        center = GravityCenter((0.0, 0.0, 0.0), self.mu)
        machine = orbital_craft() if machine is None else machine
        position, velocity = _circular_state(self.mu, craft_radius_m,
                                             craft_angle_rad)
        self.craft = MachineCraft(
            [center], machine, position_m=position, velocity_m_s=velocity,
            length_scale_m=length_scale_m, window_s=self.round_s)
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
            window_s=self.round_s, batch=len(self.specs))
        self.max_thrust = np.asarray(
            [t.max_thrust_n for t in machine.thrusters], dtype=float)
        self.order: TransferOrder | None = None
        self.report = None          # the last frame's FlightReport
        self.rendezvous: list[Rendezvous] = []
        self.plume_throttles = np.zeros(machine.thruster_count)
        self.plan_deviation_m: float | None = None
        self.path: list[tuple[float, float]] = []
        self._record_path()

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
        return self.flown_plan().t_burn2 + ARRIVAL_SETTLE_S

    def busy(self) -> bool:
        """An order is still on its way (its rendezvous is ahead)."""
        return (self.order is not None
                and self.time_s < self.arrival_s() - 1.0e-9)

    def phase_name(self) -> str:
        if self.order is None:
            return "coasting"
        plan, t = self.flown_plan(), self.time_s
        if t < plan.t_burn1:
            return "phasing wait"
        if t < plan.t_burn2:
            return "transfer"
        if t < self.arrival_s():
            return "arrival burn"
        return "on station orbit"

    def plan_curve(self, samples: int = 240) -> np.ndarray:
        """The flown plan's reference from burn 1 to arrival, (samples, 3)."""
        if self.order is None:
            return np.zeros((0, 3))
        plan = self.flown_plan()
        times = np.linspace(plan.t_burn1, plan.t_burn2, samples)
        return reference(plan, times)[0]

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

    # ------------------------------------------------------------ orders
    def select(self, index: int) -> TransferOrder:
        """Order a phased transfer from the present state to target
        ``index``.  Refused while a transfer is on its way."""
        if self.busy():
            raise RuntimeError("transfer in progress")
        craft_r, _ = self.craft.r()
        target_r, _ = self.target_state(index)
        r1 = float(np.linalg.norm(craft_r))
        r2 = float(np.linalg.norm(target_r))
        theta_0 = _polar(craft_r)
        phi_0 = _polar(target_r) - theta_0
        probe = self.planner(self.mu, r1, r2)
        timing = phasing(self.mu, r1, r2, probe.transfer_time,
                         PLAN_SWEEP_RAD[self.planner_name], phi_0, theta_0)
        plan = self.planner(self.mu, r1, r2,
                            t_burn1=self.time_s + timing.t_wait,
                            phase=timing.phase)
        self.order = TransferOrder(index, self.time_s, plan, timing, phi_0)
        self.report = None
        return self.order

    # ------------------------------------------------------------ frames
    def frame_window(self, base_s: float, burn_s: float = 2.0,
                     margin_s: float = 60.0) -> float:
        """Time warp: ``burn_s`` per frame within ``margin_s`` of a burn so
        the plumes can be seen, else ``base_s``."""
        if self.order is not None:
            t = self.time_s
            plan = self.flown_plan()
            for burn in (plan.t_burn1, plan.t_burn2):
                if burn - margin_s <= t <= burn + max(margin_s,
                                                      ARRIVAL_SETTLE_S):
                    return burn_s
        return base_s

    def step(self, window_s: float) -> float:
        """Advance craft and stations over one frame window, in lockstep.
        The window is clipped to land on the order's arrival time, where
        the rendezvous is measured.  Returns the window advanced."""
        window = float(window_s)
        arrival = None
        if self.busy():
            arrival = self.arrival_s()
            window = min(window, arrival - self.time_s)
        start = self.time_s
        impulses = self.craft.thruster_impulses_n_s
        if self.order is not None:
            self.report = fly(self.craft, self.order.plan, self.gains,
                              until_s=start + window, round_s=self.round_s,
                              replanner=self.replanner)
            self.plan_deviation_m = self.report.final_position_error_m
        else:
            remaining = window
            while remaining > 1.0e-9:
                chunk = min(self.round_s, remaining)
                self.craft.advance(chunk)
                remaining -= chunk
        advanced = self.craft.time_s - start
        if advanced > 0.0:
            # every station lane in one round of the one batched dt state
            self.stations.advance(advanced)
            fired = self.craft.thruster_impulses_n_s - impulses
            self.plume_throttles = fired / (advanced * self.max_thrust)
        self._record_path()
        if arrival is not None and self.time_s >= self.arrival_s() - 1.0e-9:
            self._measure_rendezvous()
        return advanced

    def run_until_arrival(self, window_s: float) -> Rendezvous:
        """Step until the active order's arrival is measured (no graphics)."""
        if not self.busy():
            raise RuntimeError("no transfer on its way")
        count = len(self.rendezvous)
        while len(self.rendezvous) == count:
            self.step(self.frame_window(window_s))
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
BASE_WARP_S = 60.0
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
         click: int | None = None, warp_s: float = BASE_WARP_S,
         planner: str = "hohmann", burn_shots: bool = False,
         station_dx_m: float = STATION_LENGTH_SCALE_M) -> int:
    """``frames`` > 0 runs headless, saving a PNG every ``every`` frames
    (turret_demo's check-without-a-person mode); ``burn_shots`` also saves
    a frame while plumes are lit (at most one per 8 frames)."""
    import pygame
    from OpenGL.GL import (GL_COLOR_BUFFER_BIT, GL_RGB, GL_UNSIGNED_BYTE,
                           glClear, glClearColor, glReadPixels)

    game = OrbitalGame(planner=planner, station_length_scale_m=station_dx_m)
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

    def order(index: int):
        try:
            placed = game.select(index)
        except (RuntimeError, ValueError) as error:
            say(f"{game.specs[index].name}: refused ({error})")
            return
        plan = placed.plan
        say(f"-> {game.specs[index].name}: wait "
            f"{_fmt_time(placed.phasing.t_wait)}, transfer "
            f"{_fmt_time(plan.transfer_time)}")

    if click is not None:
        order(int(click))

    clock = pygame.time.Clock()
    running, frame, captured, last_burn_shot = True, 0, 0, -100
    plan_points: list = []
    plan_for = None
    reported = 0
    replans_seen = 0
    frame_wall_s = 0.0
    while running:
        if not frames:
            clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                   pygame.K_KP_PLUS):
                    warp_s = min(warp_s * 2.0, 960.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    warp_s = max(warp_s / 2.0, 2.0)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    index = event.key - pygame.K_1
                    if index < len(game.specs):
                        order(index)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos[0] - PLAN_X, event.pos[1] - PLAN_Y
                best, best_d = None, 28.0
                for index, position in enumerate(game.station_states()[0]):
                    px, py = to_px(position[0], position[1])
                    distance = math.hypot(px - mx, py - my)
                    if distance < best_d:
                        best, best_d = index, distance
                if best is not None:
                    order(best)

        started = time.perf_counter()
        game.step(game.frame_window(warp_s))
        frame_wall_s = time.perf_counter() - started
        while reported < len(game.rendezvous):
            meet = game.rendezvous[reported]
            say(f"arrived at {game.specs[meet.target].name}: "
                f"{meet.distance_m / 1000.0:.3f} km, "
                f"{meet.relative_speed_m_s:.2f} m/s")
            reported += 1
        if game.replans() > replans_seen:
            replans_seen = game.replans()
            say(f"OFF PLAN: re-plan {replans_seen}")
        if game.order is None:
            replans_seen = 0
        flown = game.flown_plan()
        if flown is not plan_for:
            plan_for = flown
            plan_points = [to_px(p[0], p[1]) for p in game.plan_curve()]

        # ---- draw the plan view ----
        text.begin_frame()
        glClearColor(BG[0] / 255, BG[1] / 255, BG[2] / 255, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        view = pygame.Surface((PLAN_SIZE, PLAN_SIZE), pygame.SRCALPHA)
        view.fill((18, 20, 26, 255))
        pygame.draw.circle(view, BLUE, (int(centre), int(centre)),
                           max(3, int(R_EARTH * scale)))
        stations, _ = game.station_states()
        for index, (spec, position) in enumerate(zip(game.specs, stations)):
            colour = TARGET_COLOURS[index % len(TARGET_COLOURS)]
            pygame.draw.circle(view, (38, 44, 54), (int(centre), int(centre)),
                               int(spec.radius_m * scale), 1)
            px, py = to_px(position[0], position[1])
            chosen = game.order is not None and game.order.target == index
            pygame.draw.circle(view, colour, (int(px), int(py)),
                               7 if chosen else 5)
            if chosen:
                pygame.draw.circle(view, colour, (int(px), int(py)), 13, 1)
            view.blit(small.render(f"{index + 1} {spec.name}", True, colour),
                      (px + 9, py - 7))
        if len(plan_points) > 1:
            pygame.draw.lines(view, AMBER if game.on_plan() else RED, False,
                              plan_points, 1)
        if len(game.path) > 1:
            trail = [to_px(x, y) for x, y in game.path[-3000:]]
            pygame.draw.lines(view, GREEN, False, trail, 2)
        position, _ = game.craft.r()
        drawing = craft_drawing(game.craft, game.loaded_centre_of_mass)
        # on the map the craft is an icon: its declared geometry at a fixed
        # icon scale (the body's largest extent ~14 px)
        icon_scale = 14.0 / max(float(np.max(np.abs(drawing.body))) * 2.0,
                                1.0e-9)
        _draw_craft(view, to_px(position[0], position[1]), icon_scale,
                    drawing, frame, detail=False)
        text.draw_surface(view, PLAN_X, PLAN_Y, cache_key=("view", frame))

        # ---- the panel ----
        rows = [
            ("ORBITAL GAME -- machine craft", AMBER),
            (f"t = {_fmt_time(game.time_s)}   warp "
             f"{game.frame_window(warp_s):.0f} s/frame   "
             f"wall {frame_wall_s:5.2f} s", INK),
            (f"planner: {game.planner_name}   status: {game.phase_name()}",
             INK),
        ]
        if game.order is not None:
            plan = game.flown_plan()
            spec = game.specs[game.order.target]
            rows += [
                (f"tracker: {'ON PLAN' if game.on_plan() else 'OFF PLAN'}"
                 f"   re-plans {game.replans()}",
                 GREEN if game.on_plan() else RED),
                (f"target: {spec.name}", INK),
                ((f"burn 1 in {_fmt_time(plan.t_burn1 - game.time_s)}"
                  if game.time_s < plan.t_burn1 else "burn 1 done")
                 + "   " +
                 (f"arrive in {_fmt_time(plan.t_burn2 - game.time_s)}"
                  if game.time_s < plan.t_burn2 else "arrived"), INK),
                (f"plan dv {plan.dv1:+.1f} / {plan.dv2:+.1f} m/s", INK),
            ]
        if game.plan_deviation_m is not None:
            rows.append((f"plan deviation {game.plan_deviation_m:9.1f} m",
                         GREEN))
        if game.rendezvous:
            meet = game.rendezvous[-1]
            rows.append((f"last rendezvous {meet.distance_m:.1f} m "
                         f"@ {meet.relative_speed_m_s:.3f} m/s", AMBER))
        rows.append((f"propellant {game.propellant_kg:7.1f} kg "
                     f"({100.0 * game.propellant_fraction:5.1f}%)  mass "
                     f"{game.craft.mass_kg:7.1f} kg",
                     RED if game.propellant_fraction < 0.1 else INK))
        contents = game.craft.tank_propellant_kg()
        for t, tank in enumerate(game.craft.craft.tanks):
            left = contents[tank.identity]
            loaded = game.loaded_tank_kg[tank.identity]
            share = 0.0 if loaded <= 0.0 else left / loaded
            rows.append((f"  {tank.identity:<15} {tank.fluid:<10} "
                         f"{left:7.2f} kg ({100.0 * share:5.1f}%)",
                         TANK_COLOURS[t % len(TANK_COLOURS)]))
        gimbal = game.main_gimbal()
        if gimbal is not None:
            state, command = gimbal
            commanded = ("--" if math.isnan(command)
                         else f"{math.degrees(command):5.2f}")
            rows.append((f"main gimbal {math.degrees(state):5.2f} deg "
                         f"(command {commanded} deg)", INK))
        shift = game.craft.centre_of_mass() - game.loaded_centre_of_mass
        rows.append((f"centre of mass moved {1000.0 * np.linalg.norm(shift):6.1f}"
                     f" mm since loading", (255, 120, 210)))
        rows.append(("thruster groups (live / fired this frame):", DIM))
        for label, count, lit, live, fired in game.thruster_groups():
            rows.append((f"  {label:<6} {lit:2d}/{count:<2d} "
                         f"{'FIRING' if lit else '  off '}  live {live:4.2f}"
                         f"  frame {fired:4.2f}",
                         AMBER if lit else DIM))
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
        for row in ("click a station (or 1-9) to go there; +/- warp; ESC",
                    *messages):
            text.draw(row, PANEL_X, y, DIM if not row.startswith(
                ("->", "arr", "OFF")) else AMBER)
            y += 19
        text.end_frame()

        if frames:
            # a main or retro engine lit (live or over the frame)
            lit = any(max(live, fired) > 0.05 for label, _n, _lit, live, fired
                      in game.thruster_groups() if label != "RCS")
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

    text.close()
    pygame.quit()
    return 0


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
    ap.add_argument("--warp", type=float, default=BASE_WARP_S,
                    help="sim seconds per frame away from burns")
    ap.add_argument("--planner", default="hohmann", choices=sorted(PLANNERS))
    ap.add_argument("--burn-shots", action="store_true",
                    help="headless: also save frames while plumes are lit")
    ap.add_argument("--station-dx", type=float,
                    default=STATION_LENGTH_SCALE_M,
                    help="stations' controller dx (m): accuracy vs speed")
    a = ap.parse_args()
    raise SystemExit(main(frames=a.frames, out_prefix=a.out, every=a.every,
                          click=a.click, warp_s=a.warp, planner=a.planner,
                          burn_shots=a.burn_shots,
                          station_dx_m=a.station_dx))
