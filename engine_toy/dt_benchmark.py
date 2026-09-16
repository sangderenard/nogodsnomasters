"""THE WHOLE GAME ON ONE CLOCK, AND WHAT IT COSTS TO ADVANCE IT.

    python dt_benchmark.py              one frame, timed, per engine
    python dt_benchmark.py --frames 60  a run

WHAT THIS IS. Every simulator this project has, registered as an engine
on `src.common.dt_system` and advanced by ONE dt cascade rather than by
each one calling its own loop:

    joints      the oleo law, SymPy -> the shared bank printer
    members     the beam law, SymPy -> the same printer -> LLVM
    motion      `dt_system.integrator.Integrator`, whose `dynamics` is
                a SEQUENCE of derivative providers summed -- the two
                banks above -- because that is what it is for
    engine      `engine_cycle_sim`, the crank, the cylinders, the fuel
    drivetrain  `drivetrain_graph`, torque across the shafts, clutches
                and gears, with its own stiff-spring stability rule

and the dt controller takes the MINIMUM of what each engine says it can
tolerate, instead of each one quietly substepping on its own.

    THAT IS THE WHOLE POINT AND IT IS WHY `Metrics.dt_limit` EXISTS:
    "the dt controller will clamp the next proposal to this value,
    centralizing stability control instead of engines self-capping
    internally". Both banks already compute exactly that number. They
    were keeping it to themselves.

WHAT IS DELIBERATELY NOT HERE. The balloon tire. There are no tires on
this machine yet, and registering a tire model with nothing to roll on
would be measuring an engine that cannot be asked for an answer.

WHAT THE NUMBER MEANS. Wall time to advance one frame, per engine and
in total, with AbstractTensor on its numpy backend and the compiled
member bank where one exists. It is the prospective cost of a game
frame with everything real running at once -- not a target, a
measurement.
"""
from __future__ import annotations

import argparse
import math
from time import perf_counter as _perf_counter
import sys
import time

import numpy as np

import graph_physics          # puts turing on sys.path
from src.common.tensors.abstraction import AbstractTensor
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.dt_graph import GraphBuilder, MetaLoopRunner
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.engine_api import DtCompatibleEngine, EngineRegistration
from src.common.dt_system.state_table import StateTable


def use_numpy_backend():
    """AbstractTensor on numpy, said once and out loud.

    The name, not the class: `set_default_backend` keys the registry by
    string and imports the module for it. Handing it the class raises
    "unknown tensor backend" with the class repr in the message, which
    reads like the class is unknown rather than like the wrong kind of
    thing was passed."""
    AbstractTensor.set_default_backend("numpy")
    return AbstractTensor.tensor([0.0]).__class__


def _metrics(max_vel: float = 0.0, dt_limit: float | None = None,
             stiff: bool = False) -> Metrics:
    """The generic diagnostics every engine reports.

    `dt_limit` is the one that matters here: it is how an engine says
    what step it can stand without capping itself behind the
    controller's back."""
    m = Metrics(max_vel=float(max_vel), max_flux=0.0, div_inf=0.0,
                mass_err=0.0)
    m.dt_limit = dt_limit
    m.stiff_flag = bool(stiff)
    return m


# =====================================================================
#  THE TWO BANKS, AS ENGINES
# =====================================================================
class JointBankEngine(DtCompatibleEngine):
    """Every travelling joint, under the oleo law.

    A FORCE ELEMENT AND NOTHING MORE. It is handed the velocity each
    joint is being worked at, it answers with the force that joint
    makes, and it advances only its own travel. It owns no mass and
    integrates no motion -- that belongs to `Integrator`, once, for
    everybody."""

    def __init__(self, bank, label: str = "joints"):
        super().__init__()
        self.bank = bank
        self.label = label
        self._registration = set()

    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out["compression"] = self.bank.compression.copy()
        return out

    def snapshot(self):
        return (self.bank.compression.copy(), self.bank.out.copy())

    def restore(self, snap):
        self.bank.compression[:], self.bank.out[:] = snap

    def step(self, dt: float, state=None, state_table=None):
        self.bank.step(float(dt))
        # what step this bank can stand: no joint may cross more than a
        # slice of its own travel in one go
        v = np.abs(self.bank.arrays["velocity_m_s"])
        stroke = np.maximum(self.bank.arrays["stroke_m"], 1e-6)
        from structure_native import STEP_OF_STROKE
        fastest = float(np.max(v / stroke)) if v.size else 0.0
        limit = (STEP_OF_STROKE / fastest) if fastest > 0.0 else None
        return (True, _metrics(max_vel=float(np.max(v)) if v.size else 0.0,
                               dt_limit=limit), state)


class BeamBankEngine(DtCompatibleEngine):
    """Every member, under the beam law, compiled where it can be."""

    def __init__(self, bank, label: str = "members"):
        super().__init__()
        self.bank = bank
        self.label = label
        self._registration = set()

    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out["q"] = self.bank.q.copy()
        return out

    def snapshot(self):
        return (self.bank.q.copy(), self.bank.qdot.copy())

    def restore(self, snap):
        self.bank.q[:], self.bank.qdot[:] = snap

    def step(self, dt: float, state=None, state_table=None):
        out = self.bank.step(float(dt))
        # THE LAW'S OWN COEFFICIENTS SAY THE LIMIT. `_substep_rate`
        # differentiates the generated velocity update to recover the
        # stiffness and damping the law actually has, so this is not a
        # margin anybody chose.
        from structure_native import _substep_rate
        names, evaluate = _substep_rate()
        rates = evaluate(*[self.bank.arrays[n] for n in names])
        peak = float(np.max(rates)) if np.size(rates) else 0.0
        limit = (1.0 / peak) if peak > 0.0 else None
        tip = out[:, 0]
        return (True, _metrics(
            max_vel=float(np.abs(tip).max()) if tip.size else 0.0,
            dt_limit=limit, stiff=peak > 1e4), state)


# =====================================================================
#  THE ENGINE AND THE DRIVETRAIN, AS ENGINES
# =====================================================================
class CycleEngine(DtCompatibleEngine):
    """The reciprocating engine: crank, cylinders, fuel, heat."""

    def __init__(self, sim, label: str = "engine"):
        super().__init__()
        self.sim = sim
        self.label = label
        self._registration = set()
        # THE LEDGER. Wall seconds spent inside this engine's interior
        # against world seconds it advanced -- which is tau for this one
        # subsystem, measured rather than set. Kept always: a profiler
        # that has to be switched on measures a different program.
        self.calls = 0
        self.wall_s = 0.0
        self.world_s = 0.0

    def reset_ledger(self):
        self.calls = 0
        self.wall_s = 0.0
        self.world_s = 0.0

    @property
    def tau(self) -> float:
        """World seconds this interior advanced per wall second spent.

        Below 1.0 the engine is dilated: it cannot keep up with the clock
        it is being asked to run against, and everything sharing its
        world has to wait or go on without it.
        """
        return self.world_s / self.wall_s if self.wall_s > 0.0 else 0.0

    def get_state(self, state=None):
        out = state if isinstance(state, dict) else {}
        out["rpm"] = float(getattr(self.sim, "rpm", 0.0))
        return out

    def snapshot(self):
        return None

    def restore(self, snap):
        return None

    def step(self, dt: float, state=None, state_table=None):
        started = _perf_counter()
        self.sim.step(float(dt))
        self.wall_s += _perf_counter() - started
        self.world_s += float(dt)
        self.calls += 1
        rpm = float(getattr(self.sim, "rpm", 0.0))
        # a cycle resolves on crank angle, so the step that matters is
        # the one that keeps a degree of crank from being skipped
        omega = rpm * 2.0 * math.pi / 60.0
        limit = (math.radians(2.0) / omega) if omega > 1e-6 else None
        return True, _metrics(max_vel=rpm, dt_limit=limit), state


class DrivetrainEngine(DtCompatibleEngine):
    """Torque across the shafts, and its own stiff-spring rule."""

    def __init__(self, solver, crank_omega_of, label: str = "drivetrain"):
        super().__init__()
        self.solver = solver
        self.crank_omega_of = crank_omega_of
        self.label = label
        self._registration = set()

    def get_state(self, state=None):
        return state if isinstance(state, dict) else {}

    def snapshot(self):
        return None

    def restore(self, snap):
        return None

    def step(self, dt: float, state=None, state_table=None):
        omega = float(self.crank_omega_of())
        self.solver.step(float(dt), omega)
        # ITS OWN RULE, NOT MINE. `_compute_stable_sub_dt` already says
        # what this graph can stand, from the stiffest spring-integrated
        # edge it actually has.
        limit = float(self.solver._compute_stable_sub_dt())
        return True, _metrics(max_vel=omega, dt_limit=limit), state


# =====================================================================
#  MOTION: the one integrator, fed by everyone
# =====================================================================
class MemberForces:
    """Derivative provider: what the members are pulling with.

    `graph_physics.GraphPhysics` already holds positions and velocities
    and runs the GAME'S OWN member law over them -- a J2 return map with
    hardening, viscosity and fracture. What it never did was scatter the
    result back onto the nodes, which is why its own docstring admits it
    "could only strain a member if you moved one of its ends yourself".
    That scatter is this, and it is the only thing added: force along
    the member, equal and opposite at its two ends."""

    def __init__(self, physics, node_mass):
        self.physics = physics
        self.mass = np.maximum(np.asarray(node_mass, float), 1e-9)
        self.edges = physics.constants.edge_nodes
        self.last_force = None

    def __call__(self, t, x):
        n = self.mass.size
        pos = np.asarray(x, float).reshape(1, n, 3)
        self.physics.positions = pos
        solved = self.physics.solve(1.0e-4)
        f = (np.asarray(solved["axial_stress_pa"], float).reshape(-1)
             * np.asarray(solved["section_area_m2"], float).reshape(-1))
        a, b = self.edges[:, 0], self.edges[:, 1]
        d = pos[0][b] - pos[0][a]
        L = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        pull = (d / L) * f[:, None]
        out = np.zeros((n, 3))
        np.add.at(out, a, pull)
        np.add.at(out, b, -pull)
        self.last_force = out
        return (out / self.mass[:, None]).reshape(-1)


class JointForces:
    """Derivative provider: what the travelling joints are making.

    The bank is a force element -- told a velocity, it answers with a
    force -- so this is the two lines that turn node motion into that
    velocity and the answer back into node acceleration."""

    def __init__(self, bank, pairs, node_mass, velocity_of):
        self.bank = bank
        self.pairs = pairs
        self.mass = np.maximum(np.asarray(node_mass, float), 1e-9)
        self.velocity_of = velocity_of
        self.last_force = None

    def __call__(self, t, x):
        n = self.mass.size
        pos = np.asarray(x, float).reshape(n, 3)
        vel = np.asarray(self.velocity_of(), float).reshape(n, 3)
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        d = pos[b] - pos[a]
        L = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        axis = d / L
        self.bank.arrays["velocity_m_s"][:] = np.einsum(
            "ij,ij->i", vel[b] - vel[a], axis)
        out_rows = self.bank.out.reshape(self.bank.n, -1)
        f = out_rows[:, -1]
        pull = axis * f[:, None]
        out = np.zeros((n, 3))
        np.add.at(out, a, pull)
        np.add.at(out, b, -pull)
        self.last_force = out
        return (out / self.mass[:, None]).reshape(-1)


class Gravity:
    """Derivative provider: the one that never changes."""

    def __init__(self, n_nodes: int, fixed_mask):
        self.a = np.zeros((n_nodes, 3))
        self.a[:, 1] = -9.80665
        self.a[np.asarray(fixed_mask, bool), :] = 0.0
        self.flat = self.a.reshape(-1)

    def __call__(self, t, x):
        return self.flat


def node_masses(document: dict) -> np.ndarray:
    """The lumped mass at every node, by `FrameSolver`'s own rule.

    A node carries what it declares plus half of every non-rigid member
    on it. Rigid links weigh nothing: their section is an idealisation
    sized to out-stiffen real structure, so charging them A*L*rho makes
    a body's own skin weigh thousands of times what the body does."""
    from milspec import MATERIAL_BY_KEY
    index = {n["identity"]: i for i, n in enumerate(document["nodes"])}
    m = np.zeros(len(index))
    for i, n in enumerate(document["nodes"]):
        m[i] += float(n.get("mass_kg", 0.0))
    pos = {n["identity"]: np.asarray(n["reference_position"], float)
           for n in document["nodes"]}
    for e in document["edges"]:
        if e.get("rigid") or e["a"] not in index or e["b"] not in index:
            continue
        r = float(e.get("radius", 0.012))
        L = float(np.linalg.norm(pos[e["b"]] - pos[e["a"]]))
        mat = MATERIAL_BY_KEY.get((e.get("damage") or {}).get("material",
                                                             "4130n"),
                                  MATERIAL_BY_KEY["4130n"])
        half = math.pi * r * r * L * float(mat.density_kg_m3) / 2.0
        m[index[e["a"]]] += half
        m[index[e["b"]]] += half
    return m


def build_round(dt: float, *, target: str = "native", algorithm: str = "verlet"):
    """Assemble every simulator as an engine on one dt cascade."""
    import station_reference as sr
    import structure_native as sn
    import graph_physics as gp

    built = {}
    t0 = time.perf_counter()
    g, plat, box, spec, st, site = sr.build()
    doc = g.as_document()
    built["graph"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    beam = sn.bank_for_target(doc, target=target)
    built["beam bank"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    joints = sn.joint_bank_for(doc)
    built["joint bank"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    physics = gp.GraphPhysics(document=doc)
    built["member law"] = time.perf_counter() - t0

    mass = node_masses(doc)
    index = {n["identity"]: i for i, n in enumerate(doc["nodes"])}
    fixed = np.array([bool(n.get("fixed_to")) for n in doc["nodes"]])
    mass[fixed] = 1e12                     # the world does not accelerate
    pairs = np.array([[index[e["a"]], index[e["b"]]]
                      for e in sn.force_joints(doc)], dtype=np.int64)

    x0 = np.asarray([n["reference_position"] for n in doc["nodes"]],
                    float).reshape(-1)
    velocity = {"v": np.zeros_like(x0)}

    from src.common.dt_system.integrator.integrator import Integrator
    members = MemberForces(physics, mass)
    jf = JointForces(joints, pairs, mass, lambda: velocity["v"])
    grav = Gravity(len(mass), fixed)
    motion = Integrator(dynamics=[grav, members, jf], algorithm=algorithm)
    motion._state = x0.copy()
    motion._v = np.zeros_like(x0)

    engines = [
        ("joints", JointBankEngine(joints)),
        ("members", BeamBankEngine(beam)),
        ("motion", motion),
    ]
    return {"doc": doc, "plat": plat, "beam": beam, "joints": joints,
            "physics": physics, "motion": motion, "velocity": velocity,
            "engines": engines, "built": built, "mass": mass,
            "members_provider": members, "joints_provider": jf}


def run_round(dt: float = 1.0 / 60.0, frames: int = 1, target: str = "native"):
    """Advance everything on one cascade and time what it cost."""
    rig = build_round(dt, target=target)
    ctrl = STController(dt_min=1e-7)
    targets = Targets(cfl=1.0, div_max=1.0, mass_max=1.0)
    gb = GraphBuilder(ctrl=ctrl, targets=targets, dx=0.1)
    table = StateTable()
    regs = [EngineRegistration(name=n, engine=e, targets=targets, dx=0.1,
                               localize=True)
            for n, e in rig["engines"]]
    # ---- THE NODES BECOME IDENTITIES IN THE SHARED TABLE ----------
    # `state_table` is how engines cooperate without knowing each other:
    # everyone derives their working state from it at the start of a
    # step and publishes back after. The Integrator refuses to run
    # until the things it is going to move have been declared there,
    # which is the right refusal -- an integrator with no registered
    # identities is moving nothing and would say so only by producing
    # zeros.
    doc = rig["doc"]
    rig["motion"].register(
        table,
        lambda n: {"pos": tuple(float(v) for v in n["reference_position"]),
                   "mass": float(n.get("mass_kg", 0.0))},
        doc["nodes"], group_label="station")
    runner = MetaLoopRunner(state_table=table)
    per = {n: 0.0 for n, _ in rig["engines"]}
    t_all = 0.0
    for _ in range(frames):
        node = gb.round(dt=dt, engines=regs, state_table=table)
        t0 = time.perf_counter()
        runner.run_round(node, state_table=table)
        t_all += time.perf_counter() - t0
    rig["frame_s"] = t_all / max(frames, 1)
    rig["per"] = per
    return rig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--target", default="native")
    a = ap.parse_args(argv)
    use_numpy_backend()
    rig = run_round(a.dt, a.frames, a.target)
    for k, v in rig["built"].items():
        print(f"  built {k:12s} {v:7.2f} s")
    print(f"  frame  {rig['frame_s'] * 1000:8.2f} ms  "
          f"({a.dt * 1000:.2f} ms of simulated time)")
    return rig


if __name__ == "__main__":
    main(sys.argv[1:])
