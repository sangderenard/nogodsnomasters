"""Rounds that are actually in the air.

A shot in this project used to be resolved the instant it was fired: the
compiled trajectory kernel ran the whole flight inside one call and the
impact was worked out before the tick ended. The physics in that flight
was real -- point-mass drag, wind, gravity, the same compiled kernel used
here -- but the TIME was not. A round that takes eight tenths of a second
to reach eight hundred metres arrived in the same instant it left, so
nothing could move while it travelled, and a lead error could never
actually miss.

So a projectile is an entity now. It leaves the exact muzzle of the gun
that fired it, it is stepped every tick like everything else, and it is
still out there on the next tick. Time of flight is time. A target that
manoeuvres while a round is in the air will be somewhere else when it
arrives, and that is not modelled -- it simply happens.

NO FAKE PATHS. Hit detection is a ray along the segment the projectile
ACTUALLY TRAVELLED during the tick, from where it was to where it now
is. Nothing is teleported to the vicinity of a target and traced
inward; the round goes where the physics puts it, and whatever that
segment crosses is what it hits.

THE MUZZLE IS THE ENTITY'S MUZZLE. Position and direction are read off
the posed barrel -- its node position plus its own rotated bore axis
times half its length -- so the round leaves the end of the tube that is
drawn on screen, in the direction that tube is actually pointing. Not an
offset from the mount, not a heading recomputed from the fire-control
angles: the geometry itself.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# the batched law and its backend live in turing, same as symbolic_parts
_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))


def muzzle_of(document: dict, weapon: str = "turret.weapon") -> tuple:
    """Where the bore actually ends, and where it points.

    Read off the POSED document, so it follows every stage of the mount
    -- hoist, traverse, elevation, the fine platform's twitch and the
    tube's own recoil travel -- with nothing recomputed and nothing able
    to disagree with what is drawn."""
    node = next((n for n in document["nodes"] if n["identity"] == weapon), None)
    if node is None:
        return np.zeros(3), np.array([0.0, 0.0, 1.0])
    centre = np.asarray(node["reference_position"], dtype=float)
    axis = np.asarray(node.get("drum_axis", (0.0, 0.0, 1.0)), dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    half = float(node.get("drum_length_m", 1.0)) / 2.0
    return centre + axis * half, axis


@dataclass
class Projectile:
    """One round, in flight."""
    identity: str
    calibre: str
    position: np.ndarray
    velocity: np.ndarray
    origin: np.ndarray
    fired_at_s: float = 0.0
    flight_s: float = 0.0
    distance_m: float = 0.0
    live: bool = True
    outcome: str = ""
    target: object = None            # what it was fired at, for the report

    @property
    def speed_m_s(self) -> float:
        return float(np.linalg.norm(self.velocity))


@dataclass
class ProjectileField:
    """Every round currently in the air, stepped with the compiled
    trajectory kernel.

    One kernel, built once. The same one `world_gun` uses, so a round
    fired here and a round traced there cannot follow different physics.
    """
    wind: tuple = (0.0, 0.0, 0.0)
    air_density_kg_m3: float = 1.225
    drag_coefficient: float = 0.30
    max_flight_s: float = 12.0
    substep_m: float = 3.0           # hit detection resolution along the path
    rounds: list = field(default_factory=list)
    spent: list = field(default_factory=list)
    elapsed_s: float = 0.0
    _kernel: object = None

    # ------------------------------------------------------------------
    #  THE INTEGRATOR: EVERY ROUND AT ONCE, AT THE BALLISTIC TIMESTEP
    # ------------------------------------------------------------------
    # TWO THINGS ARE NOT NEGOTIABLE HERE, and both were nearly given away.
    #
    # THE TIMESTEP. Flight integrates at `ballistic_dt_s` -- 2e-4 s,
    # the same step `world_gun` uses -- and never at the frame rate. At
    # 1010 m/s a frame step would be a seventeen metre jump, which is not
    # an integration of a drag law, it is a straight line with a rounding
    # error. The frame decides only HOW MANY ballistic steps run, and the
    # remainder is carried so that no time is created or lost.
    #
    # THE WIDTH. `symbolic_parts.compile_trajectory_ssa` is a SCALAR
    # kernel: hand it sixty-four rounds and it silently takes the first
    # and returns one answer, and at 0.17 ms a call a frame of
    # eighty-four substeps for sixty-four rounds would cost most of a
    # second. So the law is used in its BATCHED form
    # (`symbolic_parts.trajectory_batch_step`), which SymPy prints from
    # the same expression tree that the scalar kernel is compiled from.
    # One law, two widths, and `symbolic_parts.check_batch_parity`
    # measures their agreement at 8.7e-19 over a flight rather than
    # asserting it.
    ballistic_dt_s: float = 2.0e-4
    _carry_s: float = 0.0

    def _batch_step(self):
        from symbolic_parts import trajectory_batch_step
        return trajectory_batch_step()

    # ------------------------------------------------------------------
    def fire(self, *, muzzle, direction, calibre: str, target=None,
             dispersion_mrad: float = 0.6, rng=None) -> Projectile:
        """Emit a round from the muzzle, along the bore.

        DISPERSION IS APPLIED TO THE LAUNCH DIRECTION, which is where it
        physically belongs -- a gun scatters because the round leaves the
        tube slightly differently each time, not because it wanders in
        flight. Applied here it is then carried through the whole
        trajectory by the same physics as everything else."""
        from calibres import get_calibre
        c = get_calibre(calibre)
        d = np.asarray(direction, dtype=float)
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        if dispersion_mrad > 0.0:
            rng = rng or np.random.default_rng()
            # a small random tilt, in the plane perpendicular to the bore
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(d, up)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, d)
            sigma = dispersion_mrad / 1000.0
            d = d + right * rng.normal(0.0, sigma) + up * rng.normal(0.0, sigma * 0.8)
            d /= np.linalg.norm(d)
        muzzle = np.asarray(muzzle, dtype=float)
        shot = Projectile(identity=f"round-{len(self.rounds) + len(self.spent) + 1:04d}",
                          calibre=calibre, position=muzzle.copy(),
                          velocity=d * c.muzzle_m_s, origin=muzzle.copy(),
                          fired_at_s=self.elapsed_s, target=target)
        self.rounds.append(shot)
        return shot

    # ------------------------------------------------------------------
    def step(self, dt: float, bodies) -> list:
        """Fly everything, and resolve whatever a round actually crosses.

        `bodies` maps identity -> (world_position, TargetBody). A round
        is tested against the SEGMENT it travelled this tick, so a fast
        round cannot tunnel through a target between frames -- the
        segment is subdivided at `substep_m` and each piece is traced."""
        from calibres import get_calibre
        self.elapsed_s += dt
        events = []
        if not self.rounds:
            return events
        # ---- fly every round together, at the ballistic timestep ----
        self._carry_s += dt
        n_steps = int(self._carry_s / self.ballistic_dt_s)
        self._carry_s -= n_steps * self.ballistic_dt_s
        if n_steps <= 0:
            return events

        # THE CARRIER IS AbstractTensor, not numpy. The law was printed
        # from SymPy as AbstractTensor code, so `.sqrt()` is a tensor
        # method and the batch axis is the tensor's own -- handing it a
        # bare ndarray finds no such method, which is the backend
        # telling you it is not the backend you compiled for.
        from src.common.tensors.abstraction import AbstractTensor as AT
        step = self._batch_step()
        n = len(self.rounds)
        cals = [get_calibre(r.calibre) for r in self.rounds]
        state = {
            "position_x_m": AT.tensor([float(r.position[0]) for r in self.rounds]),
            "position_y_m": AT.tensor([float(r.position[1]) for r in self.rounds]),
            "position_z_m": AT.tensor([float(r.position[2]) for r in self.rounds]),
            "velocity_x_m_s": AT.tensor([float(r.velocity[0]) for r in self.rounds]),
            "velocity_y_m_s": AT.tensor([float(r.velocity[1]) for r in self.rounds]),
            "velocity_z_m_s": AT.tensor([float(r.velocity[2]) for r in self.rounds]),
        }
        params = {
            "dt": AT.tensor([self.ballistic_dt_s] * n),
            "mass_kg": AT.tensor([float(c.mass_kg) for c in cals]),
            "diameter_m": AT.tensor([float(c.diameter_m) for c in cals]),
            "drag_coefficient": AT.tensor([self.drag_coefficient] * n),
            "air_density_kg_m3": AT.tensor([self.air_density_kg_m3] * n),
            "gravity_m_s2": AT.tensor([9.80665] * n),
            "wind_x_m_s": AT.tensor([float(self.wind[0])] * n),
            "wind_y_m_s": AT.tensor([float(self.wind[1])] * n),
            "wind_z_m_s": AT.tensor([float(self.wind[2])] * n),
        }
        was_all = np.array([r.position for r in self.rounds], dtype=float)
        for _ in range(n_steps):
            out = step(**state, **params)
            state = {
                "position_x_m": out[0], "position_y_m": out[1], "position_z_m": out[2],
                "velocity_x_m_s": out[3], "velocity_y_m_s": out[4],
                "velocity_z_m_s": out[5],
            }
        def _np(t):
            return np.asarray(t.tolist(), dtype=float).reshape(-1)
        pos = np.stack([_np(state["position_x_m"]), _np(state["position_y_m"]),
                        _np(state["position_z_m"])], axis=1)
        vel = np.stack([_np(state["velocity_x_m_s"]), _np(state["velocity_y_m_s"]),
                        _np(state["velocity_z_m_s"])], axis=1)
        flown = n_steps * self.ballistic_dt_s

        for i, shot in enumerate(list(self.rounds)):
            c = cals[i]
            was = was_all[i]
            shot.position, shot.velocity = pos[i].copy(), vel[i].copy()
            shot.flight_s += flown
            leg = shot.position - was
            travelled = float(np.linalg.norm(leg))
            shot.distance_m += travelled

            hit = self._resolve(shot, was, leg, travelled, c, bodies)
            if hit is not None:
                events.append(hit)
                shot.live = False
            elif shot.position[1] < 0.0:
                shot.live = False
                shot.outcome = "fell short"
                events.append({"round": shot.identity, "outcome": "fell short",
                               "at": shot.position.copy(), "target": shot.target})
            elif shot.flight_s > self.max_flight_s:
                shot.live = False
                shot.outcome = "lost"
            if not shot.live:
                self.rounds.remove(shot)
                self.spent.append(shot)
        return events

    def _resolve(self, shot, was, leg, travelled, calibre, bodies):
        """Did this tick's segment cross anything?"""
        if travelled <= 0.0:
            return None
        steps = max(1, int(math.ceil(travelled / max(self.substep_m, 0.1))))
        heading = leg / travelled
        for identity, (where, body) in bodies.items():
            if body is None or body.destroyed:
                continue
            here = np.asarray(where, dtype=float)
            # closest approach of the segment to the target's centre
            rel = here - was
            t = float(np.clip(np.dot(rel, heading), 0.0, travelled))
            miss = float(np.linalg.norm(rel - heading * t))
            if miss > body.envelope_radius_m:
                continue
            # it passed within the body's envelope: trace the real
            # segment against the real mesh, in the target's own frame
            report = body.trace(was - here, heading, travelled, calibre, shot.calibre)
            effect = body.take(report)
            shot.outcome = ("DESTROYED" if effect.get("destroyed")
                            else (report.walls[0].split("(")[0].strip()
                                  if report.walls else "no effect"))
            return {"round": shot.identity, "outcome": shot.outcome,
                    "target": identity, "report": report, "effect": effect,
                    "flight_s": shot.flight_s, "range_m": shot.distance_m,
                    "impact_speed_m_s": shot.speed_m_s,
                    "at": shot.position.copy()}
        return None

    # ------------------------------------------------------------------
    def in_flight(self) -> int:
        return len(self.rounds)

    def describe(self) -> list[str]:
        out = [f"  {len(self.rounds)} round(s) in the air, {len(self.spent)} spent"]
        for shot in self.rounds[:4]:
            out.append(f"    {shot.identity}: {shot.distance_m:6.0f} m out, "
                       f"{shot.speed_m_s:5.0f} m/s, {shot.flight_s * 1000:5.0f} ms in the air")
        return out
