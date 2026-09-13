"""Two weapons on one station, and a director that decides who shoots
what.

The arrangement is the ordinary one and it works for a reason: a MAIN
GUN that is slow, heavy and decisive, and a SECONDARY on its own swivel
that is fast, light and always available. Putting them on one mount and
one solver is not a compromise -- it is what lets the station answer
two different problems at once, because the problems genuinely differ:

  THE CANNON is aimed carefully at things worth a shell. It is slow to
  traverse because it is heavy, its ammunition is finite and expensive,
  and every engagement costs barrel life. When it has nothing to do it
  should be PARKED somewhere useful rather than left wherever the last
  target happened to be -- pointed down the most likely approach, so the
  next engagement starts with a shorter slew.

  THE MACHINE GUN is on its own swivel and does not wait for permission.
  It is fast, its ammunition is cheap, and its useful range is a
  fraction of the cannon's. It takes whatever the cannon is not taking,
  which is most things.

  ONE SOLVER SERVES BOTH. The ballistics differ -- different tables,
  different flight times, different leads -- but the target state is the
  same, and computing it twice would be computing it twice. The director
  allocates, each mount asks for its own solution against its own
  table, and both run asynchronously.

ALLOCATION IS NOT "NEAREST FIRST"

Nearest-first is the rule everyone writes and it is wrong in the way
that matters: it sends the cannon after a slow truck at four hundred
metres while something fast closes from twelve hundred. The threat
ordering here is what a real director uses -- how soon could this thing
hurt us -- which is closing speed against range, not range alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _bearing_elevation(vector) -> tuple:
    v = np.asarray(vector, dtype=float)
    horizontal = math.hypot(float(v[0]), float(v[2]))
    return (math.degrees(math.atan2(float(v[0]), float(v[2]))),
            math.degrees(math.atan2(float(v[1]), max(horizontal, 1e-12))))


def _angle_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _approach(current: float, goal: float, step: float, *, wrap: bool = False) -> float:
    delta = _angle_difference(goal, current) if wrap else (goal - current)
    if abs(delta) <= step:
        return current + delta
    return current + math.copysign(step, delta)


@dataclass
class WeaponMount:
    """One weapon with its own articulation, table and solver."""
    identity: str
    calibre: str
    table: object
    muzzle_offset: tuple = (0.0, 0.0, 0.0)     # from the station's own origin
    traverse_rate_deg_s: float = 24.0
    elevation_rate_deg_s: float = 12.0
    elevation_limits_deg: tuple = (-8.0, 42.0)
    minimum_range_m: float = 0.0
    maximum_range_m: float = 2500.0
    rounds: int = 120
    cyclic_rate_per_min: float = 120.0
    park_bearing_deg: float = 0.0
    park_elevation_deg: float = 2.0
    # live
    bearing_deg: float = 0.0
    elevation_deg: float = 0.0
    target: object = None
    service: object = None
    tracking_error_deg: float = 180.0
    on_target: bool = False
    parked: bool = True
    reload_s: float = 0.0
    fired: int = 0

    def can_engage(self, station_origin, hostile) -> bool:
        """Is this thing inside what this weapon can actually do?"""
        muzzle = np.asarray(station_origin, dtype=float) + np.asarray(self.muzzle_offset, float)
        offset = np.asarray(hostile.position, dtype=float) - muzzle
        distance = float(np.linalg.norm(offset))
        if not (self.minimum_range_m <= distance <= self.maximum_range_m):
            return False
        _, elevation = _bearing_elevation(offset)
        low, high = self.elevation_limits_deg
        return low - 1.0 <= elevation <= high + 1.0

    def attach_service(self, service) -> "WeaponMount":
        self.service = service
        service.start()
        return self

    def muzzle(self, station_origin) -> np.ndarray:
        return np.asarray(station_origin, dtype=float) + np.asarray(self.muzzle_offset, float)

    def step(self, dt: float, station_origin, crosswind_m_s: float = 0.0) -> dict:
        """Slew toward whatever it has been given, or toward its park."""
        muzzle = self.muzzle(station_origin)
        self.reload_s = max(0.0, self.reload_s - dt)
        if self.target is None:
            self.parked = True
            want_bearing, want_elevation = self.park_bearing_deg, self.park_elevation_deg
            if self.service is not None:
                self.service.update(valid=False)
        else:
            self.parked = False
            if self.service is not None:
                self.service.update(
                    target_identity=self.target.identity,
                    target_position=tuple(float(v) for v in self.target.position),
                    target_velocity=tuple(float(v) for v in self.target.velocity),
                    muzzle_position=tuple(float(v) for v in muzzle),
                    crosswind_m_s=float(crosswind_m_s), valid=True)
                plan = self.service.latest() or self.service.solve_once()
                if plan is None:
                    want_bearing, want_elevation = self.bearing_deg, self.elevation_deg
                else:
                    import time as _time
                    want_bearing, want_elevation = plan.bearing_elevation_now(
                        _time.monotonic(), muzzle)
            else:
                want_bearing, want_elevation = _bearing_elevation(
                    np.asarray(self.target.position, dtype=float) - muzzle)

        low, high = self.elevation_limits_deg
        want_elevation = max(low, min(high, want_elevation))
        self.bearing_deg = _approach(self.bearing_deg, want_bearing,
                                     self.traverse_rate_deg_s * dt, wrap=True)
        self.elevation_deg = _approach(self.elevation_deg, want_elevation,
                                       self.elevation_rate_deg_s * dt)
        self.tracking_error_deg = math.hypot(
            _angle_difference(want_bearing, self.bearing_deg),
            want_elevation - self.elevation_deg)
        self.on_target = bool(self.target is not None and self.tracking_error_deg < 0.08)
        return {
            "identity": self.identity, "calibre": self.calibre,
            "bearing_deg": round(self.bearing_deg, 3),
            "elevation_deg": round(self.elevation_deg, 3),
            "solution_bearing_deg": round(want_bearing, 3),
            "target": None if self.target is None else self.target.identity,
            "parked": self.parked, "on_target": self.on_target,
            "error_deg": round(self.tracking_error_deg, 3),
            "rounds": self.rounds, "ready": self.reload_s <= 0.0,
        }

    def take_shot(self) -> bool:
        if self.rounds <= 0 or self.reload_s > 0.0 or not self.on_target:
            return False
        self.rounds -= 1
        self.fired += 1
        self.reload_s = 60.0 / max(self.cyclic_rate_per_min, 1e-6)
        return True


# =====================================================================
#  THE DIRECTOR
# =====================================================================
@dataclass
class FireDirector:
    """Decides which weapon takes which target, every tick."""
    station_origin: tuple = (0.0, 0.0, 0.0)
    main: WeaponMount = None
    secondary: WeaponMount = None
    crosswind_m_s: float = 0.0
    log: list = field(default_factory=list)

    def threat_score(self, hostile) -> float:
        """How soon could this hurt us.

        Closing speed against range, so a fast contact at a kilometre
        outranks a slow one at four hundred metres -- which is the whole
        reason not to allocate by distance."""
        origin = np.asarray(self.station_origin, dtype=float)
        offset = np.asarray(hostile.position, dtype=float) - origin
        distance = max(float(np.linalg.norm(offset)), 1.0)
        unit = offset / distance
        closing = -float(np.dot(np.asarray(hostile.velocity, dtype=float), unit))
        # seconds until arrival, if it keeps closing; enormous if it is not
        if closing <= 0.1:
            return distance / 5.0          # opening or crossing: low urgency
        return distance / closing

    def allocate(self, hostiles) -> dict:
        """Give the cannon the worst problem it can reach, and let the
        machine gun take the best of what is left."""
        ranked = sorted((h for h in hostiles if h.label == "hostile"),
                        key=self.threat_score)
        assignment = {"main": None, "secondary": None}
        for h in ranked:
            if self.main and self.main.can_engage(self.station_origin, h):
                assignment["main"] = h
                break
        for h in ranked:
            if h is assignment["main"]:
                continue
            if self.secondary and self.secondary.can_engage(self.station_origin, h):
                assignment["secondary"] = h
                break
        # THE OPPORTUNISTIC CASE. If the cannon has taken the only thing
        # in reach, the machine gun does not sit idle staring at it -- it
        # takes the same target, because a second weapon on a target is
        # worth more than a second weapon on nothing.
        if assignment["secondary"] is None and assignment["main"] is not None:
            if self.secondary and self.secondary.can_engage(self.station_origin,
                                                            assignment["main"]):
                assignment["secondary"] = assignment["main"]
        return assignment

    def step(self, dt: float, hostiles) -> dict:
        assignment = self.allocate(hostiles)
        for role, mount in (("main", self.main), ("secondary", self.secondary)):
            if mount is None:
                continue
            chosen = assignment[role]
            if chosen is not mount.target:
                if chosen is None:
                    self.note(f"{mount.identity}: nothing in reach, parking at "
                              f"{mount.park_bearing_deg:.0f} deg")
                else:
                    self.note(f"{mount.identity}: taking {chosen.identity} "
                              f"(threat {self.threat_score(chosen):.1f} s)")
                mount.target = chosen
        states = {}
        for role, mount in (("main", self.main), ("secondary", self.secondary)):
            if mount is not None:
                states[role] = mount.step(dt, self.station_origin, self.crosswind_m_s)
        return {"assignment": {k: (v.identity if v else None) for k, v in assignment.items()},
                "mounts": states}

    def note(self, line: str) -> None:
        self.log.append(line)
        del self.log[:-10]

    def summary(self) -> list[str]:
        out = []
        for mount in (self.main, self.secondary):
            if mount is None:
                continue
            out.append(f"  {mount.identity:14s} {mount.calibre:10s} "
                       f"brg {mount.bearing_deg:7.2f}  elev {mount.elevation_deg:6.2f}  "
                       + (f"PARKED" if mount.parked else
                          f"on {mount.target.identity}"
                          f"{'  ON TARGET' if mount.on_target else f'  err {mount.tracking_error_deg:5.2f}'}")
                       + f"   {mount.rounds:4d} rds, {mount.fired} fired")
        out.extend(f"    {line}" for line in self.log[-4:])
        return out
