"""Things worth shooting at, built as real objects.

A target with no body cannot be hit, and the demo proved it: hostiles
were an identity, a position and a velocity, so every round traced
against the station's OWN mesh, found nothing at the target, and
reported no effect. Nothing could ever be destroyed because there was
never anything there. The plan view showed contacts; the world
contained none.

So a hostile is a graph document like everything else here -- the same
vocabulary an engine or a turret is written in -- and a shot resolves
against it with the penetration model that already exists. Then
"terminated" is not a rule someone wrote, it is what happens when a
round defeats something the vehicle cannot run without.

WHAT MAKES A KILL IS DECLARED, NOT GUESSED. Each body says whether
losing it stops the vehicle (`critical=True`) and how much punishment it
takes. A hull plate is not critical and a powerpack is; neither is
decided by looking at its name.

The plating is real: RHA-equivalent thicknesses for a light armoured
vehicle, which is what an anti-material round is for. Frontal arc heavier
than the sides, roof and belly thinnest, because that is how armour is
distributed when there is a fixed mass to spend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Rolled homogeneous armour equivalents, in millimetres, for a light
#: armoured vehicle -- the class an anti-material round is meant for.
PLATE_MM = {"glacis": 32.0, "side": 16.0, "rear": 12.0, "roof": 8.0, "belly": 10.0}


def target_vehicle(identity: str = "contact", *, kind: str = "light-armour") -> dict:
    """A small armoured vehicle as a graph document, at the origin.

    Built at the ORIGIN on purpose. A hostile moves, and rebuilding its
    mesh every time it did would cost more than the rest of the
    simulation; instead the mesh is built once here and each shot is
    transformed into the target's own frame before it is traced, which
    is exact and free. The same trick the mount's articulation uses.
    """
    nodes, edges = [], []

    # THE BODY IS CENTRED ON ITS OWN ORIGIN. Authored from the ground
    # up because that is how a vehicle is measured, then shifted so the
    # origin lands in the middle of the hull -- because the origin is
    # what a fire-control system aims AT. Left on the ground, every
    # round passed cleanly underneath the vehicle and reported no
    # effect, which is exactly what it should report when you shoot at
    # the dirt.
    HULL_CENTRE_Y = 0.9

    def node(name, pos, kind_, half, material, *, critical=False, **kw):
        pos = (pos[0], pos[1] - HULL_CENTRE_Y, pos[2])
        nodes.append({"identity": f"{identity}.{name}", "kind": kind_,
                      "reference_position": [float(v) for v in pos],
                      "body_half_extent_m": [float(v) for v in half],
                      "material": material, "in_view": True,
                      "critical": bool(critical), **kw})

    def edge(name, a, b, constraint="rigid-distance", radius=0.02):
        edges.append({"identity": f"{identity}.{name}", "a": f"{identity}.{a}",
                      "b": f"{identity}.{b}", "constraint": constraint,
                      "radius": radius, "material": "steel-plate", "in_view": True})

    # THE HULL. Plates, not a box: a round that goes through the side
    # and a round that stops on the glacis are different events, and
    # they can only be different if the plates are different bodies.
    # THE GEOMETRY IS THE ARMOUR. There is no separate thickness
    # attribute to declare, because the penetration model traces the
    # MESH -- so a plate's half-extent across its own face IS how much
    # steel the round must cross. Writing `armour_mm=16` beside a
    # half-extent of 0.05 m did not give this vehicle 16 mm sides, it
    # gave it a hundred millimetres of rolled homogeneous armour, and a
    # 40 mm round quite correctly failed to get through any of it.
    def plate(name, pos, half, face):
        """`face` is which axis the plate is thin across."""
        t = PLATE_MM[face] / 1000.0 / 2.0        # half-extent = half the thickness
        half = list(half)
        half["xyz".index({"glacis": "z", "side": "x", "rear": "z",
                          "roof": "y", "belly": "y"}[face])] = t
        node(name, pos, "armour-plate", tuple(half), "armour-plate",
             armour_mm=PLATE_MM[face], face=face)

    plate("glacis", (0.0, 0.9, 2.1), (1.05, 0.55, 0.0), "glacis")
    plate("side_l", (-1.05, 0.9, 0.0), (0.0, 0.55, 2.1), "side")
    plate("side_r", (1.05, 0.9, 0.0), (0.0, 0.55, 2.1), "side")
    plate("rear", (0.0, 0.9, -2.1), (1.05, 0.55, 0.0), "rear")
    plate("roof", (0.0, 1.45, 0.0), (1.05, 0.0, 2.1), "roof")
    plate("belly", (0.0, 0.35, 0.0), (1.05, 0.0, 2.1), "belly")

    # WHAT IT CANNOT RUN WITHOUT. Declared critical, so a kill is the
    # damage model's own answer rather than a threshold someone picked.
    node("powerpack", (0.0, 0.85, 1.35), "engine-block", (0.62, 0.42, 0.55),
         "cast-iron", critical=True,
         note="engine and transmission in one bay, forward under the glacis")
    node("fuel", (0.0, 0.7, -1.25), "fluid-reservoir", (0.55, 0.30, 0.40),
         "steel-plate", critical=True, fluid="diesel", fluid_volume_l=180.0,
         note="a full tank is not an instant kill, but a holed one ends the day")
    node("crew", (0.0, 1.0, -0.2), "crew-station", (0.45, 0.45, 0.55),
         "steel-plate", critical=True)

    # running gear: not critical to KILL, but a mobility hit is real
    for side, sx in (("l", -1.0), ("r", 1.0)):
        for k, z in enumerate((1.45, 0.0, -1.45)):
            node(f"wheel_{side}{k}", (sx * 1.15, 0.45, z), "road-wheel",
                 (0.12, 0.45, 0.45), "rubber", mobility=True)
            edge(f"hub_{side}{k}", f"wheel_{side}{k}", f"side_{side}")

    for a, b in (("glacis", "side_l"), ("glacis", "side_r"), ("rear", "side_l"),
                 ("rear", "side_r"), ("roof", "side_l"), ("roof", "side_r"),
                 ("belly", "side_l"), ("belly", "side_r"),
                 ("powerpack", "belly"), ("fuel", "belly"), ("crew", "belly")):
        edge(f"{a}_to_{b}", a, b, radius=0.03)

    return {"schema": "engine-toy-drivetrain-graph-v1",
            "identity": f"{identity}/target", "machine": True,
            "label": f"{kind} target", "nodes": nodes, "edges": edges}


@dataclass
class TargetBody:
    """A hostile's physical self: its graph, its mesh, and its wounds.

    The gun is armed against THIS graph, so the penetration model is
    resolving against the target instead of against the shooter, which
    is the defect this class exists to correct."""
    identity: str = "contact"
    graph: dict = None
    gun: object = None
    kind: str = "light-armour"
    destroyed: bool = False
    mobility_killed: bool = False
    hits: int = 0
    penetrations: int = 0
    killed_by: str = ""

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = target_vehicle(self.identity, kind=self.kind)
        self._critical = {n["identity"] for n in self.graph["nodes"] if n.get("critical")}
        self._mobility = {n["identity"] for n in self.graph["nodes"] if n.get("mobility")}
        self._defeated: set = set()
        self._mobility_lost: set = set()
        self._envelope = None
        self._impact_speed = 0.0

    def arm(self, calibre: str):
        """THE SHOOTER DECIDES THE ROUND, not the target.

        Arming the body once with a fixed calibre made the target decide
        what hit it: the station believed it was firing .50 AP while
        damage resolved as a 40 mm cannon shell, because that is what
        the body happened to have been armed with. Guns are cached per
        calibre and selected at the moment of firing instead, so a
        station with a cannon AND a machine gun gets the right answer
        from whichever one pulled."""
        if not hasattr(self, "_guns"):
            self._guns = {}
        if calibre not in self._guns:
            from world_gun import WorldGun
            self._guns[calibre] = WorldGun(graph=self.graph, calibre=calibre).arm()
        self.gun = self._guns[calibre]
        return self.gun

    # ------------------------------------------------------------------
    @property
    def envelope_radius_m(self) -> float:
        """A sphere that certainly contains this body, for the cheap
        broad-phase test. Derived from the geometry, so it cannot drift
        away from the thing it is supposed to bound."""
        if getattr(self, "_envelope", None) is None:
            import numpy as np
            worst = 0.0
            for n in self.graph["nodes"]:
                p = np.asarray(n["reference_position"], float)
                h = np.asarray(n["body_half_extent_m"], float)
                worst = max(worst, float(np.linalg.norm(np.abs(p) + h)))
            self._envelope = worst
        return self._envelope

    def trace(self, from_local, heading, length_m, calibre, calibre_name: str):
        """Trace the segment a round ACTUALLY travelled, in this body's
        own frame.

        No teleporting to the vicinity and firing inward: the caller
        passes where the round was and which way it was going, both
        already brought into this body's local frame, and the real
        segment is what gets traced. That is the difference between
        simulating a projectile and asserting that one arrived."""
        import numpy as np
        from engine_rays import Ray
        from ballistics import ProjectileState
        from world_gun import record_penetration, ShotReport
        gun = self.arm(calibre_name)
        start = np.asarray(from_local, dtype=float)
        d = np.asarray(heading, dtype=float)
        arriving = ProjectileState(
            mass_kg=calibre.mass_kg, diameter_m=calibre.diameter_m,
            speed_m_s=self._impact_speed, direction=tuple(float(x) for x in d),
            length_m=calibre.length_m, yaw_rad=calibre.yaw_rad,
            hardness_pa=calibre.hardness_pa)
        pen = gun._mesh.penetrate(Ray.from_points(start, start + d * max(length_m, 1.0)),
                                  energy_j=arriving.energy_j,
                                  calibre_m=calibre.diameter_m, projectile=arriving)
        recorded = record_penetration(gun.damage_states, self.graph, (), pen)
        report = ShotReport(calibre=f"{calibre_name} @ impact",
                            muzzle_energy_j=arriving.energy_j,
                            walls=list(pen.log), punctures=list(recorded))
        report.arrival_speed_m_s = self._impact_speed
        for line in pen.log:
            if line.startswith("stopped in "):
                report.stopped_in = line.split()[2]
        report.exited = bool(pen.log) and report.stopped_in is None
        return report

    def take(self, report) -> dict:
        """Absorb a shot report and decide what it cost.

        A kill is the consequence of a critical body being punctured
        THROUGH -- not of a hit count, and not of a name. A round that
        stops in the glacis has done nothing but scratch the paint,
        which is exactly what should happen to an under-powered round
        against a frontal arc."""
        self.hits += 1
        for identity, puncture in getattr(report, "punctures", ()):
            if not getattr(puncture, "through", False):
                continue
            self.penetrations += 1
            if identity in self._critical:
                self._defeated.add(identity)
            if identity in self._mobility:
                self._mobility_lost.add(identity)
        if self._defeated and not self.destroyed:
            self.destroyed = True
            self.killed_by = ", ".join(sorted(n.split(".")[-1] for n in self._defeated))
        # half the running gear gone and it is not going anywhere
        if len(self._mobility_lost) >= max(1, len(self._mobility) // 2):
            self.mobility_killed = True
        return {"destroyed": self.destroyed, "mobility_killed": self.mobility_killed,
                "defeated": sorted(self._defeated), "killed_by": self.killed_by}

    def status(self) -> str:
        if self.destroyed:
            return f"DESTROYED ({self.killed_by})"
        if self.mobility_killed:
            return "mobility kill"
        if self.penetrations:
            return f"{self.penetrations} penetration(s)"
        if self.hits:
            return f"{self.hits} hit(s), no penetration"
        return "undamaged"
