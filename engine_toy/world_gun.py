"""A ballistic test gun, as a tool the player carries in the world.

The world already has a gun: `abstract_ui_projectiles.gun_tool` fires
physics balls, which is how material reaches the validator rig's hopper.
This is its sibling for a different job -- a gun of configurable calibre
whose shots RESOLVE AGAINST GEOMETRY rather than spawning a body, so you
can walk up to an object on the rig's stage and find out what it
actually stops.

WHY THIS EXTRACTS CLEANLY, which was the open question:

  ballistics.py is pure standard library -- dataclasses, enum, math.
  It knows about projectiles, materials and what happens when one meets
  the other, and it knows nothing about engines.

  engine_rays.py is that plus NumPy, and its only couplings back into
  the engine simulation are three LAZY imports inside functions.

  engine_mesh builds its triangle soup from a graph document, and a
  world object is a graph document -- the same node/edge shape the
  production graph uses. Nothing has to be converted.

So the weapon concept was never engine-specific. It is a ray, a
material table and an energy budget, and any object with geometry can
be shot at.

WHAT A SHOT ACTUALLY DOES, and why it is worth having as a test tool:
the projectile spends energy through successive walls, each traversal
costing thickness times calibre area times that material's toughness.
A wall it gets through is holed in and out; the wall that absorbs the
rest is holed on the way in only and the projectile stops there. So the
report is not "hit" or "miss" -- it is which parts were reached, at
what remaining speed, and where it finally stopped. That is exactly
what you want to know about a structure you have just built.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from calibres import CALIBRES, get_calibre
from engine_rays import Ray, RayMesh
from engine_mesh import build_engine_mesh
from damage_state import record_penetration

# The calibres worth carrying on a test gun: a spread from something
# that will not get through a casting to something that goes through
# most of a machine, so a single walk-around tells you where the real
# thresholds are.
TEST_CALIBRES = ("9mm", ".45 acp", ".500 s&w", "7.62 nato", ".50 bmg", ".50 ap")


def ballistic_gun_tool(identity: str, calibres: tuple = TEST_CALIBRES):
    """The tool as the world's own tool vocabulary defines it.

    One MODE per calibre, because that is what a mode is for: the same
    tool doing the same thing with a different character, selected
    without leaving what you are doing. The primary action fires; the
    secondary cycles the calibre, so the whole tool is usable with two
    buttons while walking around an object."""
    from src.compiler.abstract_ui_tools import AbstractUITool, ToolHook, ToolMode

    modes = []
    for name in calibres:
        c = get_calibre(name)
        modes.append(ToolMode(
            f"{identity}/modes/{name.replace(' ', '-').replace('.', '')}",
            name,
            f"{c.mass_kg * 1000:.0f} g at {c.muzzle_m_s:.0f} m/s, "
            f"{c.muzzle_energy_j / 1000:.2f} kJ, {c.diameter_m * 1000:.2f} mm bore"
            + (f" -- {c.note}" if c.note else ""),
            "cycle-calibre", "fire-ballistic-ray"))
    return AbstractUITool(
        identity, "Ballistic test gun",
        (ToolHook("primary-action", "fire-ballistic-ray", identity),
         ToolHook("secondary-action", "cycle-calibre", identity)),
        modes=tuple(modes),
        default_mode=calibres[0] if calibres else None,
    )


# =====================================================================
#  ATTACHMENTS -- what the gun is carrying
# =====================================================================
@dataclass
class Attachment:
    """One thing fitted to the gun. A slot holds at most one."""
    identity: str
    slot: str                     # "optic" | "muzzle" | "mount" | "magazine"
    name: str
    note: str = ""

    def to_inventory_item(self, owner: str):
        """The same object as the world's own inventory sees it, so an
        attachment is a thing you carry and fit rather than a flag."""
        from src.compiler.abstract_ui_tools import InventoryItem
        return InventoryItem(f"{owner}/attachments/{self.slot}",
                             self.identity, self.name, is_tool=False)


@dataclass
class Optic(Attachment):
    """A sight, and the cartridge it was zeroed with.

    THE CARTRIDGE MATTERS. A zero is a launch angle solved for one
    trajectory; put a different cartridge behind the same sight and the
    zero is simply wrong, by more the further you shoot. The gun checks
    this rather than letting it pass silently, because in the real world
    it is the commonest reason a scoped rifle misses."""
    scope: object = None
    zeroed_for: str = ""

    def solution_for(self, range_m: float, wind):
        return self.scope.solution(range_m, wind=wind)


def iron_sights() -> Attachment:
    return Attachment("attachments/iron-sights", "optic", "Iron sights",
                      note="no compensation: you hold over by eye")


def telescopic_sight(calibre: str, *, zero_range_m: float = 100.0,
                     magnification: float = 10.0) -> Optic:
    """Fit a scope, zeroed for this cartridge at this range."""
    from scope import scoped
    s = scoped(calibre, zero_range_m=zero_range_m, magnification=magnification)
    return Optic(identity=f"attachments/scope-{magnification:.0f}x",
                 slot="optic", name=f"{magnification:.0f}x telescopic sight",
                 note=f"zeroed for {calibre} at {zero_range_m:.0f} m",
                 scope=s, zeroed_for=calibre)


# =====================================================================
#  THE SHOT
# =====================================================================
@dataclass
class ShotReport:
    """What a shot found out. Deliberately a report rather than a
    verdict: which parts it reached, at what speed, and where it
    stopped."""
    calibre: str
    muzzle_energy_j: float
    walls: list = field(default_factory=list)      # the traversal log
    punctures: list = field(default_factory=list)  # (identity, Puncture)
    stopped_in: str | None = None
    exited: bool = False
    arrival_speed_m_s: float = 0.0
    drop_m: float = 0.0
    drift_m: float = 0.0
    solution: object = None

    def lines(self) -> list[str]:
        out = [f"  {self.calibre}: {self.muzzle_energy_j / 1000:.2f} kJ, "
               f"{len(self.punctures)} puncture(s)"]
        out.extend(f"    {line}" for line in self.walls[:8])
        if not self.walls:
            # A traced shot that reaches nothing has MISSED, and saying
            # so matters: at any real range the round is well below the
            # line of sight by the time it arrives, so a gun aimed
            # straight at a target hits the ground in front of it. That
            # is the whole reason sights have elevation.
            out.append(f"    MISSED -- {self.drop_m * 100:+.0f} cm of drop at arrival"
                       if self.drop_m else "    MISSED")
        elif self.stopped_in:
            out.append(f"    stopped in {self.stopped_in}")
        elif self.exited:
            out.append("    passed clean through")
        return out


@dataclass
class WorldGun:
    """A gun pointed at one world object.

    It holds the object's mesh rather than rebuilding it per shot, and
    it accumulates damage on that object, so a magazine's worth of
    shots is a cumulative test rather than a series of unrelated ones."""
    graph: dict
    calibre: str = "7.62 nato"
    damage_states: dict = field(default_factory=dict)
    attachments: dict = field(default_factory=dict)     # slot -> Attachment
    _mesh: object = None

    # ---------------- what it is carrying ----------------
    def attach(self, attachment) -> "WorldGun":
        self.attachments[attachment.slot] = attachment
        return self

    def detach(self, slot: str):
        return self.attachments.pop(slot, None)

    @property
    def optic(self):
        return self.attachments.get("optic")

    def loadout(self) -> list[str]:
        out = [f"  {self.calibre}"]
        for slot in ("optic", "muzzle", "mount", "magazine"):
            a = self.attachments.get(slot)
            out.append(f"    {slot:9s} {a.name if a else '-- empty --'}"
                       + (f"   ({a.note})" if a and a.note else ""))
        o = self.optic
        if o is not None and getattr(o, "zeroed_for", "") and o.zeroed_for != self.calibre:
            out.append(f"    ! the sight is zeroed for {o.zeroed_for}, not {self.calibre} -- "
                       f"the zero is wrong and gets worse with range")
        return out

    def arm(self) -> "WorldGun":
        static_mesh, moving_mesh = build_engine_mesh(self.graph, covers_off=False)
        self._mesh = RayMesh(static_mesh, moving_mesh)
        return self

    def set_calibre(self, name: str) -> None:
        get_calibre(name)          # raises if it is not a real cartridge
        self.calibre = name

    def cycle_calibre(self, calibres: tuple = TEST_CALIBRES) -> str:
        i = calibres.index(self.calibre) if self.calibre in calibres else -1
        self.calibre = calibres[(i + 1) % len(calibres)]
        return self.calibre

    def fire(self, origin, aim_at) -> ShotReport:
        """One shot, from a point, at a point.

        The projectile carries its own mass, bore, length, core hardness
        and yaw, because those are what decide whether it gets through
        something -- energy alone does not. A .50 ball and a .50 AP
        round arrive with the same energy and do not do the same thing."""
        if self._mesh is None:
            self.arm()
        origin = np.asarray(origin, dtype=float)
        target = np.asarray(aim_at, dtype=float)
        direction = target - origin
        distance = float(np.linalg.norm(direction))
        c = get_calibre(self.calibre)
        shot = c.projectile(direction, range_m=max(distance, 0.0))
        pen = self._mesh.penetrate(Ray.from_points(origin, target),
                                   energy_j=shot.energy_j, calibre_m=c.diameter_m,
                                   projectile=shot)
        recorded = record_penetration(self.damage_states, self.graph, (), pen)
        report = ShotReport(calibre=self.calibre, muzzle_energy_j=c.muzzle_energy_j,
                            walls=list(pen.log), punctures=list(recorded))
        for line in pen.log:
            if line.startswith("stopped in "):
                report.stopped_in = line.split()[2]
        report.exited = bool(pen.log) and report.stopped_in is None
        return report

    def fire_scoped(self, origin, aim_at, *, wind=(0.0, 0.0, 0.0), dt: float = 5e-4) -> ShotReport:
        """Fire using the fitted optic's own firing solution.

        This is what an optic is FOR, and why fitting one changes the
        outcome rather than the view: the scope solves the trajectory,
        returns the elevation and windage the shot needs, and the gun
        applies them to the aim. Without it the shooter is holding
        straight at a target the bullet will pass a metre under.

        With iron sights or an empty optic slot it falls through to the
        uncorrected shot, which is the honest behaviour -- the
        correction has to come from somewhere, and by eye is not
        something this can do for you."""
        optic = self.optic
        origin = np.asarray(origin, dtype=float)
        target = np.asarray(aim_at, dtype=float)
        if optic is None or not isinstance(optic, Optic):
            return self.fire_from_range(origin, target, wind=wind, dt=dt)

        direction = target - origin
        range_m = float(np.linalg.norm(direction))
        unit = direction / max(range_m, 1e-9)
        sol = optic.solution_for(range_m, wind)

        # apply the solution AS ANGLES, which is what a scope produces:
        # up by the elevation, across by the windage, both in the
        # shooter's own frame rather than the world's
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(unit, world_up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, unit)
        # THE ZERO IS PART OF THE AIM, not a separate thing. A scoped
        # rifle's bore already points above the line of sight by the
        # zero angle before any dialling happens, and the solution's
        # elevation is what you add ON TOP of that. Applying only the
        # dialled correction to a bore pointed flat at the target left
        # the shot 43 cm low at 400 m -- it hit, but half a metre under
        # the aim point, which is a miss anywhere that matters.
        total_elevation_rad = optic.scope.zero_elevation_rad + sol.elevation_mrad / 1000.0
        # SIGN: `windage_mrad` is a CORRECTION, already the negative of
        # the drift, so applying it to the aim ADDS along `right`.
        # Subtracting it aimed further downwind and doubled the miss --
        # 89 cm of lateral error at 400 m, which read as a vertical
        # problem until the ray was actually traced and turned out to be
        # passing the target's plane 4.5 cm high and 89 cm to the side.
        corrected = target + up * (total_elevation_rad * range_m)                            + right * (sol.windage_mrad / 1000.0 * range_m)
        report = self.fire_from_range(origin, corrected, wind=wind, dt=dt)
        report.calibre = f"{self.calibre} @ {range_m:.0f} m (scoped)"
        report.solution = sol
        return report

    def fire_from_solution(self, origin, aim_at, *, table) -> ShotReport:
        """Fire using a BAKED ballistic table instead of integrating.

        The table already knows what the round is doing at this range,
        because that is what baking it was for. This turns a shot from
        a few hundred thousand integration steps into a lookup and one
        ray, which is what makes a firing rate possible at all."""
        if self._mesh is None:
            self.arm()
        origin = np.asarray(origin, dtype=float)
        target = np.asarray(aim_at, dtype=float)
        offset = target - origin
        distance = float(np.linalg.norm(offset))
        heading = offset / max(distance, 1e-9)
        c = get_calibre(self.calibre)
        speed = float(table.speed_at(distance))
        from ballistics import ProjectileState
        arriving = ProjectileState(
            mass_kg=c.mass_kg, diameter_m=c.diameter_m, speed_m_s=speed,
            direction=tuple(float(x) for x in heading), length_m=c.length_m,
            yaw_rad=c.yaw_rad, hardness_pa=c.hardness_pa)
        entry = target - heading * 3.0
        pen = self._mesh.penetrate(Ray.from_points(entry, entry + heading * 60.0),
                                   energy_j=arriving.energy_j, calibre_m=c.diameter_m,
                                   projectile=arriving)
        recorded = record_penetration(self.damage_states, self.graph, (), pen)
        report = ShotReport(calibre=f"{self.calibre} @ {distance:.0f} m",
                            muzzle_energy_j=arriving.energy_j,
                            walls=list(pen.log), punctures=list(recorded))
        report.arrival_speed_m_s = speed
        for line in pen.log:
            if line.startswith("stopped in "):
                report.stopped_in = line.split()[2]
        report.exited = bool(pen.log) and report.stopped_in is None
        return report

    def damage_summary(self) -> list[str]:
        """What this object has accumulated, by part."""
        out = []
        for identity, state in sorted(self.damage_states.items()):
            holes = len(state.punctures)
            through = sum(1 for p in state.punctures if p.through)
            energy = sum(i.energy_spent_j for i in state.impacts)
            out.append(f"    {identity.split('.')[-1]:26s} {holes:2d} hole(s), "
                       f"{through} through, {energy / 1000:6.2f} kJ absorbed")
        return out


    def fire_from_range(self, origin, aim_at, *, wind=(0.0, 0.0, 0.0),
                        dt: float = 2e-4, drag_coefficient: float = 0.295,
                        air_density: float = 1.225) -> ShotReport:
        """Trace the shot to the target first, then penetrate with what
        ACTUALLY ARRIVES.

        This is the pairing that makes a test gun honest at distance.
        Firing at muzzle energy pretends the target is at the muzzle; a
        rifle round has shed a third of its energy by three hundred
        metres, and whether it gets through a wall is decided by the
        velocity it turns up with, not the one it left with.

        The trajectory is the compiled symbolic law
        (symbolic_parts.symbolic_trajectory_equations) -- drag on
        velocity relative to the air, so drop and wind deflection both
        come out of the same integration rather than being added on
        afterwards."""
        from symbolic_parts import compile_trajectory_ssa
        from src.compiler.ssa_llvm_backend import (
            emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)

        origin = np.asarray(origin, dtype=float)
        target = np.asarray(aim_at, dtype=float)
        direction = target - origin
        distance = float(np.linalg.norm(direction))
        unit = direction / max(distance, 1e-9)
        c = get_calibre(self.calibre)

        compiled = compile_trajectory_ssa()
        fn = compiled.function
        art = _trajectory_artifact(compiled)
        ids = {n: v.id for n, v in zip(fn.metadata["argument_names"], fn.args)}
        outs = dict(fn.metadata["named_outputs"])

        p = origin.copy()
        v = unit * c.muzzle_m_s
        vals = {"dt": dt, "mass_kg": c.mass_kg, "diameter_m": c.diameter_m,
                "drag_coefficient": drag_coefficient, "air_density_kg_m3": air_density,
                "gravity_m_s2": 9.80665, "wind_x_m_s": wind[0],
                "wind_y_m_s": wind[1], "wind_z_m_s": wind[2]}
        # Stop on DOWNRANGE distance, not on path length. A trajectory
        # is a curve, so its arc length runs ahead of its ground
        # distance -- accumulating arc length stopped the flight early,
        # and it did so by more the more the path curved, which left
        # every long shot high by a growing amount.
        axis = unit
        for _ in range(200_000):
            vals.update({"position_x_m": p[0], "position_y_m": p[1], "position_z_m": p[2],
                         "velocity_x_m_s": v[0], "velocity_y_m_s": v[1], "velocity_z_m_s": v[2]})
            ex = prepare_artifact_execution(
                art, {ids[n]: np.array(x, dtype=np.float64) for n, x in vals.items()})
            ex.run()
            g = lambda n: float(np.asarray(ex.buffers[outs[n]]).reshape(-1)[0])
            nxt = np.array([g("position_next_x_m"), g("position_next_y_m"), g("position_next_z_m")])
            p, v = nxt, np.array([g("velocity_next_x_m_s"), g("velocity_next_y_m_s"),
                                  g("velocity_next_z_m_s")])
            if float(np.dot(p - origin, axis)) >= distance:
                break

        # the projectile as it arrives: real speed, real direction
        speed = float(np.linalg.norm(v))
        heading = v / max(speed, 1e-9)
        from ballistics import ProjectileState
        arriving = ProjectileState(
            mass_kg=c.mass_kg, diameter_m=c.diameter_m, speed_m_s=speed,
            direction=tuple(float(x) for x in heading), length_m=c.length_m,
            yaw_rad=c.yaw_rad, hardness_pa=c.hardness_pa)
        # aim the ray along the arriving heading, from just short of the target
        entry = p - heading * 2.0
        pen = self._mesh.penetrate(Ray.from_points(entry, entry + heading * 40.0),
                                   energy_j=arriving.energy_j, calibre_m=c.diameter_m,
                                   projectile=arriving)
        recorded = record_penetration(self.damage_states, self.graph, (), pen)
        report = ShotReport(calibre=f"{self.calibre} @ {distance:.0f} m",
                            muzzle_energy_j=arriving.energy_j,
                            walls=list(pen.log), punctures=list(recorded))
        for line in pen.log:
            if line.startswith("stopped in "):
                report.stopped_in = line.split()[2]
        report.exited = bool(pen.log) and report.stopped_in is None
        report.arrival_speed_m_s = speed
        report.drop_m = float(p[1] - target[1])
        report.drift_m = float(np.linalg.norm((p - target) - np.dot(p - target, unit) * unit)
                               - abs(p[1] - target[1]))
        return report


_TRAJECTORY_ARTIFACT = {}


def _trajectory_artifact(compiled):
    """Build the native trajectory kernel once and keep it."""
    if "artifact" not in _TRAJECTORY_ARTIFACT:
        from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm, compile_artifact
        _TRAJECTORY_ARTIFACT["artifact"] = compile_artifact(emit_ssa_function_to_llvm(
            compiled.module, compiled.function.name, entry_name="trajectory_step"))
    return _TRAJECTORY_ARTIFACT["artifact"]


def gun_for_world_object(graph: dict, calibre: str = "7.62 nato") -> WorldGun:
    """Point a test gun at any world object.

    The only requirement is that it is a graph document -- which every
    world object is."""
    return WorldGun(graph=graph, calibre=calibre).arm()
