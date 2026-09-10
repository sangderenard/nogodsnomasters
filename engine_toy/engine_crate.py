"""The real crate as one steppable, animated object -- tying together
three pieces that already exist, rather than a new parallel system:

  - the PRISM and the engine's own real connections to the frame
    (`EnginePackage`: a bounding volume over every BUILT_IN node, and
    `mounting` -- the real mount points, always real graph nodes that
    already sit inside that same prism or exactly on its boundary,
    since the prism is measured from the same BUILT_IN node set the
    mounts are drawn from -- never a separately invented position).
  - the animated frames living inside that prism (`EngineAnimation`,
    engine_mesh.py's own baked moving-part meshes) -- built off the
    SAME graph the package's own prism was measured from, so the two
    can never geometrically diverge.
  - ONE step function that advances the real live crank-domain sim
    (`EngineCycleSim`) by dt, indexes the already-baked animation frame
    for the resulting crank angle (never re-derives geometry -- see
    EngineAnimation.frame_for's own docstring), and hands back the
    real forces crossing the prism's own boundary at each mount point:
    this crate's own static wrench (its real weight through its real
    center of gravity, plus the crank's own real reaction torque --
    Newton's third law) resolved through engine_mounts.mount_loads.

This is the vehicle-graph side of the lease: the vehicle hands the
engine a prism (today: nothing real yet -- see EnginePackage.
correlate_mounts' own honest "no real cage source" note), the engine
answers with where ITS OWN real connections land and what load they
carry every tick; the vehicle never needs to know anything about
pistons, valves, or crank angle to use it.
"""
from __future__ import annotations

from dataclasses import dataclass

from drivetrain_graph import build_drivetrain_graph, engine_mesh_view_graph
from engine_cycle_sim import EngineCycleSim
from engine_mesh import EngineAnimation, build_animation
from engine_mounts import mount_loads
from engine_package import EnginePackage


@dataclass(frozen=True)
class CrateStepResult:
    crank_angle_deg: float
    animation_frame_index: int
    mount_loads: dict[str, tuple[float, float, float]]   # force each mount exerts on the frame, at the prism boundary


@dataclass
class EngineCrate:
    package: EnginePackage
    animation: EngineAnimation
    sim: EngineCycleSim

    @property
    def prism(self):
        return self.package.prism

    @property
    def mounting(self):
        return self.package.mounting

    @classmethod
    def build(cls, engine, install_context: str = "automotive", include_transmission: bool = False,
             transmission_mass_kg: float = 0.0, animation_divisions=24, covers_off: bool = True) -> "EngineCrate":
        package = EnginePackage.build(engine, install_context=install_context,
                                      include_transmission=include_transmission,
                                      transmission_mass_kg=transmission_mass_kg)
        graph = engine_mesh_view_graph(build_drivetrain_graph(engine))
        animation = build_animation(graph, divisions=animation_divisions, covers_off=covers_off)
        sim = EngineCycleSim(engine=engine)
        return cls(package=package, animation=animation, sim=sim)

    def step(self, dt: float) -> CrateStepResult:
        """Advances the real live sim by dt (a no-op while stalled --
        starting is the sim's own real starter-engagement action, not
        implied by stepping), looks up the already-baked animation
        frame for the resulting crank angle, and returns the real
        forces crossing the prism's own boundary this tick."""
        if not self.sim.stalled:
            self.sim.step(dt)
        angle = self.sim.state.crank_angle_deg
        frame_index = self.animation.frame_index(angle)
        structural = [m for m in self.package.mounting.mounts if m.role != "torque_strap"]
        loads = mount_loads(self.package.mounting.rigid_body, structural,
                            crank_reaction_torque_nm=self.sim.state.current_torque_nm)
        return CrateStepResult(crank_angle_deg=angle, animation_frame_index=frame_index, mount_loads=loads)
