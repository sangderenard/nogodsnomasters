"""THE WHOLE STATION, ONE GRAPH, RUNNING.

    python station_reference.py            the spec, headless
    python station_reference.py --live     the window: fire it, watch it

WHAT IS IN IT. Not a drawing of a mechanism -- the mechanism, on the
machine, in the engine:

    the BOXED DRUM      `drum_reference.build`, the real one: two
                        annuli, cell walls, crossed shear diagonals,
                        six internal motors on their cradle, thrust
                        races, traction contacts and the inner lock-up
    the TWO ARCHES      standing on the drum's own top plate. Each foot
                        is BOLTED into the plate nodes nearest it, so
                        the load goes into the drum and the drum has to
                        answer for it -- an arch foot pinned to the
                        world would be a boundary condition and the
                        deck would never feel the shot
    the DANGLING PLATFORM, the GUN PLATFORM, the CRADLE and the GUN
                        `sled_reference.build` emitting into this same
                        graph rather than onto a stand-in ring

and then it is solved by the pieces the game already uses:

    live_scene.LiveStructure    FrameSolver + modal state, stepped at
                                the engine's own stability margin
    firing_frame_view.Trial     the baked mesh and the phong pass
    calibres.recoil_of          what firing actually puts back in --
                                projectile impulse plus gas impulse,
                                and the gas is a third of it

SPACE FIRES IT. The force that arrives is the one `calibres` computes
for the bore selected, applied at the breech along the bore, decaying
over the two milliseconds it really takes. It is not a number chosen to
make the picture move.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field

import numpy as np

import drum_reference as dr
import sled_reference as sr
import stand


#: the drum's top plate ring -- what the arch feet bolt into
DECK_RING = "drum.upper.rim."
DECK_STATIONS = "drum.upper.station."

#: THE TWO PRIME MOVERS, one to a bay. Their masses are read off the
#: catalogue rather than typed here: `engines.get(...).mass_kg` is what
#: the engine actually weighs, and a placeholder beside it would be a
#: second opinion about the same engine.
BAY_ENGINES = {"port": "ldt465-multifuel-deuce",
               "starboard": "agt1500-abrams-turbine"}
#: how far above the bay frame an engine's centre of mass sits
ENGINE_RISE_M = 0.520

#: below this a mode is a mechanism freedom, not an elastic one
ZERO_HZ = 0.05
#: how many elastic modes to keep ON TOP of the mechanism's freedoms
ELASTIC_MODES = 60
#: how deep to look when counting the freedoms
MODE_PROBE = 320


def _placed_engine_mesh(mesh, identity: str, shift: np.ndarray):
    """Place one engine-owned baked mesh without changing its records."""
    from engine_mesh import EngineMesh
    return EngineMesh(
        np.asarray(mesh.vertices, float) + shift,
        np.asarray(mesh.normals, float).copy(),
        np.asarray(mesh.triangles, np.int64).copy(),
        np.asarray(mesh.material_ids, np.int32).copy(),
        [f"{identity}::{name}" for name in mesh.part_names],
        list(mesh.part_ranges), bool(mesh.moving),
        [None if group is None else f"{identity}::{group}"
         for group in (mesh.part_groups or [None] * len(mesh.part_names))])


@dataclass
class StationEngineRenderObject:
    """One live EngineCycleSim paired with its engine-owned baked object."""
    identity: str
    role: str
    sim: object
    static_mesh: object
    animation: object
    shift: np.ndarray
    enabled: bool
    power_path: tuple[str, ...] = ()
    utility: object | None = None
    coolant: object | None = None
    utility_reading: dict = field(default_factory=dict)
    solver_stats: dict = field(default_factory=dict)
    _frames: dict = field(default_factory=dict)

    def mesh(self):
        from engine_mesh import merge_engine_meshes
        index = self.animation.frame_index(self.sim.state.crank_angle_deg)
        moving = self._frames.get(index)
        if moving is None:
            moving = _placed_engine_mesh(
                self.animation.frames[index], self.identity, self.shift)
            self._frames[index] = moving
        return merge_engine_meshes([self.static_mesh, moving])

    def temperatures(self) -> dict[str, float]:
        from engine_gl_view import thermal_groups_from_state
        out = {f"{self.identity}::{name}": value
               for name, value in thermal_groups_from_state(
                   self.sim.state).items()}
        if self.utility is not None:
            prefix = f"powerplant.{self.identity.split('.')[-1]}."
            out.update({prefix + name: value for name, value in
                        self.utility.temperatures_k.items()})
        return out

    def step(self, dt: float) -> None:
        """Advance the engine, then its real shaft-fed utility package."""
        if self.enabled:
            self.sim.step(dt)
        shaft_rpm = float(self.sim.state.rpm) if self.enabled else 0.0
        if self.utility is not None:
            self.utility_reading = self.utility.step(
                dt, shaft_rpm, electrical_load_w=8_000.0,
                hydraulic_command=0.12, pneumatic_demand_l_min=40.0,
                refrigeration_command=0.20)


def station_engine_render_objects(document: dict) -> list[StationEngineRenderObject]:
    """Bake the actual bay engines and bind each to its own live cycle sim."""
    import engines
    from drivetrain_graph import build_drivetrain_graph
    from engine_cycle_sim import EngineCycleSim
    from engine_mesh import build_engine_mesh, start_animation
    from station_powerplant import PowerplantBasic, PowerplantRuntime
    from station_cooling import PlatformCoolantRuntime

    node = {n["identity"]: n for n in document["nodes"]}
    objects = []
    roles = {"port": "night/main power", "starboard": "heavy-combat backup"}
    for side in ("port", "starboard"):
        identity = f"engine.{side}"
        engine = engines.get(BAY_ENGINES[side])
        reservoirs = [n for n in document["nodes"]
                      if n.get("part_role") ==
                      "post-mounted-50-gallon-fuel-drum"
                      and f".{side}." in n["identity"]]
        if not reservoirs:
            raise ValueError(f"{identity} has no external station fuel drums")
        fuel_supply = {
            "reservoirs": tuple({
                "identity": n["identity"],
                "source_graph_identity": n["identity"],
                "capacity_l": n["fluid_volume_l"],
                "fuel": n["fluid"],
                "tare_mass_kg": n["tare_mass_kg"]} for n in reservoirs),
            "pumps": tuple({
                "identity": f"{n['identity']}.cap_pump",
                "reservoir_identity": n["identity"],
                "pump_kind": "electric",
                "flow_capacity_kg_s": 0.18,
            } for n in reservoirs),
            "inlet_identity": f"{identity}.external_fuel_in",
        }
        graph = build_drivetrain_graph(
            engine, external_fuel_supply=fuel_supply)
        static, moving0 = build_engine_mesh(graph, crank_angle_deg=0.0,
                                            covers_off=True)
        points = [m.vertices for m in (static, moving0) if len(m.vertices)]
        all_vertices = np.concatenate(points) if points else np.zeros((1, 3))
        source_centre = 0.5 * (all_vertices.min(axis=0)
                               + all_vertices.max(axis=0))
        target = np.asarray(node[identity]["reference_position"], float)
        shift = target - source_centre
        static = _placed_engine_mesh(static, identity, shift)
        animation = start_animation(graph, divisions=24, covers_off=True,
                                    detail=0.55)
        while animation.bake_next():
            pass
        sim = EngineCycleSim(engine=engine,
                             external_fuel_supply=fuel_supply)
        enabled = side == "port"
        if enabled:
            sim.start()
            sim.throttle = 0.18
        power_path = drivetrain_power_path(graph)
        target_node = node[identity]
        package = PowerplantBasic(
            f"powerplant.{side}", engine.identity,
            tuple(float(v) for v in target_node["reference_position"]),
            tuple(float(v) for v in target_node["body_half_extent_m"]))
        objects.append(StationEngineRenderObject(
            identity, roles[side], sim, static, animation, shift, enabled,
            power_path=power_path, utility=PowerplantRuntime(package),
            coolant=PlatformCoolantRuntime()))
        print(f"  baked {identity}: {engine.label}, {animation.n_frames} "
              f"moving frames -- {roles[side]}\n"
              f"    shaft path: {' -> '.join(power_path)}", flush=True)
    return objects


def drivetrain_power_path(graph: dict) -> tuple[str, ...]:
    """The existing rigid graph's continuous shaft path to its PTO end."""
    path = ("powertrain.engine", "powertrain.crank_shaft.rear",
            "powertrain.clutch", "powertrain.transmission",
            "powertrain.transfer_case")
    pairs = {(edge["a"], edge["b"]) for edge in graph["edges"]}
    missing = [(a, b) for a, b in zip(path, path[1:])
               if (a, b) not in pairs and (b, a) not in pairs]
    if missing:
        raise ValueError(f"incomplete powerplant shaft path: {missing}")
    return path

def gun_muzzle_mount(document: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Return the production gun body, its forward axis and muzzle offset."""
    nodes = {node["identity"]: node for node in document["nodes"]}
    if "turret.outer_barrel" in nodes:
        barrel = nodes["turret.outer_barrel"]
        axis = np.asarray(barrel.get("drum_axis", (0.0, 0.0, 1.0)), float)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        return ("turret.outer_barrel", axis,
                axis * float(barrel["drum_length_m"]) / 2.0)
    # Compatibility for old saved graphs; new builds use the production gun.
    return ("weapon.tube.11", np.asarray(sr.FORWARD, float),
            np.asarray(sr.FORWARD, float) * (sr.BARREL_M / 12.0 / 2.0))


def projectile_visual_document(document: dict) -> dict:
    """Add one reusable round and tracer span to the baked scene mesh."""
    barrel_id, axis, muzzle_offset = gun_muzzle_mount(document)
    centre = np.asarray(next(
        node["reference_position"] for node in document["nodes"]
        if node["identity"] == barrel_id), float)
    muzzle = centre + muzzle_offset
    tail = muzzle - axis * 0.001
    nodes = [*document["nodes"], {
        "identity": "viewer.projectile.tail",
        "kind": "chassis-load-node",
        "reference_position": tail.tolist(),
        "body_half_extent_m": [0.004, 0.004, 0.004],
        "wrench": {"force": [0.0, 0.0, 0.0],
                   "moment": [0.0, 0.0, 0.0]},
        "motion_group": "viewer-projectile",
        "assembly": "viewer-projectile",
        "in_view": True,
        "material": "brass",
        "mass_in_total": False,
        "mass_kg": 0.0,
        "part_role": "projectile-tracer-tail",
    }, {
        "identity": "viewer.projectile.round",
        "kind": "chassis-load-node",
        "reference_position": muzzle.tolist(),
        "body_half_extent_m": [0.065, 0.065, 0.18],
        "wrench": {"force": [0.0, 0.0, 0.0],
                   "moment": [0.0, 0.0, 0.0]},
        "motion_group": "viewer-projectile",
        "assembly": "viewer-projectile",
        "in_view": True,
        "material": "brass",
        "mass_in_total": False,
        "mass_kg": 0.0,
        "part_role": "projectile",
    }]
    edges = [*document["edges"], {
        "identity": "viewer.projectile.tracer",
        "a": "viewer.projectile.tail",
        "b": "viewer.projectile.round",
        "constraint": "rigid-distance",
        "assembly": "viewer-projectile",
        "constraint_token": 0,
        "freedoms_released": [],
        "release_ends": "both",
        "routed": False,
        "rest_length": 0.001,
        "radius": 0.018,
        "palette_role": "actuator-yellow",
        "material": "brass",
        "in_view": True,
        "load_path": "visible-flight-path-only",
    }]
    return dict(document, nodes=nodes, edges=edges)


def fire_viewer_projectile(projectiles, bore_mm: float, muzzle, direction,
                           *, muzzle_speed_m_s=None, shot_mass_kg=None,
                           bore_diameter_m=None):
    """Create the same physical round used by the SPACE-key path."""
    import calibres

    calibre = calibres.cannon(float(bore_mm))
    # Generated cannon calibres are deliberate runtime catalogue entries:
    # ProjectileField stores the stable name and resolves the actual mass,
    # diameter and muzzle velocity again while advancing the flight.
    calibres.CALIBRES[calibre.name] = calibre
    return projectiles.fire(
        muzzle=muzzle, direction=direction, calibre=calibre.name,
        dispersion_mrad=0.0,
        muzzle_speed_m_s=muzzle_speed_m_s, mass_kg=shot_mass_kg,
        diameter_m=bore_diameter_m,
    )


def push_barrel_until_structure_resists(
        structure, document: dict, axis, *, force_n: float = 400_000.0,
        dt: float = 1.0 / 60.0, timeout_s: float = 0.05,
        velocity_tolerance: float = 5.0e-4,
        acceleration_tolerance: float = 2.0e-2,
        require_global_stillness: bool = False) -> dict:
    """Push the unlocked barrel until the *whole graph* reaches equilibrium.

    No displacement is prescribed and no freedom is locked.  The force ramps
    onto the breech; the recoil pack, platforms, parallelograms, arches and
    every beam coordinate evolve together.  A station is allowed to ring: the
    default acceptance is finite, bounded evolution through the complete force
    path.  ``require_global_stillness`` retains the much slower laboratory
    settle criterion when that is the actual question being asked.
    """
    axis = np.asarray(axis, float)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    peak_joint_force = {edge["identity"]: 0.0
                        for edge in structure.joint_forces.edges}
    elapsed = 0.0
    settled_frames = 0
    diagnostics = {}
    ramp_duration_s = min(0.35, max(dt, timeout_s * 0.5))

    def attach_solver_extrema() -> None:
        """Record identities, not only norms, when a full solve is audited."""
        structure.inject_solver_stats()
        node_stats = [(node["identity"], node["solver_stats"])
                      for node in document["nodes"]]
        edge_stats = [(edge["identity"], edge.get("solver_stats", {}))
                      for edge in document["edges"]]
        diagnostics["fastest_nodes"] = [
            {"identity": identity,
             "speed_m_s": float(stats["speed_m_s"]),
             "velocity_m_s": list(stats["velocity_m_s"])}
            for identity, stats in sorted(
                node_stats, key=lambda item: item[1]["speed_m_s"],
                reverse=True)[:8]]
        diagnostics["highest_utilisation_edges"] = [
            {"identity": identity,
             "utilisation": float(stats["utilisation"]),
             "stress_pa": float(stats["stress_pa"])}
            for identity, stats in sorted(
                ((identity, stats) for identity, stats in edge_stats
                 if "utilisation" in stats),
                key=lambda item: item[1]["utilisation"],
                reverse=True)[:8]]
    while elapsed < timeout_s:
        ramp = min(1.0, elapsed / ramp_duration_s)
        structure.apply("turret.breech", tuple(-axis * force_n * ramp))
        structure.step(dt)
        elapsed += dt
        for identity, value in structure.joint_forces.last_element_force.items():
            peak_joint_force[identity] = max(
                peak_joint_force.get(identity, 0.0), abs(float(value)))

        velocity = structure.velocity.reshape(-1, 6)
        acceleration = structure.acceleration.reshape(-1, 6)
        max_linear_velocity = float(np.max(np.linalg.norm(
            velocity[:, :3], axis=1)))
        max_angular_velocity = float(np.max(np.linalg.norm(
            velocity[:, 3:], axis=1)))
        max_linear_acceleration = float(np.max(np.linalg.norm(
            acceleration[:, :3], axis=1)))
        max_angular_acceleration = float(np.max(np.linalg.norm(
            acceleration[:, 3:], axis=1)))
        residual_ratio = float(np.linalg.norm(
            structure.last_static_residual_free) / max(force_n, 1.0))
        diagnostics = {
            "elapsed_s": elapsed,
            "max_linear_velocity_m_s": max_linear_velocity,
            "max_angular_velocity_rad_s": max_angular_velocity,
            "max_linear_acceleration_m_s2": max_linear_acceleration,
            "max_angular_acceleration_rad_s2": max_angular_acceleration,
            "static_residual_ratio": residual_ratio,
        }
        globally_still = (
            ramp >= 1.0
            and max_linear_velocity <= velocity_tolerance
            and max_angular_velocity <= velocity_tolerance
            and max_linear_acceleration <= acceleration_tolerance
            and max_angular_acceleration <= acceleration_tolerance
            and residual_ratio <= acceleration_tolerance)
        settled_frames = settled_frames + 1 if globally_still else 0
        if settled_frames >= 6:
            attach_solver_extrema()
            diagnostics["peak_joint_force_n"] = peak_joint_force
            diagnostics["bump_stop_force_n"] = sum(abs(float(
                structure.joint_forces.last_element_force[edge["identity"]]))
                for edge in structure.joint_forces.bump_stop_edges)
            diagnostics["globally_still"] = True
            diagnostics["oscillation_permitted"] = not require_global_stillness
            return diagnostics
    attach_solver_extrema()
    diagnostics["peak_joint_force_n"] = peak_joint_force
    diagnostics["bump_stop_force_n"] = sum(abs(float(
        structure.joint_forces.last_element_force[edge["identity"]]))
        for edge in structure.joint_forces.bump_stop_edges)
    diagnostics["globally_still"] = False
    diagnostics["oscillation_permitted"] = not require_global_stillness
    finite = all(np.isfinite(float(diagnostics[name])) for name in (
        "max_linear_velocity_m_s", "max_angular_velocity_rad_s",
        "max_linear_acceleration_m_s2", "max_angular_acceleration_rad_s2",
        "static_residual_ratio"))
    if not finite:
        raise FloatingPointError(
            f"nonfinite complete-structure barrel push: {diagnostics}")
    if require_global_stillness:
        raise RuntimeError(
            "the complete unlocked structure did not settle under the barrel "
            f"push in {timeout_s}s: {diagnostics}")
    return diagnostics


def unlocked_motion_document(document: dict) -> dict:
    """A separate, truthful all-gun-motion-unlocked analysis document.

    Installation retainers (engine-pallet locks, tank straps, fasteners) stay
    installed.  The four arch crown height locks are removed from the frame;
    both parallelogram packing rams carry zero preload and their adjustable
    forward stops withdraw.  Recoil and pin releases are already encoded by
    their joint constraints and require no flag mutation.
    """
    import copy
    unlocked = copy.deepcopy(document)
    for edge in unlocked["edges"]:
        if edge.get("part_role") == "arch-crown-positive-height-lock":
            edge["lock_engaged"] = False
            edge["structural_participation"] = False
        if edge.get("part_role") == "platform-actuator":
            edge["preload_n"] = 0.0
            edge["spring_preload_n"] = 0.0
            edge["preload_force_n"] = 0.0
            edge["commanded_preload_n"] = 0.0
            edge["preload_command_frac"] = 0.0
        if edge.get("part_role") == "adjustable-forward-preload-stop":
            edge["preload_command_frac"] = 0.0
            edge["clearance_m"] = (float(edge.get("nominal_clearance_m", 0.0))
                                   - float(edge.get("release_adjustment_m", 0.0)))
    unlocked.pop("_graph_columns", None)
    return unlocked


def active_leveling_document(document: dict) -> dict:
    """Unlocked gun pose with support collars open for HCU levelling."""
    return support_leveling_document(unlocked_motion_document(document))


def support_leveling_document(document: dict) -> dict:
    """Keep the weapon pose as authored while opening support collars."""
    import copy
    active = copy.deepcopy(document)
    for edge in active["edges"]:
        if edge.get("part_role") == "outrigger-positive-length-lock":
            edge["lock_engaged"] = False
    active.pop("_graph_columns", None)
    return active


def apply_support_hydraulic_forces(structure, support_reading: dict) -> dict:
    """Bridge pressure force and pumped oil volume into the beam graph.

    Force alone is not a hydraulic cylinder: its rod may advance only as far
    as the oil delivered to its chamber permits.  The graph therefore has a
    paired compliant oil-column boundary for every support leg.  Move both
    faces to the existing LinearActuator position before applying its force.
    """
    forces = {
        f"stand.leg.{name}": float(state["force_n"])
        for name, state in support_reading["legs"].items()
    }
    extension_by_support = {
        name: float(state["extension_m"])
        for name, state in support_reading["legs"].items()
    }
    for edge in structure.joint_forces.bump_stop_edges:
        if edge.get("part_role") != "outrigger-hydraulic-volume-boundary":
            continue
        name = edge["support_name"]
        actuator_edge = structure.joint_forces.edge_by_id[
            f"stand.leg.{name}"]
        initial = float(actuator_edge.get(
            "initial_actuator_extension_m",
            actuator_edge.get("actuator_extension_m", 0.0)))
        travel = extension_by_support[name] - initial
        edge["clearance_m"] = travel + float(
            edge.get("hydraulic_position_offset_m", 0.0))
        actuator_edge["actuator_extension_m"] = extension_by_support[name]
    structure.set_edge_axial_forces(forces)
    return forces


def first_yield_under_unlocked_gravity(document: dict) -> dict:
    """Ramp assembled gravity from zero to 1 g and locate first yield.

    Two exact static solves provide the zero-g assembly-preload state and the
    1-g state.  Linear frame equilibrium makes every intermediate strain the
    affine interpolation of those endpoints; a dense scalar sweep therefore
    finds the first crossing without repeatedly factoring the same matrix.
    Normal and shear stress are combined by von Mises.
    """
    from frame_solver import FrameSolver
    from milspec import MATERIAL_BY_KEY

    unlocked = unlocked_motion_document(document)
    zero_solver = FrameSolver(unlocked, loads={}, gravity=True,
                              gravity_scale=0.0)
    zero = zero_solver.solve()
    full_solver = FrameSolver(unlocked, loads={}, gravity=True,
                              gravity_scale=1.0)
    full = full_solver.solve()
    if list(zero["identities"]) != list(full["identities"]):
        raise RuntimeError("unlocked gravity endpoint solves changed membership")

    member_by_identity = {edge["identity"]: edge
                          for edge in full_solver.members}
    result_edges = [member_by_identity[identity]
                    for identity in full["identities"]]
    e_mod, g_mod, yield_pa = [], [], []
    for edge in result_edges:
        E, G, _A, _Iy, _Iz, _J, _kappa, material = full_solver._section(edge)
        damage = edge.get("damage") or {}
        named = MATERIAL_BY_KEY.get(damage.get("material", ""), material)
        e_mod.append(E); g_mod.append(G)
        yield_pa.append(float(damage.get("yield_strength_pa",
                                         getattr(named, "yield_pa", 460e6))))
    e_mod = np.asarray(e_mod); g_mod = np.asarray(g_mod)
    yield_pa = np.asarray(yield_pa)
    a0, b0, s0 = (np.asarray(zero[name], float) for name in
                  ("axial_strain", "bending_strain", "shear_strain"))
    da = np.asarray(full["axial_strain"], float) - a0
    db = np.asarray(full["bending_strain"], float) - b0
    ds = np.asarray(full["shear_strain"], float) - s0

    def utilisation(scale: float) -> np.ndarray:
        normal = (np.abs(a0 + da * scale)
                  + np.abs(b0 + db * scale)) * e_mod
        shear = np.abs(s0 + ds * scale) * g_mod
        return np.sqrt(normal * normal + 3.0 * shear * shear) / np.maximum(
            yield_pa, 1.0)

    def first_crossing(mask: np.ndarray) -> tuple[float | None, np.ndarray]:
        first_scale = None
        first_util = utilisation(0.0)
        if float(first_util[mask].max(initial=0.0)) >= 1.0:
            return 0.0, first_util
        previous = 0.0
        for scale in np.linspace(0.001, 1.0, 1000):
            here = utilisation(float(scale))
            if float(here[mask].max(initial=0.0)) >= 1.0:
                lo, hi = previous, float(scale)
                for _ in range(36):
                    mid = (lo + hi) * 0.5
                    if float(utilisation(mid)[mask].max(initial=0.0)) >= 1.0:
                        hi = mid
                    else:
                        lo = mid
                first_scale = hi
                first_util = utilisation(hi)
                break
            previous = float(scale)
        return first_scale, first_util

    all_mask = np.ones(len(result_edges), dtype=bool)
    # The drop-in heat rejector stands on its own sand feet. It belongs in
    # the total site solve, but yielding in that independent machine cannot
    # explain motion of the arch-carried craft.
    craft_mask = np.asarray([
        edge.get("assembly") != "remote-heat-rejection"
        for edge in result_edges], dtype=bool)
    first_scale, first_util = first_crossing(all_mask)
    craft_scale, craft_util = first_crossing(craft_mask)
    full_util = utilisation(1.0)
    first_order = np.argsort(-first_util)[:8]
    craft_indices = np.flatnonzero(craft_mask)
    craft_order = craft_indices[np.argsort(-craft_util[craft_mask])[:8]]
    full_order = np.argsort(-full_util)[:8]
    identities = list(full["identities"])
    return {
        "document": unlocked,
        "first_yield_g": first_scale,
        "first_yields": [{"identity": identities[int(i)],
                           "utilisation": float(first_util[int(i)])}
                          for i in first_order],
        "craft_first_yield_g": craft_scale,
        "craft_first_yields": [{"identity": identities[int(i)],
                                  "utilisation": float(craft_util[int(i)])}
                                 for i in craft_order],
        "highest_at_1g": [{"identity": identities[int(i)],
                            "utilisation": float(full_util[int(i)])}
                           for i in full_order],
        "max_deflection_at_1g_m": float(full["max_deflection_m"]),
        "finite": bool(np.isfinite(full["displacement"]).all()
                       and np.isfinite(full_util).all()),
        "released_arch_locks": sum(
            edge.get("part_role") == "arch-crown-positive-height-lock"
            and not edge.get("structural_participation", True)
            for edge in unlocked["edges"]),
    }


def build(fold: float = 0.0, **drum_overrides):
    """The drum, everything standing on it, and the stand under it."""
    g, box, drum, races, spec = dr.build(**drum_overrides)
    deck_y = float(spec["height_m"])
    deck_nodes = [n["identity"] for n in g.nodes
                  if n["identity"].startswith((DECK_RING, DECK_STATIONS))]
    _g, plat, d, u = sr.build(fold=fold, g=g, deck_y=deck_y,
                              deck_nodes=deck_nodes)

    # ---- THE SITE UNDER IT: work area, two engine bays, four legs ----
    # The drum's own static ring was pinned to the world, which is a
    # machine hanging in space. It stands on the stand, the stand stands
    # on the legs, and the legs stand on the ground -- so the only thing
    # still fixed to the world is a pad with dirt under it.
    import engines as engine_catalogue
    heaviest = max(float(engine_catalogue.get(i).mass_kg)
                   for i in BAY_ENGINES.values())
    st = stand.Stand(deck_y=0.0, engine_mass_kg=heaviest)
    site = stand.emit_stand(g, st)
    seats = [n for n in g.nodes if n["identity"].startswith("seat.")]
    pos = {n["identity"]: np.asarray(n["reference_position"], float)
           for n in g.nodes}
    frame = list(site["corners"].values())
    g.motion_group, g.assembly = "frame", "stand"
    for n in seats:
        n.pop("fixed_to", None)
        here = pos[n["identity"]]
        near = min(frame, key=lambda k: float(np.linalg.norm(
            pos[k][[0, 2]] - here[[0, 2]])))
        g.edge(f"stand.seat_post.{n['identity'].split('.')[-1]}",
               n["identity"], near, "rigid-distance", radius=0.070,
               alloy="4130n", palette="chassis-grey", beam_solvable=True,
               part_role="turret-post",
               load_path="the-turret-standing-on-the-work-area-frame")

    # Four unmistakable columns continue from the static underside ring down
    # to the new twin-spine floor.  Short cap beams pick four quarter-ring
    # seats into column heads directly above the girder/crossmember
    # intersections.  The other seat posts remain as redundant upper-deck
    # paths; these columns make the lower floor a genuine load-sharing grillage
    # and ballast foundation, not a visual plate hung under the machine.
    from milspec import WeldedISection
    drum_column = WeldedISection(0.360, 0.300, 0.018, 0.025, "hy80",
                                 "fabricated HY-80 drum column")
    drum_cap = WeldedISection(0.280, 0.220, 0.016, 0.022, "hy80",
                              "fabricated HY-80 drum cap")

    def shaped(section, up):
        return dict(
            section_shape="welded-i", section_up=up,
            section_depth_m=section.depth_m,
            section_flange_width_m=section.flange_width_m,
            section_web_thickness_m=section.web_thickness_m,
            section_flange_thickness_m=section.flange_thickness_m,
            section_properties=section.graph_properties())

    seat_for_corner = {"fr": "seat.2", "fl": "seat.6",
                       "rl": "seat.10", "rr": "seat.14"}
    drum_columns = {}
    for tag, floor_node in site["floor_spine_column_nodes"].items():
        seat = seat_for_corner[tag]
        floor_at = pos[floor_node]
        seat_at = pos[seat]
        head = f"stand.drum_column.{tag}.head"
        g.node(head, (floor_at[0], seat_at[1], floor_at[2]),
               "chassis-load-node", material="hy80", mass_in_total=False,
               mass_kg=42.0, part_role="drum-column-cap-node",
               half_extent_m=(0.15, 0.08, 0.15))
        g.edge(f"stand.drum_column.{tag}", head, floor_node,
               "rigid-distance", radius=drum_column.flange_width_m / 2.0,
               alloy=drum_column.material, palette="chassis-grey",
               beam_solvable=True, part_role="drum-underside-floor-column",
               load_path="static-drum-ring-through-column-into-floor-spine",
               **shaped(drum_column, (0.0, 0.0, 1.0)))
        g.edge(f"stand.drum_cap.{tag}", seat, head, "rigid-distance",
               radius=drum_cap.flange_width_m / 2.0,
               alloy=drum_cap.material, palette="chassis-grey",
               beam_solvable=True, part_role="drum-column-cap-beam",
               load_path="quarter-ring-seat-into-drum-column-head",
               **shaped(drum_cap, (0.0, 1.0, 0.0)))
        drum_columns[tag] = {"seat": seat, "head": head,
                             "floor": floor_node}
    for end, left, right in (("front", "fl", "fr"),
                             ("rear", "rl", "rr")):
        g.edge(f"stand.drum_cap.cross.{end}",
               drum_columns[left]["head"], drum_columns[right]["head"],
               "rigid-distance", radius=drum_cap.flange_width_m / 2.0,
               alloy=drum_cap.material, palette="chassis-grey",
               beam_solvable=True, part_role="drum-column-head-crossmember",
               load_path="stout-cap-diaphragm-sharing-the-drum-column-load",
               **shaped(drum_cap, (0.0, 1.0, 0.0)))
    site["drum_columns"] = drum_columns
    # ---- THE PRIME MOVERS, one in each bay ----
    # Mounted on the bay frame, not on the ground and not on the turret:
    # they ride up with the site when the legs stand it up, so nothing
    # they feed ever has to cross a gap that moves.
    g.motion_group, g.assembly = "frame", "powerplant"
    powerplants = {}
    for side, identity in BAY_ENGINES.items():
        eng = engine_catalogue.get(identity)
        pts = site["bays"][side]
        centre = np.mean([pos[k] for k in pts.values()], axis=0)
        # A removable powerplant pallet, not four mounts floating in the bay.
        # Four chain winches lower the complete plate through the matching
        # lower-bay opening; positive locks carry it at either endpoint.
        from surfaces import Plate, emit_plate
        bay_xyz = np.asarray([pos[k] for k in pts.values()])
        plate_thickness_m = 0.012
        # The removable pallet plate sits inside the bay's square corner
        # blocks, not under them.  Its perimeter is the welded pallet; the
        # four winches and positive locks attach to its corners.
        pallet_half_x_m = 0.74
        pallet_half_z_m = 0.68
        plate_min_x = float(centre[0] - pallet_half_x_m)
        plate_max_x = float(centre[0] + pallet_half_x_m)
        plate_min_z = float(centre[2] - pallet_half_z_m)
        plate_max_z = float(centre[2] + pallet_half_z_m)
        plate = Plate(
            identity=f"powerplant.{side}.floor",
            corner=(plate_min_x,
                    deck_y + plate_thickness_m / 2.0,
                    plate_min_z),
            span_u=(plate_max_x - plate_min_x, 0.0, 0.0),
            span_v=(0.0, 0.0, plate_max_z - plate_min_z),
            thickness_m=plate_thickness_m, nu=2, nv=2,
            material="steel-plate", alloy="a36",
            attributes={"surface_role": "powerplant-lowering-floor",
                        "bay": side,
                        "pallet_half_extent_m": (pallet_half_x_m,
                                                  pallet_half_z_m),
                        "perimeter_fabrication": "continuous-fillet-weld"})
        plate_graph = emit_plate(g, plate, motion_group="frame",
                                 assembly="powerplant")
        plate_corner_keys = ((0, 0), (0, 2), (2, 0), (2, 2))
        plate_positions = {
            key: np.asarray(next(n["reference_position"] for n in g.nodes
                                if n["identity"] ==
                                plate_graph["nodes"][key]), float)
            for key in plate_corner_keys}
        pallet_targets = {}
        for tag, top in pts.items():
            key = min(plate_corner_keys,
                      key=lambda candidate: float(np.linalg.norm(
                          plate_positions[candidate][[0, 2]]
                          - pos[top][[0, 2]])))
            pallet = plate_graph["nodes"][key]
            pallet_targets[tag] = pallet
            initial_chain_m = abs(float(plate_positions[key][1]
                                        - pos[top][1]))
            g.edge(f"powerplant.{side}.winch.{tag}", top, pallet,
                   "chain-winch-hoist", radius=0.014, alloy="4340qt",
                   palette="chassis-grey", beam_solvable=True,
                   part_role="four-corner-powerplant-chain-winch", bay=side,
                   commanded_rest_length_m=initial_chain_m,
                   minimum_rest_length_m=initial_chain_m,
                   maximum_rest_length_m=(initial_chain_m
                                           + st.lower_room_clear_height_m),
                   stiffness_n_per_m=1.8e8,
                   linear_damping_n_s_per_m=45_000.0,
                   tension_only=True, lowering_speed_m_s=0.035,
                   load_share=0.25,
                   load_path="chain-winch-lowering-the-complete-engine-pallet")
            g.edge(f"powerplant.{side}.lock.upper.{tag}", top, pallet,
                   "direct-drive-lockup", radius=0.026, alloy="4340qt",
                   palette="chassis-grey", beam_solvable=True,
                   part_role="powerplant-pallet-positive-lock", bay=side,
                   lock_position="upper", lock_engaged=True,
                   load_path="upper-positive-lock-carries-the-installed-pallet")
            lower = site["lower_bays"][side][tag]
            g.edge(f"powerplant.{side}.lock.lower.{tag}", lower, pallet,
                   "direct-drive-lockup", radius=0.026, alloy="4340qt",
                   palette="chassis-grey", beam_solvable=False,
                   structural_participation=False,
                   part_role="powerplant-pallet-positive-lock", bay=side,
                   lock_position="lower", lock_engaged=False,
                   load_path="lower-positive-lock-carries-the-servicing-pallet")
        g.motion_group, g.assembly = "frame", "powerplant"
        seat = centre + np.array([0.0, ENGINE_RISE_M, 0.0])
        node = f"engine.{side}"
        g.node(node, tuple(float(v) for v in seat),
               "load-bearing-structure", material="4130n",
               mass_in_total=False, mass_kg=float(eng.mass_kg),
               part_role="managed-engine-object", engine_identity=identity,
               engine_label=getattr(eng, "label", identity),
               displacement_l=float(getattr(eng, "displacement_l", 0.0)),
               bay=side, half_extent_m=(0.55, 0.45, 0.40), in_view=False,
               render_primitive=False,
               object_model="engine-cycle-managed-machine",
               solver_boundary="condensed-engine-mass-inertia-and-state",
               rendered_by="engine-owned-baked-mesh",
               structural_mount_targets=tuple(pallet_targets.values()))
        fuel_port = f"{node}.external_fuel_in"
        fuel_kind = ("ultra-low-sulfur-diesel" if side == "port"
                     else "jet-a-kerosene")
        g.node(fuel_port, tuple(float(v) for v in
                               (seat + np.array((0.0, 0.0, -0.41)))),
               "engine-block-port", material="hardened-steel",
               mass_kg=0.8, mass_in_total=True,
               part_role="engine-external-fuel-intake-port",
               port_kind="external-fuel-inlet", accepted_fuel=fuel_kind,
               coupling="dry-break-JIC-service-coupling",
               solver_condensed_into=node, solver_condensed_mass=True,
               surface_port=True, render_primitive=False, in_view=False,
               engine_identity=identity)
        # FOUR MOUNTS, ONE PER BAY CORNER. An engine on fewer is an
        # engine that torques its own bay: the reaction to whatever it
        # drives comes out of the block, and it has to land somewhere.
        for tag, corner in pallet_targets.items():
            g.edge(f"engine.{side}.mount.{tag}", node, corner,
                   "engine-mount-isolator", radius=0.052, alloy="4130n",
                   palette="chassis-grey", beam_solvable=True,
                   part_role="engine-mount", bay=side,
                   port_role="structural-mount",
                   load_path="the-engine-carried-by-its-own-bay-frame")
        from station_powerplant import PowerplantBasic, emit_powerplant_basic
        package = PowerplantBasic(
            identity=f"powerplant.{side}", engine_identity=identity,
            centre=tuple(float(v) for v in seat),
            # This is the reusable installation envelope, not a claim
            # about the current engine's casting dimensions.
            engine_half_extent_m=(0.62, 0.48, 0.58))
        installed_extent = tuple(float(v) for v in next(
            n for n in g.nodes if n["identity"] == node)["body_half_extent_m"])
        if not package.accepts(installed_extent):
            raise ValueError(f"{identity} does not fit {side} powerplant envelope: "
                             f"{installed_extent} > {package.engine_half_extent_m}")
        powerplants[side] = emit_powerplant_basic(g, package, pallet_targets)

    # Shared stores are deliberately central ballast, in opposed pairs
    # around the fixed access trunk.  They mount into the beam graph;
    # their hoses are routed edges and cannot counterfeit stiffness.
    from station_powerplant import (emit_post_fuel_barrels,
                                    emit_shared_storage,
                                    route_shared_storage)
    trunk_mounts = [n["identity"] for n in g.nodes
                    if n["identity"].startswith("trunk.shell.2.")]
    trunk_mounts += list(site["corners"].values())
    storage = emit_shared_storage(g, trunk_mounts)
    fuel = emit_post_fuel_barrels(g, site)
    # Install after the two powerplant pumps exist so its large pressure and
    # return headers terminate on the actual machine objects.
    from hcu import emit_hcu
    site["hcu"] = emit_hcu(g, st, site)
    from station_cooling import (emit_desert_heat_rejector,
                                 emit_metaconduit, emit_site_cooling)
    cooling = emit_site_cooling(g, st, site)
    metaconduits = {
        side: emit_metaconduit(g, side, powerplants[side], cooling)
        for side in ("port", "starboard")}
    route_shared_storage(g, powerplants, storage,
                         metaconduits=metaconduits)
    site["heat_rejector"] = emit_desert_heat_rejector(
        g, st, cooling, metaconduits)
    site["powerplants"] = powerplants
    site["storage"] = storage
    site["fuel"] = fuel
    site["cooling"] = cooling
    site["metaconduits"] = metaconduits
    from fluid_routing import annotate_fluid_routes
    annotate_fluid_routes({"nodes": g.nodes, "edges": g.edges})
    return g, plat, box, spec, st, site


def spec_sheet(g, plat, box, spec, st=None) -> list:
    doc = g.as_document()
    from sled import describe
    import calibres
    recoiling = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
                    if x["identity"].startswith(("weapon.", "mount.", "gun.")))
    design_j = (calibres.recoil_of(calibres.cannon(120.0)).net_impulse_n_s
                ** 2) / (2.0 * recoiling)
    rot = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
              if not x["identity"].startswith("seat."))
    out = [f"THE STATION -- {len(doc['nodes'])} nodes,"
           f" {len(doc['edges'])} edges",
           f"  drum            {box.inner_radius_m:.3f} .."
           f" {box.outer_radius_m:.3f} m, {spec['height_m'] * 1000:.0f} mm"
           f" deep, deck at {spec['height_m']:.3f} m",
           f"  arch feet       bolted into {sr.PAD_BOLTS} plate nodes each,"
           f" 4 feet -- the load goes into the drum",
           f"  turning mass    {rot:.0f} kg",
           ""]
    out += describe(plat, design_j)
    return out


def paint(document: dict) -> dict:
    """Both palettes, one document: the drum's and the station's."""
    a = dr._paint(document)
    by = {n["identity"]: n for n in sr._paint(document)["nodes"]}
    be = {e["identity"]: e for e in sr._paint(document)["edges"]}
    keep = ("arch.", "dangling.", "gun.", "weapon.", "mount.", "station.")
    nodes = [by[n["identity"]] if n["identity"].startswith(keep) else n
             for n in a["nodes"]]
    edges = [be[e["identity"]] if e["identity"].startswith(keep) else e
             for e in a["edges"]]
    return dict(a, nodes=nodes, edges=edges)



# =====================================================================
#  FIRING IT THROUGH THE WHOLE CHAIN
# =====================================================================
def fire_through(g, plat, bores=(20.0, 120.0), gravity: bool = True) -> list:
    """A shot, with the force handed correctly between the sims.

    EACH SIM DOES ITS OWN JOB AND HANDS ON WHAT IT MADE. The mistake
    worth naming is putting the raw breech impulse straight onto the
    structure: that is fifteen meganewtons, it is what the mount would
    see if there were no recoil system at all, and a structure asked to
    carry it reports every member past yield. The recoil system exists
    precisely so the structure never sees it.

        calibres.recoil_of          the impulse -- projectile plus gas,
                                    and the gas is a third of it
        PlatformStage / the slide   what the absorber TRANSMITS out of
                                    that impulse. `AdaptiveRecoilDamper`
                                    inverts the energy balance for this
                                    round and reports the peak force it
                                    makes doing it
        graph_physics.solve_under_load
                                    that force as a real load: the
                                    frame solver turns it into
                                    displacements through the whole
                                    graph, and the game's own J2 member
                                    law turns the strains into stress,
                                    plastic flow and fracture demand

    The load is applied AT THE RAIL, because that is where the absorber
    pushes. Applying it at the breech would be loading the gun, which
    is on a released freedom and simply recoils.
    """
    import calibres
    import graph_physics as gp

    doc = g.as_document()
    recoiling = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
                    if x["identity"].startswith(("weapon.", "mount.",
                                                 "gun.")))
    out = [f"FIRING THROUGH THE CHAIN -- {recoiling:.0f} kg recoiling",
           ""]
    for bore in bores:
        cal = calibres.cannon(bore)
        w = calibres.recoil_of(cal)
        imp = w.net_impulse_n_s
        sized = plat.slide_damper().orifice_for(imp, recoiling)
        transmitted = float(sized.get("peak_force_n", 0.0))
        out += [f"  {cal.name}   {cal.muzzle_energy_j / 1e6:.2f} MJ"
                f" at the muzzle"]
        out += ["  " + ln for ln in w.describe()]
        out += [f"    recoil velocity {imp / recoiling:7.2f} m/s"
                f"  ->  {0.5 * imp * imp / recoiling / 1000:8.1f} kJ"]
        if transmitted <= 0.0:
            out += [f"    the absorber's spring alone stops it --"
                    f" {sized.get('note', '')}", ""]
            continue
        out += [f"    absorber        aperture"
                f" {sized['orifice_mm']:5.1f} mm, transmits"
                f" {transmitted / 1000:7.0f} kN"
                f"   (raw: {w.peak_force_n() / 1000:.0f} kN)"]
        r = gp.solve_under_load(doc,
                                loads={RAIL_NODE: (0.0, 0.0, -transmitted)},
                                gravity=gravity)
        demand = np.abs(r["fracture_demand"])
        order = np.argsort(-demand)[:6]
        failed = int((r["failed"] > 0.5).sum())
        out += [f"    deflection      {r['max_deflection_m'] * 1000:7.2f} mm"
                f"   {failed} member(s) failed",
                "    worst members, by how close to fracture:"]
        for i in order:
            out.append(
                f"      demand {demand[i]:5.3f}   "
                f"axial {r['axial_stress_pa'][i] / 1e6:8.1f}   "
                f"bend {r['bending_stress_pa'][i] / 1e6:8.1f}   "
                f"shear {r['shear_stress_pa'][i] / 1e6:7.1f} MPa   "
                f"{r['identities'][i]}")
        out.append("")
    return out


# =====================================================================
#  THE WINDOW. The engine's own solver, stepped against the clock.
# =====================================================================
def live(g, plat, bore_mm: float = 20.0, W: int = 1440, H: int = 920,
         push_to_equilibrium: bool = False, stand_spec=None,
         urgent: bool = True, adaptive: bool = True):
    """The station, solved every tick, in a window.

    Every piece here is the game's own: `LiveStructure` is the complete
    physical-coordinate reference solve that `live_scene` runs, `Trial` is the
    baked mesh and the phong pass that `firing_frame_view` runs, and
    the force is whatever `calibres.recoil_of` says a shot of that
    bore puts back into the mount. Nothing in the loop is a number
    chosen to make the picture move."""
    import pygame
    import calibres
    import firing_frame_trial as fft
    import firing_frame_view as ffv
    from projectiles import ProjectileField
    from live_scene import LiveStructure
    from articulation import _flat

    doc = (support_leveling_document(g.as_document())
           if urgent else g.as_document())
    print("  assembling the structure ...", flush=True)
    t0 = time.perf_counter()
    # ---- FULL REFERENCE SOLVE ---------------------------------------
    # This is the authoritative fine-scale system: every free beam DOF,
    # including every zero-frequency mechanism freedom, participates in the
    # same solve.  The complete spectral basis is only a factorisation of the
    # assembled Newmark matrix.  It discards nothing.  Coarser game-scale
    # matrices are to be projected/baked from this result and checked against
    # it, never substituted here as the definition of the machine.
    from frame_solver import FrameSolver
    reference_solver = FrameSolver(document=doc, loads={})
    probe = reference_solver.modes(count=MODE_PROBE)
    hz = np.asarray(probe["all_frequencies_hz"], float)
    free = int(probe["mechanism_count"])
    print(f"  {free} mechanism freedoms at ~0 Hz -- the folding and the"
          f" slide. Full reference solve: all {len(hz)} elastic modes,"
          f" starting at {hz[0]:.2f} Hz", flush=True)
    # The released supports are restrained by nonlinear oil-column laws,
    # which are intentionally absent from the linear K matrix.  A static
    # pseudo-inverse of that singular K is not a settled machine: it chooses
    # an arbitrary least-squares point along the free support coordinates.
    # Begin from the authored, post-assembled geometry with zero beam strain,
    # then let gravity, the hydraulic volume boundaries, dampers and every
    # beam settle together in this same runtime solve.
    st = LiveStructure(doc, modes=ELASTIC_MODES, solver=reference_solver,
                       reference=probe, settle_from_authored=urgent)
    n60, h60 = st.substep_plan(1.0 / 60.0)
    print(f"  {len(st.members)} members, {len(st.omega)} beam modes,"
          f" up to {st.omega[-1] / (2 * math.pi):.1f} Hz"
          f"   ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"  baked complete-matrix inverse (no reduction) -> {n60} substeps of "
          f"{h60 * 1000:.3f} ms in a 60 Hz frame",
          flush=True)
    if adaptive:
        from component_mode_atlas import (
            AdaptiveGestaltPolicy, build_component_orchestration_graph,
            build_component_stress_envelope)
        from milspec import MATERIAL_BY_KEY
        orchestration = build_component_orchestration_graph(reference_solver)
        # Explicit accuracy budget for the live/game lane. The full reference
        # matrices above remain untouched and --full-beam-live disables this.
        sleep_utilisation = 2.0e-4
        wake_utilisation = 1.0e-3
        internal_speed_m_s = 2.0e-4
        dwell_s = 0.05
        maximum_component_dofs = 144
        maximum_components = 4
        accepted = []
        configurations = {}
        edge_by_id = {edge["identity"]: edge for edge in doc["edges"]}
        candidates = sorted(
            orchestration.partitions,
            key=lambda part: (-6 * (len(part.node_indices) - 1),
                              part.identity))
        for partition in candidates:
            if len(configurations) >= maximum_components:
                break
            if len(partition.node_indices) * 6 > maximum_component_dofs:
                continue
            if any(neighbor in accepted for neighbor in
                   orchestration.weld_neighbors(partition.identity)):
                continue
            try:
                envelope = build_component_stress_envelope(
                    reference_solver, partition)
            except ValueError:
                continue
            yields = []
            for identity in partition.member_identities:
                edge = edge_by_id[identity]
                damage = edge.get("damage") or {}
                material = MATERIAL_BY_KEY.get(
                    damage.get("material", "4130n"), MATERIAL_BY_KEY["4130n"])
                yields.append(float(getattr(material, "yield_pa", 460e6)))
            yield_pa = min(yields) if yields else 460e6
            assembled = reference_solver.assemble_component_matrices(
                partition.member_identities,
                node_mass_fractions=partition.node_mass_fractions)
            mass_kg = float(np.sum(np.mean(
                np.diag(assembled["mass_matrix"]).reshape(-1, 6)[:, :3],
                axis=1)))
            policy = AdaptiveGestaltPolicy(
                sleep_stress_bound_pa=yield_pa * sleep_utilisation,
                wake_stress_bound_pa=yield_pa * wake_utilisation,
                sleep_internal_kinetic_bound_j=(
                    0.5 * mass_kg * internal_speed_m_s ** 2),
                sleep_dwell_s=dwell_s)
            configurations[partition.identity] = (policy, envelope)
            accepted.append(partition.identity)
        if configurations:
            print(f"  baking adaptive graph projection for "
                  f"{len(configurations)} components: "
                  f"{', '.join(configurations)} ...", flush=True)
            st.configure_adaptive_orchestration(orchestration, configurations)
            subspace, _diagonal = st.prime_gestalt_projection(configurations)
            print(f"  adaptive live lane: {len(st.free)} -> "
                  f"{subspace.reduced_size} coordinates after quiet dwell; "
                  f"sleep {sleep_utilisation:.1e}, wake "
                  f"{wake_utilisation:.1e} of yield, internal speed "
                  f"{internal_speed_m_s * 1000.0:.2f} mm/s",
                  flush=True)

    # ---- WHAT FIRING PUTS IN: THE EXISTING COMBUSTION LAW -----------
    # The pressure history is built before SDL starts.  SPACE consumes the
    # resulting short force segments through the same complete structure
    # step as every other load.  No one-frame impulse surrogate remains.
    from gun_rig import GunRig
    gun_rig = GunRig(bore_mm=float(bore_mm), graph=g, document=doc)
    shot_profile = gun_rig.recoil_force_profile()
    print(f"  {bore_mm:5.0f} mm   interior ballistics "
          f"{shot_profile['duration_s'] * 1000.0:.3f} ms, "
          f"{shot_profile['impulse_n_s']:.1f} N.s, "
          f"peak {shot_profile['peak_force_n'] / 1000.0:.1f} kN",
          flush=True)

    engines_live = station_engine_render_objects(doc)
    from hcu import (HCUBackupHydraulics, HydraulicControlUnit,
                     HydraulicSwitchboard, apply_engine_authority,
                     structure_sensor_frame)
    hcu_runtime = HydraulicControlUnit()
    if urgent:
        hcu_runtime.urgent()
    ignition_low_live = True
    ignition_high_live = bool(urgent)
    if stand_spec is None:
        stand_spec = stand.Stand()
    carried_kg = sum(float(node.get("mass_kg", 0.0))
                     for node in doc["nodes"] if node.get("mass_in_total", True))
    support_hydraulics = stand.outrigger_set(stand_spec, carried_kg)
    backup_hydraulics = HCUBackupHydraulics()
    hydraulic_switchboard = HydraulicSwitchboard()
    from station_cooling import DesertHeatRejectorRuntime, SiteCoolingRuntime
    site_cooling = SiteCoolingRuntime()
    heat_rejector = DesertHeatRejectorRuntime()
    last_cooling = None
    last_rejector = None
    last_support = None
    render_doc = projectile_visual_document(doc)
    # Each engine node is the condensed managed machine boundary and declares
    # render_primitive=False.  Only its engine-owned baked mesh is displayed;
    # there is no second box standing in for it.
    render_doc = dict(
        render_doc,
        mesh_overlays=[{"identity": o.identity, "mesh": o.mesh()}
                       for o in engines_live])
    trial = ffv.Trial(render_doc, width=W, height=H, hidden=False)
    ranges = [trial.part_range.get("edge_" + _flat(i)) for i in st.identities]
    owner = np.concatenate([np.full(r[1] - r[0], i, np.int32)
                            for i, r in enumerate(ranges) if r])
    tris = np.concatenate([np.arange(r[0], r[1]) for r in ranges if r])
    bands = np.asarray([lim for lim, _c, _l in fft.UTILISATION_BANDS])
    band_mat = np.asarray([trial.band_ids[c]
                           for _l, c, _n in fft.UTILISATION_BANDS])
    base_mat = trial.rest_material_ids

    LOAD_NODE = "turret.breech"
    if LOAD_NODE not in st.solver.index:
        raise KeyError(f"{LOAD_NODE} is not in the solve -- nothing to fire")

    clock = pygame.time.Clock()
    az, el, zoom = 0.62, 0.34, 1.0
    drag, running, frames = False, True, 0
    bore, fired, peak = bore_mm, 0, 0.0
    pulse_segments = ()
    pulse_index = 0
    pulse_remaining_s = 0.0
    proof_load = False
    colour_mode = "assembly"
    projectiles = ProjectileField()
    # Imports and generates the wide trajectory carrier before the SDL
    # window exists.  SPACE must never be the operation that first pays
    # runtime setup and prevents the event loop from presenting a frame.
    projectiles.warm()
    barrel_node, axis, muzzle_offset = gun_muzzle_mount(doc)
    if push_to_equilibrium:
        stopped = push_barrel_until_structure_resists(st, doc, axis)
        active = sum(force > 0.0 for force in
                     stopped["peak_joint_force_n"].values())
        state = "settled" if stopped["globally_still"] else "ringing"
        print("  complete rig barrel force-path check: "
              f"{stopped['elapsed_s']:.3f} s, {state}, "
              f"{active} joint laws loaded, "
              f"residual {stopped['static_residual_ratio']:.3e}, "
              f"rear stops {stopped['bump_stop_force_n'] / 1000.0:.1f} kN",
              flush=True)
    solved = st.positions()
    elastic = {
        identity: solved[index]
        for identity, index in st.solver.index.items()
    }
    visible_positions = elastic
    print("  SPACE fire   1 yield colour   2 assembly colour   "
          "P pack swings forward   O release swings   "
          "T turbine combat-backup   hold G proof-load slide   R reset   "
          "drag orbit   wheel zoom   ESC quit", flush=True)
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    pulse_segments = shot_profile["segments"]
                    pulse_index = 0
                    pulse_remaining_s = pulse_segments[0][0]
                    fired += 1
                    shot = fire_viewer_projectile(
                        projectiles, bore,
                        visible_positions[barrel_node] + muzzle_offset,
                        axis,
                        muzzle_speed_m_s=shot_profile["muzzle_m_s"],
                        shot_mass_kg=shot_profile["shot_mass_kg"],
                        bore_diameter_m=shot_profile["bore_diameter_m"],
                    )
                    print(f"  {shot.identity} fired: {bore:.0f} mm from "
                          f"({shot.position[0]:.2f}, {shot.position[1]:.2f}, "
                          f"{shot.position[2]:.2f}) m at "
                          f"{shot.speed_m_s:.0f} m/s", flush=True)
                elif ev.key == pygame.K_g:
                    # User-held proof load through the same beam/joint path
                    # as firing. Releasing G removes the external force.
                    proof_load = True
                elif ev.key == pygame.K_p:
                    st.joint_forces.command_platform_preload(1.0)
                    print("  swing packing rams FULL PRELOAD", flush=True)
                elif ev.key == pygame.K_o:
                    st.joint_forces.command_platform_preload(0.0)
                    print("  swing packing rams RELEASED TO HANG", flush=True)
                elif ev.key == pygame.K_r:
                    st.reset()
                    pulse_segments = ()
                    pulse_index = 0
                    pulse_remaining_s = 0.0
                    peak = 0.0
                elif ev.key == pygame.K_1:
                    colour_mode = "yield"
                    print("  colour: YIELD UTILISATION", flush=True)
                elif ev.key == pygame.K_2:
                    colour_mode = "assembly"
                    print("  colour: ASSEMBLY / MATERIAL", flush=True)
                elif ev.key == pygame.K_t:
                    ignition_high_live = not ignition_high_live
                    print("  high-platform ignition rail "
                          + ("LIVE -- HCU start authority"
                             if ignition_high_live else
                             "OPEN -- running engine is not killed"),
                          flush=True)
            elif ev.type == pygame.KEYUP and ev.key == pygame.K_g:
                proof_load = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    drag = True
                elif ev.button in (4, 5):
                    zoom = max(0.25, min(4.0, zoom
                                         * (0.9 if ev.button == 4 else 1.1)))
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                drag = False
            elif ev.type == pygame.MOUSEMOTION and drag:
                az -= ev.rel[0] * 0.008
                el = max(-0.9, min(1.4, el + ev.rel[1] * 0.005))

        # ---- FORCE HISTORY, NOT A DISPLAY-FRAME IMPULSE -------------
        # Each segment is the impulse-preserving mean of the compiled
        # interior-ballistics breech pressure over 50 microseconds.  The
        # structure advances over that same segment, so the recoil slide,
        # joints, rigid members and flexible beams all see the pressure rise
        # and fall over the actual in-bore duration.
        # Each bay object is driven by its own EngineCycleSim.  The multifuel
        # is normal/night power; the turbine advances only when combat backup
        # has been brought online.
        by_platform = {
            "low": next(o for o in engines_live if o.identity == "engine.port"),
            "high": next(o for o in engines_live if o.identity == "engine.starboard"),
        }
        hcu_sensors = structure_sensor_frame(
            st,
            platform_engine_running={
                platform: (not obj.sim.starter.engaged
                           and obj.sim.state.rpm >= obj.sim.engine.idle_rpm * .6)
                for platform, obj in by_platform.items()},
            platform_starter_engaged={
                platform: obj.sim.starter.engaged
                for platform, obj in by_platform.items()},
            ignition_low=ignition_low_live,
            ignition_high=ignition_high_live)
        hcu_command = hcu_runtime.step(dt, hcu_sensors)
        for platform, engine_object in by_platform.items():
            start_request = hcu_command.engine_start_request[platform]
            kill_request = hcu_command.engine_kill_request[platform]
            apply_engine_authority(engine_object.sim,
                                   start=start_request, kill=kill_request)
            if start_request:
                engine_object.enabled = True
            if kill_request:
                engine_object.enabled = False
        for engine_object in engines_live:
            engine_object.step(dt)

        # Three fluids, two real exchangers: sealed refrigerant removes heat
        # from the potable store; the HCU meters a separate glycol loop
        # through a double-wall water/coolant exchanger.  Each platform's
        # supplied pump becomes primary only when its engine has none, while
        # remaining available as controlled parallel assist.
        platform_cooling = []
        for engine_object in engines_live:
            platform_cooling.append(engine_object.coolant.step(
                dt, engine_object.sim, site_cooling.coolant_temp_k,
                hcu_command=1.0))
        refrigeration_w = sum(float(o.utility_reading.get(
            "refrigerant_cooling_w", 0.0)) for o in engines_live)
        last_cooling = site_cooling.step(
            dt, refrigerant_cooling_w=refrigeration_w,
            platform_heat_w=sum(float(x["station_heat_return_w"])
                                for x in platform_cooling),
            hcu_valve=1.0)
        running_engines = [o for o in engines_live if o.enabled]
        exhaust_mass = sum(max(0.0, float(getattr(
            o.sim, "_exhaust_demand_kg_s", 0.0))) for o in running_engines)
        exhaust_temp = (sum(float(o.sim.state.exhaust_tailpipe_temp_k)
                            * max(0.0, float(getattr(o.sim, "_exhaust_demand_kg_s", 0.0)))
                            for o in running_engines) / max(exhaust_mass, 1e-9)
                        if running_engines else site_cooling.ambient_k)
        last_rejector = heat_rejector.step(
            dt, ambient_k=site_cooling.ambient_k,
            coolant_in_k=site_cooling.coolant_temp_k,
            coolant_heat_capacity_j_per_k=site_cooling.coolant_l * 1.04 * 3500.0,
            coolant_flow_command=1.0, exhaust_temp_k=exhaust_temp,
            exhaust_mass_flow_kg_s=exhaust_mass, blower_command=0.65)
        site_cooling.coolant_temp_k = last_rejector["coolant_out_k"]

        # Full-oil urgent dispatch.  Both engine pumps, the independent
        # electric reserve and the HCU switchboard participate in the same
        # measured source/draw exchange.  The fluid is intentionally common
        # in this emergency mode and the manifold declares a flush service.
        if urgent:
            leg_commands = {
                name: (hcu_command.pillar_valves[name]
                       if name.startswith("inner.") else
                       hcu_command.corner_telescope_valves[name])
                for name in support_hydraulics.legs}
            support_loads = {
                name: carried_kg * 9.80665 / len(support_hydraulics.legs)
                for name in support_hydraulics.legs}

            def urgent_supply(demand_l_min, required_pressure_pa):
                sources = {}
                readings = {}
                for platform, obj in by_platform.items():
                    reading = obj.utility.step(
                        0.0, float(obj.sim.state.rpm),
                        electrical_load_w=8_000.0,
                        hydraulic_demand_l_min=demand_l_min,
                        hydraulic_required_pressure_pa=required_pressure_pa,
                        refrigeration_command=0.20)
                    readings[platform] = reading
                    sources[platform] = {
                        "available_flow_l_min": reading["hydraulic_flow_l_min"],
                        "pressure_pa": reading["hydraulic_pressure_pa"]}
                reserve = backup_hydraulics.step(
                    dt, command=hcu_command.backup_pump_command,
                    demand_l_min=demand_l_min,
                    required_pressure_pa=required_pressure_pa)
                sources["backup"] = {
                    "available_flow_l_min": reserve["delivered_l_min"],
                    "pressure_pa": reserve["pressure_pa"]}
                dispatched = hydraulic_switchboard.dispatch(
                    sources, {"support": {
                        "requested_flow_l_min": demand_l_min,
                        "required_pressure_pa": required_pressure_pa,
                        "priority": 100}})
                contributions = dispatched["contributions_l_min"]["support"]
                delivered = sum(contributions.values())
                for platform, obj in by_platform.items():
                    obj.sim.known_accessory_shaft_load_w = float(
                        readings[platform]["total_shaft_load_w"])
                return {"pressure_pa": min(21_000_000.0,
                                            max(101_325.0, required_pressure_pa)),
                        "delivered_l_min": delivered,
                        "shaft_load_w": sum(float(r["total_shaft_load_w"])
                                            for r in readings.values())
                                        + reserve["electrical_w"],
                        "source_contributions_l_min": contributions,
                        "common_oil_emergency_mode": True}

            last_support = support_hydraulics.step(
                dt, leg_commands, supply=urgent_supply,
                loads_n=support_loads)
            apply_support_hydraulic_forces(st, last_support)

        frame_remaining_s = dt
        while frame_remaining_s > max(1.0e-15, dt * 1.0e-12):
            segment_s = frame_remaining_s
            shot_force_n = 0.0
            if pulse_index < len(pulse_segments):
                segment_s = min(segment_s, pulse_remaining_s)
                shot_force_n = float(pulse_segments[pulse_index][1])
            total_force_n = shot_force_n + (15_000.0 if proof_load else 0.0)
            if total_force_n > 0.0:
                st.apply(LOAD_NODE, tuple(-total_force_n * v for v in axis))
            else:
                st.clear()
            st.step(segment_s)
            frame_remaining_s = max(0.0, frame_remaining_s - segment_s)
            if pulse_index < len(pulse_segments):
                pulse_remaining_s -= segment_s
                if pulse_remaining_s <= 1.0e-15:
                    pulse_index += 1
                    if pulse_index < len(pulse_segments):
                        pulse_remaining_s = pulse_segments[pulse_index][0]
        if not proof_load:
            st.clear()
        projectiles.step(dt, {})

        # Put the authoritative beam/joint result on the graph objects before
        # rendering it.  Visual strain colour, HCU feedback and an analysis
        # probe now all read the same evolving node/edge records.
        solver_stats = st.inject_solver_stats()
        util = solver_stats["member_utilisation"]
        node_records = {node["identity"]: node for node in doc["nodes"]}
        for engine_object in engines_live:
            engine_object.solver_stats = node_records[
                engine_object.identity]["solver_stats"]
        mat = ffv.material_ids_for_colour_mode(
            base_mat, tris, owner, util, bands, band_mat, colour_mode)
        trial.mesh.material_ids = mat
        solved = st.positions()
        elastic = {
            identity: solved[index]
            for identity, index in st.solver.index.items()
        }
        visible_positions = elastic
        muzzle = visible_positions[barrel_node] + muzzle_offset
        if projectiles.rounds:
            round_ = projectiles.rounds[-1]
            visible_positions["viewer.projectile.tail"] = round_.origin
            visible_positions["viewer.projectile.round"] = round_.position
        else:
            visible_positions["viewer.projectile.tail"] = muzzle - axis * 0.001
            visible_positions["viewer.projectile.round"] = muzzle
        trial.articulated.displace(visible_positions)
        thermal_state = {}
        for engine_object in engines_live:
            trial.view.update_mesh_overlay(engine_object.identity,
                                           engine_object.mesh())
            thermal_state.update(engine_object.temperatures())
        trial.view.set_thermal_state(thermal_state)
        trial.view.restage_static_mesh()
        trial.view._elevation = el
        trial.view._zoom = zoom
        trial.present(az)
        pygame.display.flip()
        frames += 1
        peak = max(peak, float(util.max()))
        if frames % 60 == 0:
            hot = int((util > 0.85).sum())
            mechanism_motion = float(np.linalg.norm(
                st.dynamic_displacement.reshape(-1, 6)[:, :3]))
            print(f"  {clock.get_fps():5.1f} fps   mechanism"
                  f" {mechanism_motion:8.4f}   bore {bore:3.0f} mm"
                  f"   {fired} fired/{projectiles.in_flight()} flying"
                  f"   now {util.max() * 100:6.2f}%"
                  f"   peak {peak * 100:6.2f}% of yield   {hot} hot",
                  flush=True)
            if last_support is not None:
                inner = min(last_support["legs"][name]["extension_m"]
                            for name in last_support["legs"]
                            if name.startswith("inner."))
                outer = min(last_support["legs"][name]["extension_m"]
                            for name in last_support["legs"]
                            if name.startswith("outer."))
                print(f"    HCU {hcu_command.mode}: full common-oil draw, "
                      f"inner {inner:.3f} m outer {outer:.3f} m, "
                      f"flow {last_support['pump']['delivered_l_min']:.1f} L/min",
                      flush=True)
            if last_cooling is not None and last_rejector is not None:
                print(f"    cooling: glycol {last_cooling['coolant_temp_k'] - 273.15:.1f} C, "
                      f"potable {last_cooling['potable_water_temp_k'] - 273.15:.1f} C, "
                      f"diffuser {last_rejector['diffuser_outlet_temp_k'] - 273.15:.1f} C",
                      flush=True)
    pygame.quit()


def render_still(g, path: str = "station_powerplant_still.png",
                 *, width: int = 1600, height: int = 1000,
                 azimuth: float = 0.70, service_detail: bool = False) -> str:
    """Render with the same graph, baked mesh, and engine objects as --live."""
    from pathlib import Path
    from PIL import Image
    from firing_frame_view import Trial

    document = projectile_visual_document(paint(g.as_document()))
    if service_detail:
        def service_item(identity: str) -> bool:
            return identity.startswith((
                "engine.", "powerplant.", "station.storage.",
                "station.service.", "station.fuel.", "trunk.",
                "stand.work", "stand.bay",
                "stand.hcu"))
        nodes = [n for n in document["nodes"]
                 if service_item(n["identity"])]
        kept = {n["identity"] for n in nodes}
        edges = [e for e in document["edges"]
                 if service_item(e["identity"])
                 and e["a"] in kept and e["b"] in kept]
        document = dict(document, nodes=nodes, edges=edges)
    engines_live = station_engine_render_objects(document)
    document = dict(
        document,
        mesh_overlays=[{"identity": o.identity, "mesh": o.mesh()}
                       for o in engines_live])
    trial = Trial(document, width=width, height=height, hidden=True)
    thermal = {}
    for engine_object in engines_live:
        engine_object.step(1.0 / 60.0)
        trial.view.update_mesh_overlay(engine_object.identity,
                                       engine_object.mesh())
        thermal.update(engine_object.temperatures())
    trial.view.set_thermal_state(thermal)
    trial.view.restage_static_mesh()
    image = trial.draw(azimuth)
    target = Path(path).resolve()
    Image.fromarray(image).save(target)
    return str(target)


def main(argv) -> None:
    print("building the drum and everything on it ...", flush=True)
    g, plat, box, spec, st, site = build()
    print("\n".join(spec_sheet(g, plat, box, spec)), flush=True)
    problems = list(dict.fromkeys(g.check()))
    print(f"\n  check  {'PASS' if not problems else problems[:2]}", flush=True)
    print()
    if "--live" in argv:
        live(g, plat,
             push_to_equilibrium="--push-to-equilibrium" in argv,
             stand_spec=st, urgent="--no-urgent" not in argv,
             adaptive="--full-beam-live" not in argv)
    if "--still" in argv:
        print(f"  still  {render_still(g)}", flush=True)
    if "--powerplant-still" in argv:
        print("  powerplant still  " + render_still(
            g, "station_powerplant_detail.png", width=1500, height=900,
            azimuth=0.76, service_detail=True), flush=True)
    if "--unlocked-settle" in argv:
        audit = first_yield_under_unlocked_gravity(g.as_document())
        first = audit["craft_first_yield_g"]
        print("  unlocked settle: four crown locks released; "
              f"finite={audit['finite']}; 1g deflection "
              f"{audit['max_deflection_at_1g_m']:.6g} m", flush=True)
        if first is None:
            print("  no craft member reaches yield through 1.0 g", flush=True)
        else:
            print(f"  FIRST CRAFT YIELD at {first:.9f} g", flush=True)
            for item in audit["craft_first_yields"]:
                print(f"    {item['utilisation'] * 100:9.3f}%  "
                      f"{item['identity']}", flush=True)
        if audit["first_yield_g"] != audit["craft_first_yield_g"]:
            print("  independent site-machine first yield at "
                  f"{audit['first_yield_g']:.9f} g: "
                  f"{audit['first_yields'][0]['identity']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
