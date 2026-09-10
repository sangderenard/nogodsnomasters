"""The engine as ONE mesh: vertices, triangles, normals, and a material
per triangle -- the dressed, colour-coded design lifted into the form
a renderer, an exporter and a game all consume. Static parts (castings,
covers, headers, dressing) are built once per engine; MOVING parts
(pistons, rods, crank, valves, springs, followers, rotors, crossheads)
are rebuilt from the live crank angle.

Materials mirror the spectral analyzer's Phong record (ambient,
spec_strength, shininess, inner colour) over a PBR base (albedo,
roughness, opacity): the same fields base_material.frag.glsl reads
from its SSBOs, so this mesh's material table can be uploaded to
that renderer unchanged, and the software Phong in mesh_visualizer
evaluates the same lighting on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np

from vehicle_mesh import build_drivetrain_solid_parts, build_throttle_parts


@dataclass(frozen=True)
class Material:
    name: str
    label: str
    albedo: tuple            # linear-ish sRGB 0..1
    opacity: float = 1.0
    roughness: float = 0.6
    ambient: float = 0.22    # phong[0]
    spec_strength: float = 0.25   # phong[1]
    shininess: float = 24.0  # phong[2]
    metallic: float = 0.0

    def pbr_record(self) -> list[float]:
        """PBRBaseRecord, 16 floats (albedo, roughness, metallic,
        transmission, ior, opacity, emission, reserved)."""
        return [*self.albedo, self.roughness, self.metallic, 0.0, 1.5, self.opacity, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0]

    def phong_record(self) -> list[float]:
        """PhongRecord, 8 floats (ambient, spec_strength, shininess,
        reserved, inner colour rgb, pad)."""
        return [self.ambient, self.spec_strength, self.shininess, 0.0, *self.albedo, 0.0]


def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# (name-substrings, material) -- first match wins; the colour system of the design renders
MATERIAL_RULES = [
    (("_valve_head", "_valve_stem"), Material("valve", "valves", _rgb("#ece4d0"), 1.0, 0.25, 0.2, 0.7, 60.0, 0.8)),
    (("_spring",), Material("spring", "valve springs", _rgb("#e0b030"), 1.0, 0.35, 0.2, 0.5, 40.0, 0.6)),
    (("_retainer", "_bucket", "_rocker", "_pushrod"), Material("follower", "followers / rockers / pushrods", _rgb("#c08040"), 1.0, 0.4, 0.2, 0.4, 30.0, 0.5)),
    (("camshaft", "cam_lobe", "cam_in_block"), Material("cam", "camshaft + lobes", _rgb("#d09040"), 1.0, 0.3, 0.2, 0.6, 50.0, 0.7)),
    (("_piston",), Material("piston", "pistons", _rgb("#c9d3da"), 1.0, 0.3, 0.22, 0.6, 50.0, 0.7)),
    (("_con_rod", "_piston_rod", "_crosshead"), Material("rod", "con rods", _rgb("#b8892a"), 1.0, 0.35, 0.2, 0.5, 40.0, 0.6)),
    (("crank_", "crankpin", "flywheel", "_pinion", "_freewheel", "_eccentric"), Material("crank", "crankshaft + flywheel", _rgb("#23262b"), 1.0, 0.45, 0.2, 0.45, 32.0, 0.5)),
    (("_rotor",), Material("rotor", "rotor", _rgb("#e0c060"), 1.0, 0.35, 0.2, 0.5, 40.0, 0.6)),
    (("_intake_port", "intake_port_", "transfer_port", "scavenge", "_air_port", "_gas_port"), Material("intake_port", "intake ports", _rgb("#4c8dff"), 0.95, 0.5, 0.25, 0.3, 20.0)),
    (("_exhaust_port", "exhaust_port_", "_head_exhaust", "_crank_exhaust"), Material("exhaust_port", "exhaust ports", _rgb("#ff5a3c"), 0.95, 0.5, 0.25, 0.3, 20.0)),
    (("spark_plug", "glow_plug", "_flame_port", "leading_plug", "trailing_plug"), Material("igniter", "plugs / igniters", _rgb("#ffd23c"), 1.0, 0.4, 0.25, 0.5, 30.0)),
    (("_injector",), Material("injector", "injector bosses", _rgb("#3ccf6a"), 1.0, 0.4, 0.25, 0.4, 30.0)),
    (("_admission", "_drain_cock", "_lubricator", "_rod_gland"), Material("expander_port", "expander passages / cocks", _rgb("#b070ff"), 0.95, 0.5, 0.25, 0.3, 20.0)),
    (("edge_powertrain_cylinder", "edge_powertrain_exhaust", "node_powertrain_exhaust_collector"), Material("exhaust", "exhaust primaries / collector", _rgb("#e06040"), 0.6, 0.5, 0.25, 0.35, 24.0)),
    # Air filter, throttle body, and plenum/runners each get their own
    # distinct color -- these used to share one "intake" material, which
    # made the real, genuinely different boxes in a real distributor
    # chain (air_filter -> throttle_body(s) -> plenum chamber(s) ->
    # runners -> ports) hard to tell apart by eye even though the real
    # graph topology connecting them was always correct.
    (("node_powertrain_air_filter",), Material("air_filter", "air filter / cleaner", _rgb("#c8102e"), 0.85, 0.7, 0.28, 0.25, 18.0)),   # the red cotton-gauze cleaner look
    (("throttle_plate", "throttle_linkage"), Material("throttle_plate", "throttle plate + linkage", _rgb("#e8e8ec"), 1.0, 0.3, 0.2, 0.6, 45.0, 0.6)),
    (("node_powertrain_throttle_body", "throttle_body"), Material("throttle_body", "throttle body / barrel(s)", _rgb("#d4a017"), 0.85, 0.5, 0.25, 0.35, 26.0)),
    (("runner", "node_powertrain_intake_plenum", "stack"), Material("intake", "intake runners / plenum", _rgb("#8fb0d8"), 0.5, 0.5, 0.25, 0.3, 24.0)),
    (("fuel_rail", "rail_feed", "fuel_filter", "edge_fuel"), Material("fuel", "fuel rail + filter", _rgb("#3ccf6a"), 0.75, 0.5, 0.25, 0.3, 24.0)),
    (("lead", "distributor", "ignition_coil", "coil_pack", "leading_coil", "trailing_coil"), Material("ignition", "distributor / coils / plug leads", _rgb("#d21f26"), 0.95, 0.45, 0.28, 0.35, 26.0)),   # the red aftermarket-ignition look
    (("magneto", "glow_plug_bus"), Material("magneto", "magneto / glow-plug bus", _rgb("#202020"), 0.9, 0.7, 0.3, 0.15, 12.0)),
    (("oil_filter", "oil_reserve", "scavenge", "oil_bath", "oil_pump", "edge_powertrain_oil", "edge_powertrain_pan", "edge_powertrain_trough", "splash"), Material("lube", "lubrication", _rgb("#8a6a30"), 0.75, 0.6, 0.25, 0.25, 16.0)),
    (("port_",), Material("casting_port", "casting ports (fill, galleries, coolant, breather)", _rgb("#404040"), 0.95, 0.7, 0.3, 0.15, 12.0)),
    (("head_casting", "cam_box"), Material("head", "head castings", _rgb("#98a4b4"), 0.24, 0.65, 0.25, 0.25, 16.0)),
    (("valve_cover", "valley_cover", "case_top_cover"), Material("cover", "covers", _rgb("#606a78"), 0.2, 0.55, 0.25, 0.35, 20.0)),
    (("_bore", "_water_jacket", "_fin_", "_head", "_crank_cover", "_hopper", "_rotor_housing", "_side_plate", "_valve_chest"), Material("cylinder", "cylinders / jackets / fins", _rgb("#57626e"), 0.22, 0.65, 0.25, 0.25, 16.0)),
    (("crankcase", "sump", "front_cover", "rear_main", "bedplate", "main_pedestal", "frame_plate", "guide_bar", "rear_accessory"), Material("case", "crankcase / sump / frame", _rgb("#34363a"), 0.2, 0.7, 0.25, 0.2, 14.0)),
    (("node_mount_",), Material("mount", "engine mounts / isolators", _rgb("#8a3324"), 0.3, 0.6, 0.3, 0.3, 18.0)),
    # universal bolt-ons (engine_parts.py)
    (("_pulley", "belt_tensioner", "_sprocket", "timing_chain_run"), Material("pulley", "pulleys / tensioner / sprockets", _rgb("#3a3d44"), 1.0, 0.4, 0.22, 0.5, 34.0, 0.7)),
    (("harmonic_balancer",), Material("damper", "harmonic damper", _rgb("#2c2f36"), 1.0, 0.45, 0.2, 0.45, 32.0, 0.5)),
    (("starter_motor", "recoil_starter", "crank_nose_fitting"), Material("starter", "starter hardware", _rgb("#4a4f58"), 1.0, 0.5, 0.25, 0.35, 24.0, 0.4)),
    (("expansion_bottle", "heater_core"), Material("coolant_gear", "coolant bottle / heater core", _rgb("#e8e6d8"), 0.85, 0.5, 0.3, 0.2, 16.0)),
    (("egr_valve",), Material("egr", "EGR valve", _rgb("#8a5a3a"), 1.0, 0.55, 0.25, 0.3, 20.0)),
    (("pcv_valve",), Material("pcv", "PCV valve", _rgb("#202020"), 1.0, 0.6, 0.3, 0.2, 14.0)),
    (("timing_cover", "bellhousing"), Material("housing", "timing cover / bellhousing", _rgb("#3f4349"), 0.3, 0.6, 0.25, 0.25, 16.0)),
    # forced induction (engine_parts.py)
    (("blower_case", "blower_manifold", "blower_snout"), Material("blower", "blower case / manifold / snout", _rgb("#b9bec6"), 1.0, 0.3, 0.22, 0.6, 48.0, 0.85)),
    (("blower_burst_panel",), Material("burst_panel", "burst panel / restraint", _rgb("#2a2a2e"), 1.0, 0.6, 0.25, 0.25, 16.0)),
    (("blower_pulley",), Material("blower_pulley", "blower drive pulley", _rgb("#3a3d44"), 1.0, 0.4, 0.22, 0.5, 34.0, 0.7)),
    (("compressor_housing", "blow_off_valve", "charge_cooler", "charge_pipe"), Material("charge_side", "compressor / charge cooler / BOV", _rgb("#c9ced6"), 0.95, 0.35, 0.24, 0.55, 40.0, 0.8)),
    (("turbine_housing", "wastegate", "downpipe", "up_pipe"), Material("hot_side", "turbine housing / wastegate / downpipe", _rgb("#8c5a3c"), 0.9, 0.55, 0.25, 0.3, 20.0)),
    (("barrel_valve",), Material("barrel_valve", "barrel valve", _rgb("#3ccf6a"), 1.0, 0.4, 0.25, 0.4, 30.0)),
    # aircraft / industrial / air-cooled families (engine_parts.py)
    (("prop_reduction_gearbox", "prop_shaft", "prop_hub"), Material("prop_drive", "propeller reduction / shaft / hub", _rgb("#5a6068"), 1.0, 0.35, 0.22, 0.5, 36.0, 0.75)),
    (("propeller_governor", "flyball_governor", "governor_latch"), Material("governor", "governors + linkage", _rgb("#8a7a2a"), 1.0, 0.4, 0.25, 0.45, 30.0, 0.5)),
    (("injection_pump",), Material("injection_pump", "diesel injection pump", _rgb("#2f6b3a"), 1.0, 0.5, 0.25, 0.35, 24.0)),
    (("oil_cooler",), Material("oil_cooler", "oil cooler", _rgb("#8a6a30"), 0.85, 0.55, 0.25, 0.3, 20.0)),
    (("cooling_fan_housing", "cooling_tin_", "_baffle", "cowl_flap_"), Material("cooling_tin", "fan housing / tins / baffles / cowl flaps", _rgb("#4b545e"), 0.55, 0.6, 0.25, 0.3, 18.0)),
]
DEFAULT_MATERIAL = Material("other", "other graph parts", _rgb("#b0b0b8"), 0.35, 0.7, 0.25, 0.2, 14.0)
MATERIALS: list[Material] = [m for _, m in MATERIAL_RULES] + [DEFAULT_MATERIAL]
MATERIAL_INDEX = {m.name: i for i, m in enumerate(MATERIALS)}

MOVING_KEYS = ("_piston", "_con_rod", "_piston_rod", "_crosshead", "crank_", "crankpin", "_valve_head", "_valve_stem",
               "_spring", "_retainer", "_bucket", "_rocker", "_pushrod", "cam_lobe", "_rotor", "_eccentric")
STATIC_OVERRIDES = ("_rotor_housing",)


def classify(name: str) -> int:
    for keys, mat in MATERIAL_RULES:
        if any(k in name for k in keys):
            return MATERIAL_INDEX[mat.name]
    return MATERIAL_INDEX[DEFAULT_MATERIAL.name]


def is_moving(name: str) -> bool:
    if any(k in name for k in STATIC_OVERRIDES):
        return False
    return any(k in name for k in MOVING_KEYS)


def wanted_in_view(name: str) -> bool:
    """Graph nodes/edges the engine view keeps: the dressing and the
    routed lines, not every harness wire and mount."""
    if "muffler" in name or "tailpipe" in name:
        # chassis plumbing (engine_parts.py marks these chassis_side):
        # real, in the graph for the exhaust circuit and the acoustic
        # model, but hung from the body, not the engine -- a metre of
        # tailpipe is not part of the engine's own view
        return False
    if name.startswith("node_"):
        return any(k in name for k in ("intake_plenum", "throttle_body", "air_filter", "fuel_rail", "distributor", "ignition_coil",
                                       "coil_pack", "leading_coil", "trailing_coil", "magneto", "oil_filter", "oil_reserve",
                                       "scavenge", "oil_bath", "glow_plug_bus", "exhaust_collector", "fuel_filter", "oil_pump",
                                       "mount_",
                                       # universal bolt-ons (engine_parts.py)
                                       "harmonic_balancer", "flywheel", "starter_motor", "recoil_starter", "crank_nose_fitting",
                                       "_pulley", "belt_tensioner", "timing_cover", "timing_drive", "expansion_bottle",
                                       "heater_core", "egr_valve", "pcv_valve", "bellhousing",
                                       # forced induction
                                       "blower_", "supercharger_rotor", "compressor_housing", "turbine_housing",
                                       "wastegate", "blow_off_valve", "charge_cooler", "downpipe", "barrel_valve",
                                       "fuel_pump",
                                       # aircraft / industrial / air-cooled
                                       "prop_reduction_gearbox", "prop_shaft", "prop_hub", "propeller_governor",
                                       "flyball_governor", "governor_latch", "injection_pump", "oil_cooler",
                                       "cooling_fan_housing", "cooling_tin_", "_baffle", "cowl_flap_", "magneto_2"))
    if name.startswith("edge_"):
        return any(k in name for k in ("cylinder", "exhaust", "runner", "lead", "rail_feed", "oil", "pan", "trough", "scavenge",
                                       "air_filter", "stack", "coil", "throttle", "splash",
                                       # real visible lines from engine_parts.py: hoses, the EGR
                                       # tube, the timing run -- NOT the hub/bolted-joint edges,
                                       # which are joints, not pipes, and stay unlisted
                                       "heater_", "expansion_bottle", "egr_tube", "timing_chain",
                                       "blower_belt", "charge_pipe", "up_pipe", "turbine_to_downpipe",
                                       "blower_case_to_plenum_charge", "hat_nozzle", "barrel_valve",
                                       "cooling_tin_bank", "dry_sump_belt", "cooling_fan_belt", "oil_cooler",
                                       "injection_pump_to_rail"))
    return True


@dataclass
class EngineMesh:
    vertices: np.ndarray                 # (V, 3)
    normals: np.ndarray                  # (V, 3)
    triangles: np.ndarray                # (T, 3) indices
    material_ids: np.ndarray             # (T,)
    part_names: list = field(default_factory=list)
    part_ranges: list = field(default_factory=list)   # (tri_start, tri_end) per part
    moving: bool = False

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def tri_vertices(self) -> np.ndarray:
        return self.vertices[self.triangles]          # (T, 3, 3)

    def tri_normals(self) -> np.ndarray:
        return self.normals[self.triangles]


def _from_parts(parts, moving: bool) -> EngineMesh:
    verts, norms, mats, names, ranges = [], [], [], [], []
    t0 = 0
    for p in parts:
        n_tri = len(p.vertices) // 3
        if n_tri == 0:
            continue
        verts.append(np.asarray(p.vertices, dtype=np.float64))
        norms.append(np.asarray(p.normals, dtype=np.float64))
        mats.append(np.full(n_tri, classify(p.name), dtype=np.int32))
        names.append(p.name); ranges.append((t0, t0 + n_tri)); t0 += n_tri
    if not verts:
        return EngineMesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int32), [], [], moving)
    v = np.concatenate(verts); n = np.concatenate(norms)
    tris = np.arange(len(v), dtype=np.int64).reshape(-1, 3)
    return EngineMesh(v, n, tris, np.concatenate(mats), names, ranges, moving)


def build_engine_mesh(graph: dict, crank_angle_deg: float = 0.0, covers_off: bool = False) -> tuple[EngineMesh, EngineMesh]:
    """(static, moving) meshes for this graph at this crank angle."""
    parts = [p for p in build_drivetrain_solid_parts(graph, crank_angle_deg=crank_angle_deg, covers_off=covers_off)
             if wanted_in_view(p.name)]
    static = _from_parts([p for p in parts if not is_moving(p.name)], moving=False)
    moving = _from_parts([p for p in parts if is_moving(p.name)], moving=True)
    return static, moving


def build_moving_mesh(graph: dict, crank_angle_deg: float, covers_off: bool = False, spring_style: str = "helix") -> EngineMesh:
    """Only the moving parts, at this crank angle -- the per-frame call."""
    from cylinder_ports import deserialize_layout, build_parts_from_layout
    layout_data = graph.get("cylinder_layout")
    if not layout_data:
        return _from_parts([], moving=True)
    parts = build_parts_from_layout(deserialize_layout(layout_data), crank_angle_deg=crank_angle_deg,
                                    covers_off=covers_off, moving_only=True, spring_style=spring_style)
    return _from_parts([p for p in parts if is_moving(p.name)], moving=True)


# ---------------------------------------------------------------------
# Throttle-plate animation: a SEPARATE small baked set, keyed by
# throttle position (0..1) instead of crank angle -- composited
# alongside whatever crank-angle frame is showing, not folded into it
# (a full crank-angle x throttle-position cross product would be N
# times the frame count for geometry that's a handful of triangles;
# indexing two small independent sets and drawing both is the same
# real "bake once, index during playback" contract as EngineAnimation,
# just on its own real driving parameter).
# ---------------------------------------------------------------------

def build_throttle_mesh(graph: dict, throttle_frac: float = 1.0) -> EngineMesh:
    parts = build_throttle_parts(graph, throttle_frac=throttle_frac)
    return _from_parts(parts, moving=True)


@dataclass
class ThrottleAnimation:
    fracs: np.ndarray
    frames: list                    # EngineMesh per fraction, baked upfront (cheap: a handful of triangles each)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def frame_index(self, throttle_frac: float) -> int:
        f = max(0.0, min(1.0, float(throttle_frac)))
        return int(np.argmin(np.abs(self.fracs - f)))

    def frame_for(self, throttle_frac: float):
        return self.frames[self.frame_index(throttle_frac)]


def build_throttle_animation(graph: dict, divisions: int = 9) -> ThrottleAnimation:
    """Baked once, upfront -- unlike EngineAnimation's incremental
    bake_next(), there's no per-frame cost worth spreading across
    ticks here (this is a plate and a lever arm, not a whole engine's
    moving parts)."""
    fracs = np.linspace(0.0, 1.0, max(2, divisions))
    frames = [build_throttle_mesh(graph, throttle_frac=float(f)) for f in fracs]
    return ThrottleAnimation(fracs=fracs, frames=frames)


def export_obj_mtl(static: EngineMesh, moving: EngineMesh, path_obj: str) -> tuple[str, str]:
    """Write the whole engine as OBJ + MTL with one material per rule
    (Kd = albedo, d = opacity, Ns = shininess, Ks = spec strength)."""
    path_mtl = os.path.splitext(path_obj)[0] + ".mtl"
    with open(path_mtl, "w") as f:
        for m in MATERIALS:
            f.write(f"newmtl {m.name}\nKd {m.albedo[0]:.4f} {m.albedo[1]:.4f} {m.albedo[2]:.4f}\n"
                    f"Ka {m.ambient:.3f} {m.ambient:.3f} {m.ambient:.3f}\nKs {m.spec_strength:.3f} {m.spec_strength:.3f} {m.spec_strength:.3f}\n"
                    f"Ns {m.shininess:.1f}\nd {m.opacity:.3f}\nillum 2\n\n")
    with open(path_obj, "w") as f:
        f.write(f"mtllib {os.path.basename(path_mtl)}\n")
        offset = 1
        for mesh in (static, moving):
            for k, name in enumerate(mesh.part_names):
                a, b = mesh.part_ranges[k]
                tri = mesh.triangles[a:b]
                verts = mesh.vertices[tri.reshape(-1)]
                norms = mesh.normals[tri.reshape(-1)]
                f.write(f"o {name}\nusemtl {MATERIALS[int(mesh.material_ids[a])].name}\n")
                for v in verts:
                    f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
                for n in norms:
                    f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
                for t in range(len(tri)):
                    i0 = offset + 3 * t
                    f.write(f"f {i0}//{i0} {i0 + 1}//{i0 + 1} {i0 + 2}//{i0 + 2}\n")
                offset += len(verts)
    return path_obj, path_mtl


def material_table() -> dict:
    """The material records in the spectral analyzer's chunk layout."""
    return {"names": [m.name for m in MATERIALS], "labels": [m.label for m in MATERIALS],
            "pbr": np.array([m.pbr_record() for m in MATERIALS], dtype=np.float32),
            "phong": np.array([m.phong_record() for m in MATERIALS], dtype=np.float32),
            "chunk_strides": {"pbr": 16, "phong": 8}}


# ---------------------------------------------------------------------
# Baked animation: moving parts at chosen crank divisions
# ---------------------------------------------------------------------

CYCLE_DEG = 720.0


@dataclass
class EngineAnimation:
    """The moving parts baked at a set of crank angles. A game plays
    these frames scaled by rpm (frame_for(angle)) and never re-derives
    geometry at run time; the divisions are the caller's choice -- an
    integer count evenly over one 720-degree cycle, or an explicit
    list of angles in degrees."""
    angles_deg: np.ndarray
    frames: list                    # EngineMesh per angle
    covers_off: bool

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    _graph: dict = None
    _detail: float = None

    @property
    def baked(self) -> int:
        return sum(1 for f in self.frames if f is not None)

    def bake_next(self) -> bool:
        """Bake the next missing frame (in an order that fills the cycle
        evenly: 0, half, quarters, ...). Returns False when complete."""
        missing = [i for i, f in enumerate(self.frames) if f is None]
        if not missing:
            return False
        n = len(self.frames)
        order = sorted(range(n), key=lambda i: (bin(i)[::-1].index("1") if i else -1, i)) if n > 1 else [0]
        # bit-reversed-ish fill: 0 first, then the middle, then quarters
        seq = [i for i in _even_fill_order(n) if self.frames[i] is None]
        i = seq[0]
        import mesh_primitives as mp
        prev = mp.DETAIL
        if self._detail is not None:
            mp.set_detail(self._detail)
        try:
            self.frames[i] = build_moving_mesh(self._graph, float(self.angles_deg[i]), covers_off=self.covers_off,
                                               spring_style=getattr(self, "_spring_style", "helix"))
        finally:
            mp.DETAIL = prev
        return any(f is None for f in self.frames)

    def frame_index(self, crank_angle_deg: float) -> int:
        a = float(crank_angle_deg) % CYCLE_DEG
        # nearest BAKED angle on the cycle (wrapping)
        d = np.abs((self.angles_deg - a + CYCLE_DEG / 2.0) % CYCLE_DEG - CYCLE_DEG / 2.0)
        d = np.where([f is None for f in self.frames], np.inf, d)
        return int(np.argmin(d))

    def frame_for(self, crank_angle_deg: float):
        return self.frames[self.frame_index(crank_angle_deg)]


def _even_fill_order(n: int) -> list:
    """0, n/2, n/4, 3n/4, ... -- so a partly baked animation already
    spans the cycle instead of clustering at its start."""
    if n <= 1:
        return [0]
    order, seen = [], set()
    step = n
    while step >= 1:
        for i in range(0, n, step):
            if i not in seen:
                order.append(i); seen.add(i)
        step //= 2
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order


def animation_angles(divisions) -> np.ndarray:
    if isinstance(divisions, int):
        return np.linspace(0.0, CYCLE_DEG, max(1, divisions), endpoint=False)
    return np.array(sorted(float(a) % CYCLE_DEG for a in divisions), dtype=np.float64)


def build_animation(graph: dict, divisions=24, covers_off: bool = False, detail: float | None = None,
                    spring_style: str = "helix") -> EngineAnimation:
    """Bake the moving parts once at every requested crank angle."""
    anim = start_animation(graph, divisions, covers_off, detail, spring_style)
    while anim.bake_next():
        pass
    return anim


def start_animation(graph: dict, divisions=24, covers_off: bool = False, detail: float | None = None,
                    spring_style: str = "helix") -> EngineAnimation:
    """An animation with only its first frame baked; bake_next() fills
    the rest one frame at a time (a live view calls it between renders
    so an engine switch never stalls the screen). Until a frame is
    baked, frame_for() returns the nearest baked one."""
    angles = animation_angles(divisions)
    anim = EngineAnimation(angles_deg=angles, frames=[None] * len(angles), covers_off=covers_off)
    anim._graph = graph
    anim._detail = detail
    anim._spring_style = spring_style
    anim.bake_next()
    return anim
