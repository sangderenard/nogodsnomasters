"""Real drivetrain-graph geometry, built with the ported mesh primitives
(mesh_primitives.py) from the real graph engine_toy already gets out of
the production compiler's `_vehicle_powertrain_graph` subunit
(drivetrain_graph.build_drivetrain_graph).

Two kinds of shape come out of one graph, and they are NOT equally real:

  - Every edge becomes its declared real section -- a tube by default,
    or the web and two flanges of a section-authored welded I member --
    using real node positions
    (`reference_position`, straight from the game's own `node()` closure)
    and a real radius (the edge's own `radius` attribute when the real
    subunit declared one, else the same 0.012m default the real `edge()`
    closure itself falls back to). This is the same routing data the real
    in-game renderer draws with, run through the same tube-sweep
    algorithm.
  - Every node becomes a placeholder box -- there is no real box-extent
    data at this subunit level (that lives in the whole-vehicle
    `part_mesh`/VehicleConfiguration system this toy deliberately doesn't
    build). Box size here is a disclosed toy-only visualization aid
    (scaled off mass_kg when present, a small fixed default otherwise),
    not a real dimension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import math

from mesh_primitives import tube_mesh, cuboid_mesh, tube_wireframe, cuboid_wireframe, write_obj, _perpendicular_basis, _normalized3

DEFAULT_TUBE_RADIUS_M = 0.012
DEFAULT_BODY_HALF_EXTENT_M = 0.035

# circuit_identity -> the real state field on EngineCycleState carrying
# that circuit's actual simulated temperature (engine_cycle_sim.py).
# Anything not in this map -- torque shafts, mounts, wiring, mechanical
# rotating masses -- has no real simulated temperature at all, and the
# visualizer must show that honestly (a fixed structural color) rather
# than invent one.
CIRCUIT_IDENTITY_TO_THERMAL_GROUP = {
    "coolant": "coolant",
    "oil": "oil",
    "exhaust": "exhaust",
    "intake-air": "intake",
}
# real, approximate operating ranges per group (kelvin), used only to
# normalize color -- not a claim of precision, just cold..hot for a
# defensible real range each circuit actually spans in this sim
THERMAL_GROUP_RANGE_K = {
    "coolant": (293.15, 380.0),   # ambient .. near boiling
    "oil": (293.15, 410.0),       # ambient .. a real hot-oil ceiling
    "exhaust": (293.15, 1200.0),  # ambient .. genuinely hot exhaust gas
    "intake": (293.15, 450.0),    # ambient .. heavily boosted charge
}


def _join_geometry(parts):
    live = [(v, n) for v, n in parts if len(v)]
    if not live:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate([p[0] for p in live]), np.concatenate([p[1] for p in live])


def _section_frame(start, end, section_up) -> np.ndarray:
    """Rows are the authored section's local x/y/z axes."""
    x = _normalized3(np.asarray(end, float) - np.asarray(start, float))
    z = np.asarray(section_up, float)
    z = z - float(z @ x) * x
    if np.linalg.norm(z) <= 1.0e-10:
        raise ValueError("section_up must not be parallel to shaped member")
    z = _normalized3(z)
    y = _normalized3(np.cross(z, x))
    z = _normalized3(np.cross(x, y))
    return np.vstack((x, y, z))


def _oriented_box(center, half_extent, frame):
    vertices, normals = cuboid_mesh((0.0, 0.0, 0.0), half_extent)
    return vertices @ frame + np.asarray(center, float), normals @ frame


def _plate_cell_mesh(node: dict):
    """Exact visual cell belonging to one structural plate station.

    Surface emitters keep their wrench nodes on grid intersections.  The
    render centre may be offset inward at a sheet boundary so the collection
    of cells terminates on the declared plate edge instead of overhanging it.
    """
    axes = np.asarray(node["plate_cell_axes"], dtype=np.float64)
    centre = (np.asarray(node["reference_position"], dtype=np.float64)
              + np.asarray(node.get("render_center_offset_m", (0, 0, 0)),
                           dtype=np.float64))
    half = np.asarray(node["plate_cell_half_extent_m"], dtype=np.float64)
    return _oriented_box(centre, half, axes)


def _plate_cell_wireframe(node: dict):
    axes = np.asarray(node["plate_cell_axes"], dtype=np.float64)
    centre = (np.asarray(node["reference_position"], dtype=np.float64)
              + np.asarray(node.get("render_center_offset_m", (0, 0, 0)),
                           dtype=np.float64))
    half = np.asarray(node["plate_cell_half_extent_m"], dtype=np.float64)
    return [(np.asarray(a) @ axes + centre,
             np.asarray(b) @ axes + centre)
            for a, b in cuboid_wireframe((0.0, 0.0, 0.0), half)]


def _welded_i_mesh(start, end, edge):
    """Three welded plates, rendered in the same axes the solver uses."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    frame = _section_frame(start, end, edge.get("section_up", (0.0, 1.0, 0.0)))
    centre = (start + end) / 2.0
    length = float(np.linalg.norm(end - start))
    depth = float(edge["section_depth_m"])
    width = float(edge["section_flange_width_m"])
    web = float(edge["section_web_thickness_m"])
    flange = float(edge["section_flange_thickness_m"])
    web_depth = depth - 2.0 * flange
    pieces = [_oriented_box(centre, (length / 2.0, web / 2.0,
                                     web_depth / 2.0), frame)]
    for sign in (-1.0, 1.0):
        local_offset = np.array((0.0, 0.0,
                                 sign * (depth / 2.0 - flange / 2.0)))
        pieces.append(_oriented_box(centre + local_offset @ frame,
                                    (length / 2.0, width / 2.0, flange / 2.0),
                                    frame))
    return _join_geometry(pieces)


def _flat_strap_mesh(start, end, edge):
    """A bent strap segment as plate, in the solver's authored axes."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    frame = _section_frame(start, end, edge.get("section_up", (0.0, 0.0, 1.0)))
    centre = (start + end) / 2.0
    length = float(np.linalg.norm(end - start))
    return _oriented_box(
        centre,
        (length / 2.0, float(edge["section_thickness_m"]) / 2.0,
         float(edge["section_width_m"]) / 2.0), frame)


def _agt1500_module_geometry(node: dict) -> tuple[np.ndarray, np.ndarray]:
    """Procedural casing for one real AGT1500 module record."""
    from mesh_primitives import capped_tube_mesh, frustum_mesh, ring_mesh
    c = np.asarray(node["reference_position"], float)
    hx, hy, hz = (float(v) for v in node["body_half_extent_m"])
    axis = np.array([1.0, 0.0, 0.0])
    role = str(node.get("part_role", ""))
    left, right = c - axis * hx, c + axis * hx
    radius = min(hy, hz)
    if role == "air-inlet":
        return frustum_mesh(left, right, radius * 0.96, radius * 0.68,
                            sides=28)
    if role == "compressor":
        pieces = [frustum_mesh(left, right, radius * 0.92,
                               radius * 0.68, sides=32)]
        for fraction in np.linspace(0.08, 0.92, int(node.get("stages", 6)) + 1):
            p = left + (right - left) * fraction
            pieces.append(ring_mesh(p, axis, radius * (0.93 - 0.20 * fraction),
                                    radius * (0.78 - 0.18 * fraction),
                                    hx * 0.035, segments=28))
        return _join_geometry(pieces)
    if role == "heat-exchanger":
        pieces = [capped_tube_mesh(left, right, radius, sides=36)]
        # Concentric ceramic matrix bands plus radial separators make the
        # twin rotary recuperators readable as heat-exchanger drums.
        for rr in np.linspace(radius * 0.18, radius * 0.88, 6):
            pieces.append(ring_mesh(right + axis * 0.002, axis,
                                    rr + radius * 0.018,
                                    max(0.0, rr - radius * 0.018),
                                    hx * 0.025, segments=36))
        for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            radial = np.array([0.0, math.cos(angle), math.sin(angle)])
            pieces.append(tube_mesh(right + radial * radius * 0.10,
                                    right + radial * radius * 0.92,
                                    radius * 0.012, sides=6))
        return _join_geometry(pieces)
    if role == "combustor":
        pieces = [capped_tube_mesh(left, right, radius, sides=32),
                  ring_mesh(left - axis * 0.002, axis, radius * 0.92,
                            radius * 0.48, hx * 0.06, segments=32),
                  ring_mesh(right + axis * 0.002, axis, radius * 0.84,
                            radius * 0.42, hx * 0.06, segments=32)]
        return _join_geometry(pieces)
    if role == "turbine-stage":
        return _join_geometry([
            capped_tube_mesh(left, right, radius, sides=32),
            ring_mesh(left, axis, radius * 0.88, radius * 0.52,
                      hx * 0.08, segments=32),
            ring_mesh(right, axis, radius * 0.84, radius * 0.48,
                      hx * 0.08, segments=32)])
    if role == "exhaust":
        bend = c + np.array([hx * 0.15, hy * 0.55, 0.0])
        return _join_geometry([
            frustum_mesh(left, bend, radius * 0.62, radius * 0.86, sides=28),
            frustum_mesh(bend, right + np.array([0.0, hy * 0.45, 0.0]),
                         radius * 0.86, radius, sides=28)])
    if role == "gearbox":
        pieces = [cuboid_mesh(c, (hx, hy, hz))]
        for dx, rr in ((-hx * 0.55, min(hy, hz) * 0.40),
                       (hx * 0.50, min(hy, hz) * 0.30)):
            p = c + axis * dx
            pieces.append(capped_tube_mesh(p - axis * hx * 0.12,
                                           p + axis * hx * 0.12,
                                           rr, sides=24))
        return _join_geometry(pieces)
    if role == "lubrication":
        return capped_tube_mesh(left, right, radius, sides=24)
    return cuboid_mesh(c, (hx, hy, hz))


def build_agt1500_moving_parts(graph: dict, crank_angle_deg: float) -> list[SolidPart]:
    """Rotor hardware carried by the real AGT1500 module nodes."""
    if not str(graph.get("identity", "")).startswith("agt1500-abrams-turbine/"):
        return []
    parts = []
    phase = math.radians(float(crank_angle_deg))
    for node in graph["nodes"]:
        if node.get("of_engine") != "agt1500-abrams-turbine":
            continue
        role = node.get("part_role")
        if role not in {"compressor", "turbine-stage", "heat-exchanger"}:
            continue
        c = np.asarray(node["reference_position"], float)
        hx, hy, hz = (float(v) for v in node["body_half_extent_m"])
        radius = min(hy, hz)
        pieces = []
        speed = 0.035 if role == "heat-exchanger" else (1.0 if role == "compressor" else 0.72)
        rotor_phase = phase * speed
        if role == "compressor":
            stages, blades = int(node.get("stages", 6)), 10
        elif role == "turbine-stage":
            stages, blades = int(node.get("stages", 2)), 12
        else:
            stages, blades = 1, 12
        for stage in range(stages):
            x = c[0] + (stage - (stages - 1) / 2.0) * (2.0 * hx * 0.72 / max(stages, 1))
            hub = np.array([x, c[1], c[2]])
            pieces.append(tube_mesh(hub - np.array([hx * 0.06, 0.0, 0.0]),
                                    hub + np.array([hx * 0.06, 0.0, 0.0]),
                                    radius * 0.20, sides=16))
            for blade in range(blades):
                a = rotor_phase + 2.0 * math.pi * blade / blades + stage * 0.17
                radial = np.array([0.0, math.cos(a), math.sin(a)])
                pieces.append(tube_mesh(hub + radial * radius * 0.18,
                                        hub + radial * radius * 0.78,
                                        radius * 0.025, sides=5))
        vertices, normals = _join_geometry(pieces)
        name = node["identity"].replace(".", "_")
        parts.append(SolidPart(
            vertices=vertices, normals=normals,
            thermal_group=node.get("thermal_group"),
            name=f"node_{name}_rotor", declared_material="steel-shaft"))
    return parts


def _node_position_lookup(nodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {n["identity"]: np.array(n["reference_position"], dtype=np.float64) for n in nodes}


def _chamber_from_graph(graph: dict[str, Any]):
    """The declared CombustionChamber the graph carries (drivetrain_graph
    serialises it), or None -- so the head roof can be cut from the
    graph alone, the way everything else here is drawn."""
    data = graph.get("combustion_chamber")
    if not data:
        return None
    from engines import CombustionChamber
    try:
        return CombustionChamber(**data)
    except TypeError:
        return None


def _chassis_side_ids(nodes: list[dict[str, Any]]) -> set[str]:
    """Node identities hung from the body, not the engine (fuel tank,
    muffler/tailpipe, heater core, expansion bottle, ...). Node boxes
    for these are already skipped below; an edge with EITHER endpoint
    in this set is real chassis plumbing/wiring too (a coolant line
    into the expansion bottle, a fuel line into the tank) and gets the
    same cut -- otherwise the tube itself still reaches out to the
    chassis-side node's real position even with the box gone, which is
    exactly what was blowing out the engine-only view's camera fit."""
    return {n["identity"] for n in nodes if n.get("chassis_side")}


# Real components housed INSIDE the block/bell-housing -- a clutch disc,
# the camshaft, the oil pump, the flywheel wrench port -- as opposed to
# something genuinely bolted externally (an alternator, a water pump, a
# supercharger). Everything used to get the identical box treatment,
# which reads as "this is a separate part hanging off the engine" even
# for parts that are, in reality, buried inside it and never visible.
# Not a mass/size question (a real clutch disc can outweigh a small
# alternator) -- a real visibility/enclosure one, so this is its own
# explicit list, not folded into the mass-based scale below.
INTERNAL_HOUSING_IDENTITIES = frozenset({
    "powertrain.clutch", "powertrain.pre_clutch_flywheel_wrench",
    "powertrain.camshaft", "powertrain.oil_pump", "powertrain.direct_drive_bypass",
})
INTERNAL_HOUSING_SCALE_FRAC = 0.45


def _body_half_extent(node: dict[str, Any]) -> np.ndarray:
    """Toy-only sizing -- see module docstring. A node that declares its
    own real body_half_extent_m (the engine block casting -- see
    drivetrain_graph.py's powertrain.engine_block_body) is drawn at
    exactly that size, bypassing the mass-based guess below: a real
    bounding-geometry claim, not a generic placeholder. Otherwise: mass
    gives a rough, honest-effort visual scale (heavier part -> bigger
    box) when a real mass_kg is present; otherwise a small fixed
    placeholder. A real enclosed-inside-the-block part
    (INTERNAL_HOUSING_IDENTITIES) is shrunk on top of that -- still
    sized by its own real mass, just read as recessed/internal rather
    than a standalone bolt-on body."""
    explicit = node.get("body_half_extent_m")
    if explicit is not None:
        return np.array([float(v) for v in explicit])
    if node["kind"] == "powertrain-mount":
        # a real bolt-on mount/isolator bracket -- these used to be
        # skipped as bodiless waypoints; they're real structural
        # hardware (engine_mounts.py's own mounting policy assigns a
        # real technique/hardware to exactly these points), so a
        # small, real, roughly isolator-bracket-sized box, not the
        # generic tiny placeholder every other unsized accessory gets.
        return np.array([0.03, 0.02, 0.035])
    mass_kg = float(node.get("mass_kg") or 0.0)
    if mass_kg > 0.0:
        scale = DEFAULT_BODY_HALF_EXTENT_M * (mass_kg ** (1.0 / 3.0)) / (10.0 ** (1.0 / 3.0))
        scale = min(max(scale, 0.012), 0.18)
    else:
        scale = DEFAULT_BODY_HALF_EXTENT_M
    if node["identity"] in INTERNAL_HOUSING_IDENTITIES:
        scale *= INTERNAL_HOUSING_SCALE_FRAC
    return np.array([scale, scale, scale])


def _edge_points(a: np.ndarray, b: np.ndarray, edge: dict) -> list[np.ndarray]:
    return [np.asarray(a, dtype=np.float64),
            *(np.asarray(p, dtype=np.float64) for p in edge.get("waypoints", ())),
            np.asarray(b, dtype=np.float64)]


def _edge_tube_mesh(a: np.ndarray, b: np.ndarray, edge: dict,
                    radius: float, sides: int) -> tuple[np.ndarray, np.ndarray]:
    """One named edge may be several physical pipe/hose segments."""
    pieces = []
    points = _edge_points(a, b, edge)
    for p, q in zip(points, points[1:]):
        if not np.allclose(p, q):
            pieces.append(tube_mesh(p, q, radius, sides=sides))
    if not pieces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return (np.concatenate([v for v, _n in pieces]),
            np.concatenate([n for _v, n in pieces]))


def build_drivetrain_mesh(graph: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Returns a list of (vertices, normals, part_name) triangle-soup
    parts: one tube per edge, one placeholder box per node."""
    positions = _node_position_lookup(graph["nodes"])
    parts: list[tuple[np.ndarray, np.ndarray, str]] = []

    for edge in graph["edges"]:
        a = positions.get(edge["a"])
        b = positions.get(edge["b"])
        if a is None or b is None:
            continue
        if np.allclose(a, b):
            continue
        if edge.get("constraint") == "oil-splash-path":
            # not a pipe: the crank's dipper (hole_emitters splash
            # emitters draw the flung oil itself); the edge stays as the
            # circuit's declaration only
            continue
        if edge.get("section_shape") == "welded-i":
            vertices, normals = _welded_i_mesh(a, b, edge)
        elif edge.get("section_shape") == "flat-strap":
            vertices, normals = _flat_strap_mesh(a, b, edge)
        else:
            radius = float(edge.get("radius", DEFAULT_TUBE_RADIUS_M))
            if edge.get("fluid_route_class") == "trivial":
                radius *= 0.24
            sides = 6 if edge.get("routing") == "relaxed-multi-segment-harness" else 10
            vertices, normals = _edge_tube_mesh(a, b, edge, radius, sides)
        name = edge["identity"].replace("/", "_").replace(".", "_")
        parts.append((vertices, normals, f"edge_{name}"))

    for node in graph["nodes"]:
        if node.get("render_primitive") is False:
            continue
        if node["kind"] == "powertrain-mount":
            continue  # a mount is an attachment point, not a body -- no box to draw
        if node["kind"] == "engine-block-port":
            # a real port is a connection point machined into the
            # block/head (a cylinder's intake or exhaust port, an oil-
            # pan port) -- not a separate body at all. Drawing a generic
            # placeholder box at every one of these (16 for a V8's
            # cylinder ports alone) is exactly the "everything looks
            # like a bolted-on accessory" problem: the real tube edge
            # already attached there is what actually marks its
            # location.
            continue
        center = np.array(node["reference_position"], dtype=np.float64)
        if node.get("shape") == "plate-cell":
            vertices, normals = _plate_cell_mesh(node)
        else:
            half_extent = _body_half_extent(node)
            vertices, normals = cuboid_mesh(center, half_extent)
        name = node["identity"].replace("/", "_").replace(".", "_")
        parts.append((vertices, normals, f"node_{name}"))

    return parts


@dataclass
class WireframePart:
    segments: list[tuple[np.ndarray, np.ndarray]]
    thermal_group: str | None  # a key into THERMAL_GROUP_RANGE_K, or None (no real temp)
    name: str


def _thermal_group_for(circuit_identity: str | None, identity: str) -> str | None:
    if circuit_identity in CIRCUIT_IDENTITY_TO_THERMAL_GROUP:
        return CIRCUIT_IDENTITY_TO_THERMAL_GROUP[circuit_identity]
    # A per-cylinder block segment (drivetrain_graph.py's
    # powertrain.engine_block_body.cylinder_N) gets its OWN real group,
    # not the single shared "coolant"/"oil"/etc. every other part in a
    # circuit shares -- each cylinder bay has a genuinely different
    # real temperature (EngineCycleSim.state.cylinder_block_temps_k),
    # not one lumped value.
    if ".engine_block_body.cylinder_" in identity:
        return f"block_cyl_{identity.rsplit('_', 1)[-1]}"
    # nodes carry no circuit_identity of their own -- infer membership
    # the same substring way drivetrain_graph.py's own circuit discovery
    # already does elsewhere in this toy, for consistency
    lowered = identity.lower()
    if "coolant" in lowered or "radiator" in lowered:
        return "coolant"
    if "oil" in lowered:
        return "oil"
    # the turbo's real hot side (turbine housing, wastegate, downpipe,
    # the swept up-pipe waypoints) rides at exhaust temperature just
    # like the header it's bolted to; the cold side (compressor,
    # charge pipe, BOV, cooler) is intake-charge air
    if "exhaust" in lowered or "turbine_housing" in lowered or "wastegate" in lowered or "downpipe" in lowered \
            or "up_pipe" in lowered or "exhaust_duct" in lowered:
        return "exhaust"
    if "intake" in lowered or "compressor_housing" in lowered or "charge_" in lowered or "blow_off" in lowered:
        return "intake"
    return None


@dataclass
class SolidPart:
    """One real triangle-soup part (vertices/normals straight from the
    ported tube_mesh/cuboid_mesh generators -- the same geometry
    export_drivetrain_mesh writes to disk), tagged with the same real
    thermal-circuit membership WireframePart carries, for a live
    renderer that wants genuine shaded solid geometry instead of
    skeleton lines."""
    vertices: np.ndarray
    normals: np.ndarray
    thermal_group: str | None
    name: str
    # What this part is made of, as the graph DECLARED it. The material
    # table is otherwise matched from the part's name, which works for
    # engine hardware whose names the table was written around and does
    # not work at all for a machine (machines.py) whose parts are called
    # things like turret.shell -- every one of those fell through to the
    # translucent default, so a turret rendered as a set of grey ghosts.
    declared_material: str | None = None


# Real max butterfly-plate opening -- the SAME real constant engine_
# cycle_sim.py/throttle_body.py already use for the actual airflow
# physics (not re-derived independently; this just draws the same
# real angle their own model already computes).
BUTTERFLY_MAX_ANGLE_DEG = 78.0


def build_throttle_parts(graph: dict[str, Any], throttle_frac: float = 1.0) -> list["SolidPart"]:
    """A real rotating butterfly plate (plus its own external lever
    arm, sharing one shaft with it) for every real throttle-body/
    carburetor-barrel/individual-throttle-body node dressing.py already
    declares -- driven by throttle_frac alone, completely independent
    of crank angle, so this is meant to be baked as its OWN small set
    of frames (keyed by throttle position) and composited alongside
    whatever crank-angle frame is currently showing, not folded into
    the crank-angle animation's own per-cylinder frame count.

    Geometry-only: this reads flow_axis/plate_radius_m metadata dressing.
    py already attaches to the existing throttle-body node (no new graph
    node/edge -- the plate is a real sub-feature of hardware already
    there, not a separate part to wire up and risk adding more of the
    confusing extra edges this was built to clear up, not add to).

    The plate's face normal is `flow_axis` when closed (fully blocking)
    and rotates toward `flow_axis` itself as throttle_frac -> 1 (nearly
    parallel to the bore, i.e. nearly open) -- a real butterfly's own
    actual kinematics, not a stand-in animation."""
    theta = max(0.0, min(1.0, throttle_frac)) * math.radians(BUTTERFLY_MAX_ANGLE_DEG)
    parts: list[SolidPart] = []
    for node in graph["nodes"]:
        axis_raw = node.get("flow_axis")
        if axis_raw is None:
            continue
        center = np.array(node["reference_position"], dtype=np.float64)
        axis = _normalized3(np.array(axis_raw, dtype=np.float64))
        r = float(node.get("plate_radius_m", 0.02))
        shaft_dir, swept0 = _perpendicular_basis(axis)
        swept = swept0 * math.cos(theta) + axis * math.sin(theta)
        thickness_dir = _normalized3(np.cross(shaft_dir, swept))
        basis = np.array([shaft_dir, swept, thickness_dir])   # rows: local (u, v, w) -> world

        # every barrel of this one casting gets its own plate, side by
        # side along the crank axis at the declared pitch, all on one
        # shaft -- a 2- or 4-barrel is one unit with N bores, never N
        # separate throttle bodies; one external lever serves the shaft
        n_barrels = max(1, int(node.get("barrels", 1)))
        pitch = float(node.get("barrel_pitch_m", r * 2.2))
        local_v, local_n = cuboid_mesh((0.0, 0.0, 0.0), (r * 0.95, r * 0.95, r * 0.12))
        plate_vs, plate_ns = [], []
        for bi in range(n_barrels):
            offset = np.array([(bi - (n_barrels - 1) / 2.0) * pitch, 0.0, 0.0])
            plate_vs.append(center[None, :] + offset[None, :] + local_v @ basis)
            plate_ns.append(local_n @ basis)
        plate_v = np.concatenate(plate_vs)
        plate_n = np.concatenate(plate_ns)

        lever_start = center + shaft_dir * (r * 1.1)
        lever_end = lever_start + swept * (r * 1.2)
        lever_v, lever_n = tube_mesh(lever_start, lever_end, radius=r * 0.08, sides=6)

        vertices = np.concatenate([plate_v, lever_v])
        normals = np.concatenate([plate_n, lever_n])
        name = node["identity"].replace(".", "_").replace("/", "_")
        parts.append(SolidPart(vertices=vertices, normals=normals, thermal_group=None,
                               name=f"node_{name}_throttle_plate_linkage"))
    return parts


def build_drivetrain_solid_parts(graph: dict[str, Any], crank_angle_deg: float = 0.0, covers_off: bool = False) -> list[SolidPart]:
    """Every real line connection in the graph -- fluid lines (fuel,
    intake air, exhaust, coolant, oil, nitrous), torque shafts, cables,
    wiring harnesses, belts -- gets a real tube from the SAME ported
    tube_mesh generator build_drivetrain_mesh/export_drivetrain_mesh
    already use, not a cheaper stand-in: this is that generator's
    output, just also carrying each part's real thermal-circuit
    membership (WireframePart's own tagging) so a live renderer can
    shade/color it without re-deriving that from scratch."""
    positions = _node_position_lookup(graph["nodes"])
    chassis_side_ids = _chassis_side_ids(graph["nodes"])
    parts: list[SolidPart] = []

    for edge in graph["edges"]:
        if edge["a"] in chassis_side_ids or edge["b"] in chassis_side_ids:
            continue
        a = positions.get(edge["a"])
        b = positions.get(edge["b"])
        if a is None or b is None or np.allclose(a, b):
            continue
        if edge.get("section_shape") == "welded-i":
            vertices, normals = _welded_i_mesh(a, b, edge)
        elif edge.get("section_shape") == "flat-strap":
            vertices, normals = _flat_strap_mesh(a, b, edge)
        else:
            radius = float(edge.get("radius", DEFAULT_TUBE_RADIUS_M))
            if edge.get("fluid_route_class") == "trivial":
                radius *= 0.24
            sides = 6 if edge.get("routing") == "relaxed-multi-segment-harness" else 8
            vertices, normals = _edge_tube_mesh(a, b, edge, radius, sides)
        group = _thermal_group_for(edge.get("circuit_identity"), edge["identity"])
        name = edge["identity"].replace("/", "_").replace(".", "_")
        parts.append(SolidPart(vertices=vertices, normals=normals, thermal_group=group,
                               name=f"edge_{name}", declared_material=edge.get("material")))

    # the real cylinders themselves (cylinder_ports.py: bore wall, head
    # or open top, piston, valve chest / rack, and a stub for every real
    # port), built from the layout drivetrain_graph put on the document
    layout_data = graph.get("cylinder_layout")
    if layout_data:
        from cylinder_ports import deserialize_layout, build_parts_from_layout
        parts.extend(build_parts_from_layout(deserialize_layout(layout_data), crank_angle_deg=crank_angle_deg, covers_off=covers_off,
                                             chamber=_chamber_from_graph(graph)))

    for node in graph["nodes"]:
        if node.get("render_primitive") is False:
            continue
        if node["kind"] == "engine-block-port":
            # a real port is already drawn as a stub by the cylinder
            # layout above
            continue
        if node.get("drawn_by"):
            # a real part whose geometry a procedural mesh already
            # produces (crank_mesh's pulley/flywheel): the node exists
            # for identity, mass and picking -- drawing a box here too
            # would put a second body on top of the real one
            continue
        if node.get("chassis_side"):
            # hung from the body, not the engine (tank, electric pump,
            # muffler, tailpipe): real, in the graph, not in the engine's
            # own view -- the same cut EnginePackage makes for sourcing
            continue
        if layout_data and (node["identity"].startswith("powertrain.engine_block_body")
                            or node["identity"] == "powertrain.oil_pan"):
            # the generic block/pan boxes stand in only when there is no
            # real cylinder + crankcase geometry to draw
            continue
        center = np.array(node["reference_position"], dtype=np.float64)
        fan_radius = node.get("fan_disk_radius_m")
        drum_axis = node.get("drum_axis")
        # A DECLARED SHAPE wins over everything below. The rules that
        # follow were written for engine parts and infer a shape from
        # which attributes happen to be present, which cannot express
        # "this is a ring" at all and quietly drew every machine part as
        # a box. A part that knows what shape it is should just say so.
        shape = node.get("shape")
        if shape == "plate-cell":
            vertices, normals = _plate_cell_mesh(node)
            group = _thermal_group_for(None, node["identity"])
            name = node["identity"].replace("/", "_").replace(".", "_")
            parts.append(SolidPart(
                vertices=vertices, normals=normals, thermal_group=group,
                name=f"node_{name}",
                declared_material=node.get("material")))
            continue
        if (node.get("of_engine") == "agt1500-abrams-turbine"
                and str(shape).startswith("agt1500-")):
            vertices, normals = _agt1500_module_geometry(node)
            name = node["identity"].replace("/", "_").replace(".", "_")
            parts.append(SolidPart(
                vertices=vertices, normals=normals,
                thermal_group=node.get("thermal_group"),
                name=f"node_{name}",
                declared_material=node.get("material")))
            continue
        if shape == "ring":
            from mesh_primitives import ring_mesh
            ax = np.array(node.get("ring_axis", (0.0, 1.0, 0.0)), dtype=np.float64)
            vertices, normals = ring_mesh(
                center, ax, float(node.get("ring_outer_radius_m", 0.5)),
                float(node.get("ring_inner_radius_m", 0.3)),
                float(node.get("ring_thickness_m", 0.06)))
            group = _thermal_group_for(None, node["identity"])
            name = node["identity"].replace("/", "_").replace(".", "_")
            parts.append(SolidPart(vertices=vertices, normals=normals, thermal_group=group,
                                   name=f"node_{name}", declared_material=node.get("material")))
            continue
        if shape == "annulus":
            # A DISC WITH A BORE, which mesh_primitives has had all
            # along -- `ring_mesh` is even documented as "the shape a
            # slew bearing actually is", complete with the note that
            # drawing one as a box gives it no hole for a round to pass
            # through. Nothing was wired to it, so every annulus in this
            # project drew as a box or as a ring of little boxes.
            from mesh_primitives import ring_mesh
            ax = np.array(node.get("annulus_axis", (0.0, 1.0, 0.0)),
                          dtype=np.float64)
            vertices, normals = ring_mesh(
                center, ax,
                float(node.get("outer_radius_m", 0.5)),
                float(node.get("inner_radius_m", 0.2)),
                float(node.get("thickness_m", 0.04)),
                segments=int(node.get("draw_segments", 48)))
            group = _thermal_group_for(None, node["identity"])
            name = node["identity"].replace("/", "_").replace(".", "_")
            parts.append(SolidPart(vertices=vertices, normals=normals,
                                   thermal_group=group, name=f"node_{name}",
                                   declared_material=node.get("material")))
            continue
        if shape == "drum" and drum_axis is not None:
            from mesh_primitives import capped_tube_mesh
            ax = np.array(drum_axis, dtype=np.float64)
            ax = ax / max(np.linalg.norm(ax), 1e-9)
            half = float(node.get("drum_length_m", 0.10)) / 2.0
            vertices, normals = capped_tube_mesh(center - ax * half, center + ax * half,
                                                 float(node.get("drum_radius_m", 0.05)), sides=16)
            group = _thermal_group_for(None, node["identity"])
            name = node["identity"].replace("/", "_").replace(".", "_")
            parts.append(SolidPart(vertices=vertices, normals=normals, thermal_group=group,
                                   name=f"node_{name}", declared_material=node.get("material")))
            continue
        if drum_axis is not None and node.get("body_half_extent_m") is None:
            # a real drum-shaped part (air cleaner, spin-on filter,
            # distributor, coil, dry-sump tank) along its own axis
            from mesh_primitives import capped_tube_mesh, beveled_drum_mesh
            ax = np.array(drum_axis, dtype=np.float64); ax = ax / max(np.linalg.norm(ax), 1e-9)
            half = float(node.get("drum_length_m", 0.05)) / 2.0
            bevel = node.get("drum_bevel_m")
            if bevel:
                # a can with bevelled corners (cat / muffler shell): conical
                # transitions from the pipe radius up to the body
                vertices, normals = beveled_drum_mesh(center - ax * half, center + ax * half, float(node.get("drum_radius_m", 0.03)),
                                                      float(bevel), float(node.get("drum_end_radius_m", 0.02)), sides=18)
            else:
                vertices, normals = capped_tube_mesh(center - ax * half, center + ax * half, float(node.get("drum_radius_m", 0.03)), sides=18)
        elif fan_radius is not None:
            # A real circular fan blade sweep -- a short, wide tube
            # along the crank's own rotation axis (X), tube_mesh's own
            # cross-section (perpendicular to that axis) genuinely IS a
            # disc, rather than the generic axis-aligned cube every
            # other small node gets.
            half_depth = 0.012
            vertices, normals = tube_mesh(center - np.array([half_depth, 0.0, 0.0]),
                                          center + np.array([half_depth, 0.0, 0.0]),
                                          float(fan_radius), sides=16)
        else:
            half_extent = _body_half_extent(node)
            vertices, normals = cuboid_mesh(center, half_extent)
        group = _thermal_group_for(None, node["identity"])
        name = node["identity"].replace("/", "_").replace(".", "_")
        parts.append(SolidPart(vertices=vertices, normals=normals, thermal_group=group,
                               name=f"node_{name}", declared_material=node.get("material")))

    parts.extend(build_intake_bung_parts(graph))
    parts.extend(build_agt1500_moving_parts(graph, crank_angle_deg))
    return parts


def build_intake_bung_parts(graph: dict[str, Any]) -> list[SolidPart]:
    """Default-plugged bungs dressing.py declares on the intake hardware
    (MAP/vacuum taps, IAT bosses, injector bungs) drawn as the SAME short
    stub every other casting port gets (head_mesh.build_head_parts'
    oil-port stubs -- same depth, same `port_` naming, so they land in
    the casting-port material). A plugged bung is drawn a touch shorter
    than an open, plumbed one: a plug, not a fitting."""
    parts: list[SolidPart] = []
    for node in graph["nodes"]:
        if node.get("kind") != "engine-block-port" or not node.get("bung"):
            continue
        pos = np.array(node["reference_position"], dtype=np.float64)
        d = _normalized3(np.array(node.get("port_direction", [0.0, 1.0, 0.0]), dtype=np.float64))
        radius = float(node.get("port_radius_m", 0.004))
        depth = 0.010 if node.get("plugged", True) else 0.018
        vertices, normals = tube_mesh(pos - d * 0.004, pos + d * depth, radius, sides=10)
        name = node["identity"].replace("powertrain.", "").replace("/", "_").replace(".", "_")
        parts.append(SolidPart(vertices=vertices, normals=normals, thermal_group=None, name=f"port_{name}"))
    return parts


def build_drivetrain_wireframe(graph: dict[str, Any]) -> list[WireframePart]:
    """The cheap wireframe cousin of build_drivetrain_mesh, each part
    tagged with the real thermal circuit it belongs to (or None, for
    the many real parts -- torque shafts, mounts, wiring, rotating
    masses -- this sim has no simulated temperature for at all)."""
    positions = _node_position_lookup(graph["nodes"])
    parts: list[WireframePart] = []

    for edge in graph["edges"]:
        a = positions.get(edge["a"])
        b = positions.get(edge["b"])
        if a is None or b is None or np.allclose(a, b):
            continue
        radius = float(edge.get("radius", DEFAULT_TUBE_RADIUS_M))
        if edge.get("fluid_route_class") == "trivial":
            radius *= 0.24
        sides = 6 if edge.get("routing") == "relaxed-multi-segment-harness" else 8
        segments = []
        points = _edge_points(a, b, edge)
        for p, q in zip(points, points[1:]):
            if not np.allclose(p, q):
                segments.extend(tube_wireframe(p, q, radius, sides=sides))
        group = _thermal_group_for(edge.get("circuit_identity"), edge["identity"])
        name = edge["identity"].replace("/", "_").replace(".", "_")
        parts.append(WireframePart(segments=segments, thermal_group=group, name=f"edge_{name}"))

    for node in graph["nodes"]:
        if node.get("render_primitive") is False:
            continue
        if node["kind"] == "powertrain-mount" or node["kind"] == "engine-block-port":
            continue
        center = np.array(node["reference_position"], dtype=np.float64)
        if node.get("shape") == "plate-cell":
            segments = _plate_cell_wireframe(node)
        else:
            half_extent = _body_half_extent(node)
            segments = cuboid_wireframe(center, half_extent)
        group = _thermal_group_for(None, node["identity"])
        name = node["identity"].replace("/", "_").replace(".", "_")
        parts.append(WireframePart(segments=segments, thermal_group=group, name=f"node_{name}"))

    return parts


def export_drivetrain_mesh(graph: dict[str, Any], out_path: str) -> int:
    """Builds the mesh and writes it to a real .obj file for direct
    inspection in any 3D tool. Returns the total triangle-soup vertex
    count written."""
    parts = build_drivetrain_mesh(graph)
    write_obj(out_path, parts)
    return sum(len(vertices) for vertices, _normals, _name in parts)
