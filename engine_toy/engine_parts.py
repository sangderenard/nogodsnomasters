"""The parts every real crank engine carries that the graph did not
declare -- emitted from data the engine ALREADY declares, never from a
fixed per-engine list:

  exhaust downstream   ExhaustSystem.segments (the same ordered cat/
                       muffler/tailpipe chain the acoustic model reads),
                       extended from each real collector along its own
                       real outlet direction
  crank-end hardware   the damper/pulley and the flywheel + ring gear,
                       declared exactly where crank_mesh already DRAWS
                       them (crank_end_fittings) -- nodes for parts that
                       had geometry but no identity
  starter hardware     the motor body/solenoid, recoil drum, hand crank,
                       or air motor, chosen by the real STARTING_SYSTEMS
                       `engages`/`drive` strings production already puts
                       on powertrain.starter_drive
  belt drive           a pulley on every belt-driven accessory the
                       accessory ring already has, plus a tensioner/idler
  timing drive         cover + chain/belt run between crank nose and cam
  coolant plumbing     expansion bottle, heater core and its hoses, on
                       liquid-cooled engines only
  emissions            EGR valve + tube (EGRSystem.has_egr), the PCV
                       valve on the existing PCV port
  bellhousing          the case spanning crank rear -> clutch -> gearbox

Everything here is a real bolt-on with a real mounting point relative
to an existing node. Nothing invents an engine feature: a part appears
only when the Engine declares the thing it belongs to.
"""
from __future__ import annotations

import numpy as np

import engine_geometry
import duct_routing
from crank_mesh import crank_end_fittings, CRANK_AXIS

UP = np.array([0.0, 1.0, 0.0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _pos(n: dict) -> np.ndarray:
    return np.array(n["reference_position"], dtype=np.float64)


# ---------------------------------------------------------------------
# exhaust: collector -> (cat) -> (muffler) -> tailpipe
# ---------------------------------------------------------------------

def _emit_exhaust_downstream(engine, nodes, edges, node, edge) -> None:
    segments = [s for s in engine.exhaust_system.segments if s.kind not in ("primary", "collector")]
    if not segments:
        return   # open headers: the collector IS the outlet
    collectors = [n for n in nodes if n["identity"].startswith("powertrain.exhaust_collector_")
                  and n.get("kind") == "exhaust-collector"]
    for coll in collectors:
        start = _pos(coll)
        d = _unit(coll.get("outlet_direction", [0.0, -1.0, 0.0]))
        # a real system turns rearward under the car after the drop; a
        # collector already pointing along the crank axis just runs on
        rear = CRANK_AXIS if abs(float(np.dot(d, CRANK_AXIS))) < 0.9 else d
        prev_id = coll["identity"]
        # a short drop along the collector's own outlet direction to
        # clear the block, then every real segment runs rearward at
        # that level -- a real system does not hang a 25 cm cat
        # straight down off the header
        r0 = segments[0].diameter_mm / 2000.0
        cursor = start + d * (r0 * 3.0)
        run_dir = rear
        for seg in segments:
            r = seg.diameter_mm / 2000.0
            a = cursor.copy()
            b = a + run_dir * seg.length_m
            sid = f"{coll['identity']}.{seg.kind.replace('-', '_')}"
            # the body sits at the segment's own midpoint; cat/muffler are
            # real cans wider than the pipe, the tailpipe is just pipe
            mid = (a + b) / 2.0
            can = seg.kind in ("catalytic-converter", "muffler")
            # the cat is close-coupled -- it hangs off the engine's own
            # downpipe and ships with a crate; the muffler and tailpipe are
            # CHASSIS plumbing (hung from the body, cut to the vehicle),
            # exactly as a fuel tank is: real, in the graph for the exhaust
            # circuit and the acoustic model, but supplied elsewhere in
            # crate terms and not part of the engine's own view
            chassis_side = seg.kind in ("muffler", "tailpipe")
            node(sid, [float(v) for v in mid], "exhaust-component", segment_kind=seg.kind,
                 length_m=seg.length_m, diameter_mm=seg.diameter_mm, restriction=seg.restriction,
                 chassis_side=chassis_side,
                 mass_kg=(4.0 if seg.kind == "muffler" else 3.0 if can else 1.2),
                 # no body_half_extent_m: vehicle_mesh draws a box whenever
                 # one is present, and a box is exactly what made the cat
                 # read as a mystery reservoir at the end of the header --
                 # every segment is a drum ON the pipe's own axis -- a cat or
                 # muffler is a fat round can, not a rectangular box that reads
                 # as some unrelated reservoir sitting at the end of the header
                 drum_axis=[float(v) for v in run_dir],
                 drum_radius_m=r * (2.2 if can else 1.0), drum_length_m=seg.length_m,
                 # a can's corners are bevelled: conical inlet/outlet cones
                 # from the pipe bore up to the shell, as on a real cat/muffler
                 drum_bevel_m=(min(seg.length_m * 0.2, r * 2.5) if can else None),
                 drum_end_radius_m=(r if can else None))
            edge(f"{sid}.inlet", prev_id, sid, "exhaust-flow-path", radius=r, circuit_identity="exhaust",
                 medium_rate_state="exhaust-pulse-pressure-and-temperature")
            prev_id = sid
            cursor = b
        # the production collector -> exhaust_manifold junction edge is left
        # exactly as it is: the exhaust fluid circuit, the turbo's heat path
        # and the acoustic model all key on that junction. The real chain
        # above hangs off the same collector; its edges carry no flow
        # capacity of their own, so the circuit's summed capacity is
        # unchanged -- the chain adds real parts, not a second flow path.
        # an O2 sensor bung just downstream of the collector, plugged
        node(f"{coll['identity']}.o2_bung", [float(v) for v in (start + d * (segments[0].diameter_mm / 2000.0 * 1.5))],
             "engine-block-port", port_kind="sensor-boss", port_direction=[float(v) for v in _unit(np.cross(d, CRANK_AXIS) if abs(np.dot(d, CRANK_AXIS)) < 0.9 else UP)],
             port_radius_m=0.006, fluid_role="exhaust", mating=False, connected=False, plugged=True,
             part=coll["identity"], bung=True)


# ---------------------------------------------------------------------
# crank ends: damper/pulley, flywheel + ring gear, starter hardware
# ---------------------------------------------------------------------

def _emit_crank_end_hardware(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    fit = crank_end_fittings(layout)
    if fit is None:
        return
    crank_id = "powertrain.engine"
    if "pulley_centre" in fit:
        node("powertrain.harmonic_balancer", fit["pulley_centre"], "rotating-mass",
             mass_kg=engine.mass_kg * 0.012, inertia_kg_m2=0.5 * engine.mass_kg * 0.012 * fit["pulley_radius_m"] ** 2,
             radius_m=fit["pulley_radius_m"], drawn_by="crank_mesh:crank_pulley",
             body_half_extent_m=[fit["pulley_half_len_m"], fit["pulley_radius_m"], fit["pulley_radius_m"]])
        edge("powertrain.crank_to_balancer", crank_id, "powertrain.harmonic_balancer", "rigid-keyed-hub", radius=0.01)
    else:
        node("powertrain.flywheel_front", fit["front_flywheel_centre"], "rotating-mass",
             mass_kg=engine.mass_kg * 0.08, radius_m=fit["front_flywheel_radius_m"], drawn_by="crank_mesh:flywheel_front",
             body_half_extent_m=[fit["bore_m"] * 0.12, fit["front_flywheel_radius_m"], fit["front_flywheel_radius_m"]])
        edge("powertrain.crank_to_front_flywheel", crank_id, "powertrain.flywheel_front", "rigid-keyed-hub", radius=0.01)
    node("powertrain.flywheel", fit["flywheel_centre"], "rotating-mass",
         mass_kg=engine.mass_kg * 0.06, radius_m=fit["flywheel_radius_m"], drawn_by="crank_mesh:flywheel",
         body_half_extent_m=[fit["flywheel_half_len_m"], fit["flywheel_radius_m"], fit["flywheel_radius_m"]])
    edge("powertrain.crank_to_flywheel", crank_id, "powertrain.flywheel", "rigid-keyed-hub", radius=0.012)

    starter = node_by_id.get("powertrain.starter_drive")
    if starter is None:
        return
    engages = starter.get("engages", "")
    drive = starter.get("drive", "")
    sp = _pos(starter)
    if engages == "flywheel-ring-gear":
        # the ring gear on the flywheel rim, and the motor body it meshes
        # with: a series DC motor + bendix/solenoid, or a pneumatic vane
        # motor -- both sit beside the flywheel, pinion toward the rim
        fw = np.array(fit["flywheel_centre"])
        node("powertrain.flywheel.ring_gear", fit["flywheel_centre"], "rotating-mass", mass_kg=1.2,
             radius_m=fit["flywheel_radius_m"] * 1.02, drawn_by="crank_mesh:flywheel",
             body_half_extent_m=[fit["bore_m"] * 0.03, fit["flywheel_radius_m"] * 1.02, fit["flywheel_radius_m"] * 1.02])
        edge("powertrain.flywheel_to_ring_gear", "powertrain.flywheel", "powertrain.flywheel.ring_gear", "rigid-keyed-hub", radius=0.008)
        # the motor bolts to the bellhousing flange and lies OUTSIDE the
        # case, alongside it, low on the flank, its pinion reaching in
        # through the flange to the ring gear -- not inside the crankcase
        # envelope where a point 1.25 flywheel radii from the flywheel
        # centre landed it
        half = engine_geometry.block_half_yz_m(engine)
        pneumatic = "pneumatic" in drive
        r_body = fit["bore_m"] * 0.42
        side = -1.0 if sp[2] < 0 else 1.0
        body_c = np.array([fw[0] - fit["bore_m"] * 0.55, fw[1] - half * 0.35, side * (half * 1.15 + r_body)])
        node("powertrain.starter_motor", [float(v) for v in body_c], "engine-block-component",
             mass_kg=(6.0 if pneumatic else 4.5), starter_kind=("air-motor" if pneumatic else "series-dc-motor"),
             drum_axis=[float(v) for v in CRANK_AXIS], drum_radius_m=fit["bore_m"] * 0.42, drum_length_m=fit["bore_m"] * 1.1)
        if not pneumatic:
            node("powertrain.starter_motor.solenoid", [float(v) for v in (body_c + UP * fit["bore_m"] * 0.45)],
                 "engine-block-component", mass_kg=0.6,
                 drum_axis=[float(v) for v in CRANK_AXIS], drum_radius_m=fit["bore_m"] * 0.18, drum_length_m=fit["bore_m"] * 0.5)
            edge("powertrain.starter_solenoid_to_motor", "powertrain.starter_motor.solenoid", "powertrain.starter_motor",
                 "rigid-bolted-joint", radius=0.006)
        edge("powertrain.starter_motor_to_drive", "powertrain.starter_motor", "powertrain.starter_drive",
             "rigid-bolted-joint", radius=0.006)
        edge("powertrain.starter_pinion_to_ring_gear", "powertrain.starter_drive", "powertrain.flywheel.ring_gear",
             "starter-pinion-mesh", radius=0.004, engagement="overrunning-when-caught")
    elif engages == "crank-nose-ratchet-drum":
        # recoil: a rope drum with pawls on the nose, under a cup
        nose = np.array(fit["nose_end"])
        node("powertrain.recoil_starter", [float(v) for v in (nose - CRANK_AXIS * fit["bore_m"] * 0.3)], "engine-block-component",
             mass_kg=0.5, drum_axis=[float(v) for v in CRANK_AXIS], drum_radius_m=fit["bore_m"] * 0.9, drum_length_m=fit["bore_m"] * 0.4)
        edge("powertrain.recoil_to_crank_nose", "powertrain.recoil_starter", "powertrain.starter_drive", "rigid-bolted-joint", radius=0.005)
    elif engages in ("crank-nose-dog", "crank-nose-hex", "crank-nose-clutch-face"):
        # hand crank / external cart / inertia starter dog on the nose
        nose = np.array(fit["nose_end"])
        kind = {"crank-nose-dog": "hand-crank-dog", "crank-nose-hex": "starter-cart-hex",
                "crank-nose-clutch-face": "inertia-starter-clutch"}[engages]
        node("powertrain.crank_nose_fitting", [float(v) for v in (nose - CRANK_AXIS * fit["bore_m"] * 0.12)], "engine-block-component",
             mass_kg=0.4, fitting_kind=kind,
             drum_axis=[float(v) for v in CRANK_AXIS], drum_radius_m=fit["r_main_m"] * 1.4, drum_length_m=fit["bore_m"] * 0.2)
        edge("powertrain.nose_fitting_to_drive", "powertrain.crank_nose_fitting", "powertrain.starter_drive", "rigid-bolted-joint", radius=0.005)
    # flywheel-bar / air-start engage the flywheel rim or the heads directly:
    # the flywheel node above IS the part; nothing more to declare


# ---------------------------------------------------------------------
# belt drive: a pulley per accessory on the ring, a tensioner/idler
# ---------------------------------------------------------------------

def _block_front_face_x(engine, nodes) -> float:
    """The real front face of the block casting: the forward-most face
    of the block segments the graph already declares with their own
    half-extents -- not crank_x_min plus a fixed offset, which lands
    inside the casting on a long block and floats ahead of it on a
    short one."""
    faces = [n["reference_position"][0] - float(n["body_half_extent_m"][0])
             for n in nodes if n["identity"].startswith("powertrain.engine_block_body") and n.get("body_half_extent_m")]
    if faces:
        return min(faces)
    x_min, _ = engine_geometry.crank_extent(engine_geometry.cylinder_sites(engine))
    return x_min - 0.03


def _emit_belt_drive(engine, nodes, edges, node, edge, node_by_id) -> None:
    ring = {"alternator": "electrical.alternator", "water_pump": "powertrain.water_pump",
            "fan": "powertrain.cooling_fan", "ac_compressor": "ac_compressor",
            "pneumatic_compressor": "pneumatic_compressor"}
    points = engine_geometry.accessory_mount_points(engine)
    present = [(name, node_by_id[ring[name]]) for name in points if name in ring and ring[name] in node_by_id]
    if not present:
        return
    # pulleys hang in the belt plane just ahead of the timing cover, which
    # itself sits flush on the block's real front face
    face_x = _block_front_face_x(engine, nodes) - 0.024 - 0.03
    radius = engine_geometry.block_half_yz_m(engine) * 0.28
    for name, acc in present:
        p = _pos(acc)
        pid = f"{acc['identity']}.pulley"
        node(pid, [face_x - 0.015, float(p[1]), float(p[2])], "rotating-mass", mass_kg=0.6,
             drum_axis=[1.0, 0.0, 0.0], drum_radius_m=radius, drum_length_m=0.025)
        edge(f"{pid}.hub", acc["identity"], pid, "rigid-keyed-hub", radius=0.006)
    if len(present) >= 2:
        # a spring tensioner/idler between the first two ring members, on
        # the slack side -- a real serpentine drive has one
        a = _pos(present[0][1]); b = _pos(present[1][1])
        mid = (a + b) / 2.0
        outward = _unit(np.array([0.0, mid[1], mid[2]]))
        tp = np.array([face_x - 0.015, mid[1], mid[2]]) + outward * radius * 1.4
        node("powertrain.belt_tensioner", [float(v) for v in tp], "rotating-mass", mass_kg=0.5,
             drum_axis=[1.0, 0.0, 0.0], drum_radius_m=radius * 0.6, drum_length_m=0.022)
        edge("powertrain.belt_tensioner_to_block", "powertrain.belt_tensioner", "powertrain.engine", "rigid-bolted-joint", radius=0.006)


# ---------------------------------------------------------------------
# timing drive: cover + chain/belt run
# ---------------------------------------------------------------------

def _block_rear_face_x(engine, nodes) -> float:
    """The flywheel-end face of the block casting -- _block_front_face_x's
    mirror, for the engines whose timing drive lives back there."""
    faces = [n["reference_position"][0] + float(n["body_half_extent_m"][0])
             for n in nodes if n["identity"].startswith("powertrain.engine_block_body") and n.get("body_half_extent_m")]
    if faces:
        return max(faces)
    _, x_max = engine_geometry.crank_extent(engine_geometry.cylinder_sites(engine))
    return x_max + 0.03


def camshaft_ids(engine, node_by_id) -> list[str]:
    """Every camshaft node this engine actually has, in order. One is the
    ordinary case; a SOHC vee engine has one per bank and a DOHC one has
    two per bank (drivetrain_graph.py builds the extras from the
    architecture's declared camshaft_count)."""
    ids = ["powertrain.camshaft"] + [f"powertrain.camshaft_{i}" for i in range(2, 9)]
    return [i for i in ids if i in node_by_id]


# Real relative torsional stiffness of the three things that actually
# drive a camshaft, as multiples of a roller chain's. A toothed rubber
# belt is markedly more compliant than a chain (its tensile cords
# stretch where a chain's plates do not) and a gear train is stiffer
# than either. These are disclosed ratios chosen to be the right
# ORDER, not measured figures for any specific drive.
TIMING_DRIVE_STIFFNESS_FACTOR = {"chain": 1.0, "belt": 0.35, "gear": 3.0}


def _emit_timing_drive(engine, nodes, edges, node, edge, node_by_id) -> None:
    cams = camshaft_ids(engine, node_by_id)
    if not cams:
        return
    arch = engine.architecture
    medium = str(getattr(arch, "timing_drive", "chain") or "chain")
    at_rear = str(getattr(arch, "timing_drive_at", "front") or "front") == "rear"
    # which end of the block the drive actually lives on, and which way
    # the cover and sprockets stand off from that face
    face_x = _block_rear_face_x(engine, nodes) if at_rear else _block_front_face_x(engine, nodes)
    out = 1.0 if at_rear else -1.0
    half = engine_geometry.block_half_yz_m(engine)
    cam_ps = [_pos(node_by_id[c]) for c in cams]
    reach_y = max(abs(p[1]) for p in cam_ps)
    cover_c = np.array([face_x + out * 0.012, float(np.mean([p[1] for p in cam_ps])) * 0.5, 0.0])
    node("powertrain.timing_cover", [float(v) for v in cover_c], "engine-block-component", mass_kg=1.5,
         body_half_extent_m=[0.012, max(reach_y * 0.5 + half * 0.3, half * 0.6), half * 0.7])
    edge("powertrain.timing_cover_to_block", "powertrain.timing_cover", "powertrain.engine", "rigid-bolted-joint", radius=0.006)
    sx = face_x + out * 0.006
    node("powertrain.timing_drive.crank_sprocket", [float(sx), 0.0, 0.0], "rotating-mass", mass_kg=0.3,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=half * 0.22, drum_length_m=0.012)
    edge("powertrain.timing_crank_sprocket_hub", "powertrain.engine", "powertrain.timing_drive.crank_sprocket", "rigid-keyed-hub", radius=0.006)
    # one sprocket and one run per camshaft, off the single crank
    # sprocket -- which is how a multi-cam engine is really driven,
    # whether that is one belt wrapping every cam pulley or a chain per
    # bank
    for i, (cam_id, cam_p) in enumerate(zip(cams, cam_ps), start=1):
        suffix = "" if i == 1 else f"_{i}"
        sprocket = f"powertrain.timing_drive.cam_sprocket{suffix}"
        node(sprocket, [float(sx), float(cam_p[1]), float(cam_p[2])], "rotating-mass", mass_kg=0.4,
             drum_axis=[1.0, 0.0, 0.0], drum_radius_m=half * 0.44, drum_length_m=0.012)
        edge(f"powertrain.timing_cam_sprocket_hub{suffix}", cam_id, sprocket, "rigid-keyed-hub", radius=0.006)
        edge(f"powertrain.timing_{medium}_run{suffix}", "powertrain.timing_drive.crank_sprocket", sprocket,
             "cosmetic-belt-wrap", radius=0.004)


# ---------------------------------------------------------------------
# coolant plumbing, emissions, bellhousing
# ---------------------------------------------------------------------

def _emit_coolant_plumbing(engine, nodes, edges, node, edge, node_by_id) -> None:
    radiator = node_by_id.get("powertrain.radiator")
    thermostat = node_by_id.get("powertrain.coolant_thermostat")
    if radiator is None or thermostat is None:
        return
    rp = _pos(radiator); tp = _pos(thermostat)
    half = engine_geometry.block_half_yz_m(engine)
    bottle = rp + np.array([0.0, half * 1.2, half * 1.6])
    # the bottle sits on the radiator support and the heater core in the
    # dash: both BODY-mounted -- real, plumbed, in the graph, but chassis
    # side like the tank, not part of the engine's own view or crate
    node("powertrain.coolant_expansion_bottle", [float(v) for v in bottle], "engine-block-component", mass_kg=0.8,
         fluid_volume_l=1.5, body_half_extent_m=[half * 0.35, half * 0.5, half * 0.35], chassis_side=True)
    edge("powertrain.expansion_bottle_overflow", "powertrain.radiator", "powertrain.coolant_expansion_bottle", "coolant-line",
         radius=0.005, circuit_identity="coolant", medium_rate_state="coolant-flow-and-temperature")
    # heater core: a small exchanger tapped off the head outlet, returning
    # to the pump suction -- present on any liquid-cooled road engine
    core = tp + np.array([0.0, half * 0.6, -half * 2.2])
    node("powertrain.heater_core", [float(v) for v in core], "engine-block-component", mass_kg=1.2,
         exchanger_kind="fin-and-tube-heater-core", body_half_extent_m=[half * 0.5, half * 0.35, half * 0.12], chassis_side=True)
    edge("powertrain.heater_supply_hose", "powertrain.coolant_thermostat", "powertrain.heater_core", "coolant-line",
         radius=0.008, circuit_identity="coolant", medium_rate_state="coolant-flow-and-temperature")
    pump = node_by_id.get("powertrain.water_pump")
    if pump is not None:
        edge("powertrain.heater_return_hose", "powertrain.heater_core", "powertrain.water_pump", "coolant-line",
             radius=0.008, circuit_identity="coolant", medium_rate_state="coolant-flow-and-temperature")


def _emit_compression_brake(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """The compression-release brake as real parts: one slave-piston
    actuator over every exhaust valve whose port declares
    `actuation="compression-release"` (cylinder_ports.PortSpec), and one
    housing per bank spanning them -- the Jacobs-style brake housing
    that bolts over the exhaust rockers under the valve cover."""
    if not getattr(engine, "compression_release_brake", False) or not layout:
        return
    by_bank: dict[int, list[tuple[int, np.ndarray, np.ndarray, float]]] = {}
    for g, ports in layout:
        axis = _unit(g.axis)
        for p in ports:
            if getattr(p, "actuation", "cam") != "compression-release":
                continue
            pos = np.asarray(p.position, dtype=np.float64)
            # bank by the sign of the bore axis's z (a V's two banks); an
            # inline engine's single bank gets key 0
            key = 0 if abs(axis[2]) < 0.3 else (1 if axis[2] > 0 else -1)
            by_bank.setdefault(key, []).append((g.number, pos, axis, g.bore_m))
    for key, items in by_bank.items():
        ids = []
        for number, pos, axis, bore in items:
            aid = f"powertrain.compression_brake.cyl_{number}"
            apos = pos + axis * (bore * 0.42)
            node(aid, [float(v) for v in apos], "engine-block-component", mass_kg=0.6,
                 drum_axis=[float(v) for v in axis], drum_radius_m=bore * 0.09, drum_length_m=bore * 0.22,
                 actuation="compression-release", cylinder=number)
            edge(f"{aid}.slave_to_valve", aid, f"powertrain.cylinder_{number}.exhaust_port", "rigid-bolted-joint", radius=0.004)
            ids.append((aid, apos, axis, bore))
        centre = sum(a for _, a, _, _ in ids) / len(ids)
        axis = ids[0][2]; bore = ids[0][3]
        xs = [a[0] for _, a, _, _ in ids]
        hid = f"powertrain.compression_brake_housing_bank_{key}"
        hpos = centre + axis * (bore * 0.18)
        node(hid, [float(v) for v in hpos], "engine-block-component", mass_kg=4.0 + 0.6 * len(ids),
             body_half_extent_m=[max(0.05, (max(xs) - min(xs)) / 2.0 + bore * 0.35), bore * 0.12, bore * 0.30],
             brake_kind="compression-release")
        for aid, _, _, _ in ids:
            edge(f"{aid}.housing", aid, hid, "rigid-bolted-joint", radius=0.004)
        edge(f"{hid}.oil_feed", "powertrain.oil_pump", hid, "oil-line", radius=0.005, circuit_identity="oil",
             medium_rate_state="oil-pressure-and-flow") if "powertrain.oil_pump" in node_by_id else None


def _emit_pneumatic_idle_assist(engine, nodes, edges, node, edge, node_by_id) -> None:
    """A real, declared second consumer on the same reserve tank/
    receiver the compressor (and, on some engines, the air-motor
    starter) already uses: a fixed-diameter port straight into the
    intake plenum. Only the real graph connection lives here -- the
    actual dump decision (trip rpm, the live on/off toggle) is a
    runtime call engine_cycle_sim.py makes every tick, same as the
    nitrous/WMI solenoids; this function only wires the real hardware
    the graph needs for that call to have somewhere to send the air."""
    pn = engine.pneumatics
    if not (pn.compressor_fitted and pn.idle_assist_fitted):
        return
    plenum = node_by_id.get("powertrain.intake_plenum")
    tank = node_by_id.get("pneumatic_reserve_tank")
    if plenum is None or tank is None:
        return
    pp = _pos(plenum)
    r = max(0.001, float(pn.idle_assist_port_diameter_mm) / 2000.0)
    port_pos = pp + np.array([0.0, -0.02, r * 3.0])
    node("powertrain.intake_plenum.compressed_air_port", [float(v) for v in port_pos], "engine-block-port",
         port_kind="compressed-air-dump", bung=True, plugged=False,
         port_direction=[0.0, -1.0, 0.0], port_radius_m=r)
    # off the protected side's isolation valve when the brake system is
    # fitted (real PBS plumbing), straight off the wet tank otherwise
    feed_id = "pneumatic_isolation_valve" if "pneumatic_isolation_valve" in node_by_id else "pneumatic_reserve_tank"
    edge("powertrain.pneumatic_idle_assist_line", feed_id,
         "powertrain.intake_plenum.compressed_air_port", "compressed-air-line",
         radius=r, circuit_identity="pneumatic-reserve", medium_rate_state="pneumatic-reserve-pressure-and-fill")
    edge("powertrain.pneumatic_idle_assist_port_seat", "powertrain.intake_plenum.compressed_air_port",
         "powertrain.intake_plenum", "rigid-bolted-joint", radius=r)


def _emit_emissions(engine, nodes, edges, node, edge, node_by_id) -> None:
    pcv = node_by_id.get("powertrain.engine_block_port.pcv")
    if pcv is not None:
        p = _pos(pcv)
        node("powertrain.pcv_valve", [float(v) for v in (p + UP * 0.02)], "engine-block-component", mass_kg=0.1,
             drum_axis=[0.0, 1.0, 0.0], drum_radius_m=0.012, drum_length_m=0.035)
        edge("powertrain.pcv_valve_seat", "powertrain.engine_block_port.pcv", "powertrain.pcv_valve", "rigid-bolted-joint", radius=0.004)
    egr = getattr(engine, "egr", None)
    if egr is not None and getattr(egr, "has_egr", False):
        man = node_by_id.get("powertrain.exhaust_manifold"); plen = node_by_id.get("powertrain.intake_plenum")
        if man is not None and plen is not None:
            mp_, pp_ = _pos(man), _pos(plen)
            v = (mp_ + pp_) / 2.0 + UP * 0.03
            node("powertrain.egr_valve", [float(x) for x in v], "engine-block-component", mass_kg=0.9,
                 max_egr_frac=float(getattr(egr, "max_egr_frac", 0.0)), body_half_extent_m=[0.03, 0.025, 0.03])
            edge("powertrain.egr_tube", "powertrain.exhaust_manifold", "powertrain.egr_valve", "exhaust-flow-path", radius=0.006,
                 circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature")
            edge("powertrain.egr_to_plenum", "powertrain.egr_valve", "powertrain.intake_plenum", "low-pressure-air-line", radius=0.006,
                 circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")


def _emit_bellhousing(engine, nodes, edges, node, edge, node_by_id) -> None:
    rear = node_by_id.get("powertrain.crank_shaft.rear"); clutch = node_by_id.get("powertrain.clutch")
    # a transverse install's transaxle carries its own integral clutch
    # housing -- the bellhousing spans to it instead of a gearbox
    trans = node_by_id.get("powertrain.transmission") or node_by_id.get("powertrain.transaxle")
    if rear is None or clutch is None or trans is None:
        return
    a, b = _pos(rear), _pos(trans)
    mid = (a + b) / 2.0
    half = engine_geometry.block_half_yz_m(engine)
    node("powertrain.bellhousing", [float(v) for v in mid], "engine-block-component", mass_kg=engine.mass_kg * 0.04,
         body_half_extent_m=[max(abs(b[0] - a[0]) / 2.0, 0.03), half * 0.9, half * 0.9])
    edge("powertrain.bellhousing_to_block", "powertrain.engine", "powertrain.bellhousing", "rigid-bolted-joint", radius=0.008)
    edge("powertrain.bellhousing_to_transmission", "powertrain.bellhousing", trans["identity"], "rigid-bolted-joint", radius=0.008)


# ---------------------------------------------------------------------
# forced induction: the blower on its manifold with the hat on top; each
# turbo as real housings with wastegate, blow-off and downpipe; the
# charge cooler and its pipes; mechanical injection's barrel valve and
# hat nozzles
# ---------------------------------------------------------------------

def _emit_centrifugal_blower(engine, nodes, edges, node, edge, node_by_id, rotor, plenum, half) -> None:
    """A gear-driven centrifugal supercharger as real parts: one volute
    per stage at the REAR of the block (a Merlin's wheelcase blower, a
    radial's rear-mounted impeller -- never a case on the manifold),
    the production rotor node becoming the first-stage impeller, a gear
    housing off the crank rear, the throttle/carburettor relocated to
    the blower INLET (a Merlin's updraft carb genuinely feeds the
    supercharger, not the manifold), and a routed outlet duct to the
    plenum -- the real duct-to-duct machine."""
    fi = engine.forced_induction
    pp = _pos(plenum)
    x_min, x_max = engine_geometry.crank_extent(engine_geometry.cylinder_sites(engine))
    stages = max(1, int(fi.stages))
    r_imp = half * (0.42 + 0.18 * min(fi.max_boost_frac, 1.2))       # impeller radius grows with the pressure ratio
    vol_r = r_imp * 1.45                                              # the volute around it
    vol_len = r_imp * 0.55
    centre0 = np.array([x_max + half * 0.6 + vol_len, half * 0.35, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    # gear housing on the crank rear
    gear_c = np.array([x_max + half * 0.3, half * 0.1, 0.0])
    node("powertrain.blower_gear_housing", [float(v) for v in gear_c], "engine-block-component", mass_kg=engine.mass_kg * 0.02,
         body_half_extent_m=[half * 0.25, half * 0.35, half * 0.35])
    if "powertrain.crank_shaft.rear" in node_by_id:
        edge("powertrain.blower_gear_housing_to_crank", "powertrain.crank_shaft.rear", "powertrain.blower_gear_housing", "rigid-bolted-joint", radius=0.006)
    prev_out = None
    for k in range(stages):
        c = centre0 + axis * (k * vol_len * 2.4)
        vid = "powertrain.blower_case" if k == 0 else f"powertrain.blower_case_stage_{k + 1}"
        node(vid, [float(v) for v in c], "engine-block-component", mass_kg=engine.mass_kg * (0.05 if k == 0 else 0.04),
             blower_type="centrifugal", stage=k + 1, drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(vol_r), drum_length_m=float(vol_len))
        if k == 0:
            rotor["reference_position"] = [float(v) for v in c]
            rotor["drum_axis"] = [1.0, 0.0, 0.0]; rotor["drum_radius_m"] = float(r_imp); rotor["drum_length_m"] = float(vol_len * 0.6)
            rotor["mass_kg"] = engine.mass_kg * 0.012
            edge("powertrain.blower_gear_to_impeller", "powertrain.blower_gear_housing", "supercharger_rotor", "rigid-keyed-hub", radius=0.008)
            edge("powertrain.blower_case_mount", vid, "powertrain.blower_gear_housing", "rigid-bolted-joint", radius=0.006)
        else:
            iid = f"supercharger_impeller_stage_{k + 1}"
            node(iid, [float(v) for v in c], "rotating-mass", mass_kg=engine.mass_kg * 0.01,
                 inertia_kg_m2=float(rotor.get("inertia_kg_m2", 0.0015)) * 0.8,
                 drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(r_imp * 0.9), drum_length_m=float(vol_len * 0.6))
            edge(f"{iid}_shaft", "supercharger_rotor", iid, "rigid-keyed-hub", radius=0.008)
            edge(f"{vid}_to_previous", prev_out, vid, "low-pressure-air-line", radius=float(r_imp * 0.5),
                 circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
            edge(f"{vid}_mount", vid, prev_out, "rigid-bolted-joint", radius=0.006)
        prev_out = vid
    # the throttle/carburettor moves to the blower inlet (draw-through)
    hat = node_by_id.get("powertrain.throttle_body")
    inlet_c = centre0 + np.array([-(vol_len + half * 0.35), half * 0.15, 0.0])
    if hat is not None:
        delta = inlet_c - _pos(hat)
        hat["reference_position"] = [float(v) for v in inlet_c]
        hat["inlet_kind"] = hat.get("inlet_kind", "throttle-body")
        for n in nodes:
            if n.get("part") == "powertrain.throttle_body" and n.get("kind") == "engine-block-port":
                n["reference_position"] = [float(v) for v in (_pos(n) + delta)]
        for e in edges:
            if e["identity"] == "powertrain.throttle_body_to_plenum":
                e["b"] = "powertrain.blower_case"
        af = node_by_id.get("powertrain.air_filter")
        if af is not None:
            af["reference_position"] = [float(v) for v in (inlet_c + np.array([-half * 0.45, half * 0.25, 0.0]))]
    # outlet: the last stage's volute to the plenum, a real routed duct
    duct_pts = duct_routing.duct_waypoints(centre0 + axis * ((stages - 1) * vol_len * 2.4), pp, rise=half * 0.9)
    duct_routing.lay_routed_pipe(
        node, edge, "powertrain.blower_case_to_plenum_charge", duct_pts, radius=float(r_imp * 0.5),
        circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature",
        start_identity=prev_out, end_identity="powertrain.intake_plenum")


def _emit_supercharger(engine, nodes, edges, node, edge, node_by_id) -> None:
    fi = engine.forced_induction
    rotor = node_by_id.get("supercharger_rotor")
    plenum = node_by_id.get("powertrain.intake_plenum")
    if fi.kind != "supercharger" or rotor is None or plenum is None:
        return
    half = engine_geometry.block_half_yz_m(engine)
    pp = _pos(plenum)
    if getattr(fi, "blower_type", "roots") == "centrifugal":
        _emit_centrifugal_blower(engine, nodes, edges, node, edge, node_by_id, rotor, plenum, half)
        return
    # the case is sized by the real declared rotor pack: lobe count sets
    # the rotor diameter class, boost fraction the case length (a 14-71
    # is longer than a 6-71; both real Roots sizes), belt ratio nothing
    # geometric -- disclosed proportions of the block's own half-width
    lobes = max(2, int(fi.lobe_count))
    r_rotor = half * (0.34 + 0.05 * (lobes - 2))
    case_half = np.array([half * (1.3 + 0.6 * min(fi.max_boost_frac, 1.5)), r_rotor * 1.15, r_rotor * 2.25])
    remote = fi.blower_mount == "remote"
    if remote:
        # "pass duct to duct": the case doesn't sit on the plenum at
        # all -- no manifold plate, no rigid stack. A declared position
        # if the engine gave one; otherwise a real default (low, to one
        # side, level with the belt plane) rather than guessing a
        # different default every call.
        x_min, _x_max = engine_geometry.crank_extent(engine_geometry.cylinder_sites(engine))
        default_pos = np.array([x_min - half * 1.6, -half * 0.2, half * 1.6])
        case_c = np.array(fi.blower_position_local, dtype=np.float64) if fi.blower_position_local is not None else default_pos
    else:
        plate_c = pp + UP * (float(plenum.get("body_half_extent_m", [0, half * 0.3, 0])[1]) + 0.015)
        case_c = plate_c + UP * (0.015 + case_half[1])
        node("powertrain.blower_manifold", [float(v) for v in plate_c], "engine-block-component", mass_kg=engine.mass_kg * 0.03,
             body_half_extent_m=[float(case_half[0]), 0.015, float(case_half[2])])
        edge("powertrain.blower_manifold_to_plenum", "powertrain.blower_manifold", "powertrain.intake_plenum", "rigid-bolted-joint", radius=0.008)
    node("powertrain.blower_case", [float(v) for v in case_c], "engine-block-component", mass_kg=engine.mass_kg * 0.09,
         lobe_count=lobes, body_half_extent_m=[float(v) for v in case_half])
    if not remote:
        edge("powertrain.blower_case_to_manifold", "powertrain.blower_case", "powertrain.blower_manifold", "rigid-bolted-joint", radius=0.008)
    # the production rotor node becomes rotor 1 inside the case; rotor 2
    # beside it, timed to it by the gear pair at the front bearing plate
    rotor["reference_position"] = [float(v) for v in (case_c + np.array([0.0, 0.0, -r_rotor]))]
    rotor["drum_axis"] = [1.0, 0.0, 0.0]; rotor["drum_radius_m"] = float(r_rotor * 0.96); rotor["drum_length_m"] = float(case_half[0] * 1.9)
    rotor["mass_kg"] = engine.mass_kg * 0.02
    node("supercharger_rotor_2", [float(v) for v in (case_c + np.array([0.0, 0.0, r_rotor]))], "rotating-mass",
         mass_kg=engine.mass_kg * 0.02, inertia_kg_m2=float(rotor.get("inertia_kg_m2", 0.0015)),
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(r_rotor * 0.96), drum_length_m=float(case_half[0] * 1.9))
    edge("supercharger_rotor_timing_gears", "supercharger_rotor", "supercharger_rotor_2", "geared-timing-drive",
         radius=0.006, ratio=-1.0)
    # drive snout forward to the belt plane, blower pulley on it, crank
    # pulley below on the damper, the cogged belt between them
    snout_len = max(0.06, abs(float(rotor["reference_position"][0]) - float(case_c[0]) - case_half[0]) + 0.08)
    snout_c = case_c + np.array([-(case_half[0] + snout_len / 2.0), 0.0, 0.0])
    node("powertrain.blower_snout", [float(v) for v in snout_c], "engine-block-component", mass_kg=3.0,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(r_rotor * 0.5), drum_length_m=float(snout_len))
    edge("powertrain.blower_snout_to_case", "powertrain.blower_snout", "powertrain.blower_case", "rigid-bolted-joint", radius=0.006)
    pulley_c = snout_c + np.array([-(snout_len / 2.0 + 0.02), 0.0, 0.0])
    node("powertrain.blower_pulley", [float(v) for v in pulley_c], "rotating-mass", mass_kg=1.5,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(r_rotor * 0.9), drum_length_m=0.04)
    edge("powertrain.blower_pulley_hub", "powertrain.blower_pulley", "supercharger_rotor", "rigid-keyed-hub", radius=0.008)
    if "powertrain.harmonic_balancer" in node_by_id:
        edge("powertrain.blower_belt", "powertrain.harmonic_balancer", "powertrain.blower_pulley", "cosmetic-belt-wrap", radius=0.006)
    # the burst panel / restraint strap over the case: real, mandated on
    # a nitro/alcohol blower, a plate and a strap over the top
    node("powertrain.blower_burst_panel", [float(v) for v in (case_c + UP * (case_half[1] + 0.006))], "engine-block-component",
         mass_kg=1.0, body_half_extent_m=[float(case_half[0] * 0.5), 0.006, float(case_half[2] * 0.9)])
    edge("powertrain.burst_panel_to_case", "powertrain.blower_burst_panel", "powertrain.blower_case", "rigid-bolted-joint", radius=0.005)
    # the hat (the throttle body dressing placed 1.3 radii over the plenum)
    # moves on top of the case, its butterflies now above the blower; the
    # scoop (air_filter "stacks") sits on the hat
    hat = node_by_id.get("powertrain.throttle_body")
    if hat is not None:
        hat_h = max(float(hat.get("body_half_extent_m", [0, half * 0.3, 0])[1]), half * 0.25)
        hat_c = case_c + UP * (case_half[1] + 0.012 + hat_h)
        delta = hat_c - _pos(hat)
        hat["reference_position"] = [float(v) for v in hat_c]
        hat["inlet_kind"] = "injector-hat"
        # the hat's own bungs (dressing's vacuum tap) ride up with it
        for n in nodes:
            if n.get("part") == "powertrain.throttle_body" and n.get("kind") == "engine-block-port":
                n["reference_position"] = [float(v) for v in (_pos(n) + delta)]
        for e in edges:
            if e["identity"] == "powertrain.throttle_body_to_plenum":
                e["b"] = "powertrain.blower_case"   # the hat feeds the blower, the blower feeds the plenum
        if remote:
            # far from the plenum now -- a real routed duct (up and
            # across, clearing whatever sits between them), not a
            # direct tube through the block
            duct_pts = duct_routing.duct_waypoints(case_c, pp, rise=half * 0.8)
            duct_routing.lay_routed_pipe(
                node, edge, "powertrain.blower_case_to_plenum_charge", duct_pts, radius=float(r_rotor * 0.6),
                circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature",
                start_identity="powertrain.blower_case", end_identity="powertrain.intake_plenum")
        else:
            edge("powertrain.blower_case_to_plenum_charge", "powertrain.blower_case", "powertrain.intake_plenum", "low-pressure-air-line",
                 radius=float(r_rotor * 0.6), circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
        af = node_by_id.get("powertrain.air_filter")
        if af is not None:
            af["reference_position"] = [float(v) for v in (hat_c + UP * (hat_h + half * 0.35))]
            af["scoop"] = True
            af["body_half_extent_m"] = [float(case_half[0] * 0.55), half * 0.3, float(case_half[2] * 0.6)]
            af["drum_axis"] = None


def _emit_mechanical_injection(engine, nodes, edges, node, edge, node_by_id) -> None:
    """Hat nozzles and the barrel valve of a mechanical (constant-flow)
    injection system: present when the build declares a mechanical pump
    AND runs an injector hat. Port nozzles are the real injector bosses
    the rail already feeds."""
    if engine.fuel_delivery.pump_kind != "mechanical":
        return
    hat = node_by_id.get("powertrain.throttle_body")
    if hat is None or hat.get("inlet_kind") != "injector-hat":
        return
    hp = _pos(hat)
    half = engine_geometry.block_half_yz_m(engine)
    barrels = int(hat.get("barrels", 2))
    pitch = float(hat.get("barrel_pitch_m", half * 0.5))
    # the barrel valve rides the throttle linkage on the hat's side
    bv = hp + np.array([0.0, 0.0, float(hat.get("body_half_extent_m", [0, 0, half * 0.4])[2]) + 0.03])
    node("powertrain.barrel_valve", [float(v) for v in bv], "engine-block-component", mass_kg=0.8,
         body_half_extent_m=[0.03, 0.025, 0.02])
    edge("powertrain.barrel_valve_to_hat", "powertrain.barrel_valve", "powertrain.throttle_body", "rigid-bolted-joint", radius=0.005)
    edge("powertrain.pump_to_barrel_valve", "fuel.pump", "powertrain.barrel_valve", "fuel-supply-line",
         radius=0.005, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
    if "powertrain.fuel_rail" in node_by_id:
        edge("powertrain.barrel_valve_to_port_rail", "powertrain.barrel_valve", "powertrain.fuel_rail", "fuel-supply-line",
             radius=0.004, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
    for bi in range(barrels):
        x = hp[0] + (bi - (barrels - 1) / 2.0) * pitch
        for side in (-1.0, 1.0):
            nid = f"powertrain.throttle_body.hat_nozzle_{bi + 1}{'l' if side < 0 else 'r'}"
            pos = np.array([x, hp[1] + 0.01, hp[2] + side * float(hat.get("body_half_extent_m", [0, 0, half * 0.4])[2]) * 0.6])
            node(nid, [float(v) for v in pos], "engine-block-port", port_kind="hat-nozzle",
                 port_direction=[0.0, 0.0, -side], port_radius_m=0.004, fluid_role="fuel", mating=False,
                 connected=True, plugged=False, part="powertrain.throttle_body", bung=True)
            edge(f"{nid}.feed", "powertrain.barrel_valve", nid, "fuel-supply-line", radius=0.003,
                 circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")


def _turbo_swoop_points(collector_pos: np.ndarray, x_min: float, half: float, side: float,
                        stage: int = 0) -> list[np.ndarray]:
    """A real top-mount-style header sweep: off the collector, UP (clear
    the block/valve cover), FORWARD (toward the front of the engine,
    -x), then BACK down into the turbine inlet from above -- the
    standard way a turbo ends up mounted up front and high with a short
    path back to a front-mounted intake, instead of hanging directly off
    the side of the header. `stage` shifts a later unit in a serial
    (compound) chain progressively further forward/up than the one
    before it, so each stage gets its own real position along the same
    swoop instead of stacking on top of the first.
    Returns real waypoints (collector -> ... -> turbine-inlet-arrival),
    consumed as one routed multi-segment pipe, the same real pattern
    dressing.py's own intake runners already route through waypoints."""
    stage_reach = 1.0 + stage * 0.85
    rise = half * (1.1 + stage * 0.35)
    p_lift = collector_pos + UP * (half * 0.6) + np.array([0.0, 0.0, side * half * 0.15])
    p_apex = np.array([x_min - half * (1.3 * stage_reach), collector_pos[1] + rise, side * half * 0.9])
    p_arrival = p_apex + np.array([-half * 0.4, half * 0.15, 0.0])   # swoop continues slightly forward-and-up into the inlet
    return [collector_pos, p_lift, p_apex, p_arrival]


def _emit_remote_air_intake(engine, nodes, edges, node, edge, node_by_id) -> None:
    """Where the filter actually breathes from -- independent of
    forced induction and independent of plenum_placement (that's about
    the chamber under the throttle; this is about the filter itself).
    "engine-bay" (default, most engines): leaves dressing.py's own
    air_filter placement alone entirely -- this whole function is a
    no-op. "remote-box"/"underside-snorkel": relocates that SAME real
    filter node to a real declared (or derived) spot elsewhere on the
    vehicle and re-routes its existing feed edge(s) through a real
    hose duct instead of teleporting air across empty space -- exactly
    the "mount anywhere, pass duct to duct" ask, applied to the intake
    side instead of the boost hardware."""
    intake = engine.intake_system
    source = getattr(intake, "air_source", "engine-bay")
    if source == "engine-bay":
        return
    af = node_by_id.get("powertrain.air_filter")
    if af is None:
        return
    half = engine_geometry.block_half_yz_m(engine)
    old_pos = _pos(af)
    # a real stub exactly where the filter used to sit, so every
    # existing downstream edge (one throttle body, or several on a
    # multi-throttle engine) keeps its own real connection unchanged --
    # only its "a"/"b" endpoint identity is repointed, never its count
    # or its own real routing
    hose_stub_id = "powertrain.air_filter.hose_end"
    node(hose_stub_id, [float(v) for v in old_pos], "engine-block-port", port_kind="air-hose-stub")
    for e in edges:
        if e.get("a") == "powertrain.air_filter":
            e["a"] = hose_stub_id
        if e.get("b") == "powertrain.air_filter":
            e["b"] = hose_stub_id
    x_min, _x_max = engine_geometry.crank_extent(engine_geometry.cylinder_sites(engine))
    if intake.air_box_position_local is not None:
        box_pos = np.array(intake.air_box_position_local, dtype=np.float64)
    else:
        # a real default -- low, ahead of the block, off to one side (a
        # fender-well/behind-the-headlight kind of spot), not reused
        # from any other accessory's own default position
        box_pos = np.array([x_min - half * 1.2, -half * 0.6, half * 2.4])
    af["reference_position"] = [float(v) for v in box_pos]
    filter_radius = max(0.02, float(af.get("drum_radius_m") or 0.03))
    duct_pts = duct_routing.duct_waypoints(box_pos, old_pos, rise=half * 0.5)
    duct_routing.lay_routed_pipe(
        node, edge, "powertrain.air_box_hose", duct_pts, radius=filter_radius,
        circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature",
        start_identity="powertrain.air_filter", end_identity=hose_stub_id)
    if source == "underside-snorkel":
        if intake.snorkel_inlet_position_local is not None:
            inlet_pos = np.array(intake.snorkel_inlet_position_local, dtype=np.float64)
        else:
            inlet_pos = box_pos + np.array([0.0, half * 4.0, 0.0])   # up to hood/roof height, a real default rise
        node("powertrain.snorkel_inlet", [float(v) for v in inlet_pos], "engine-block-component", mass_kg=0.6,
             body_half_extent_m=[half * 0.18, half * 0.18, half * 0.18])
        # a straight riser, deliberately NOT the over-the-top duct_
        # waypoints shape -- a real snorkel runs its box straight up,
        # it doesn't need to dodge anything between box and inlet
        duct_routing.lay_routed_pipe(
            node, edge, "powertrain.snorkel_riser", [inlet_pos, box_pos], radius=filter_radius,
            circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature",
            start_identity="powertrain.snorkel_inlet", end_identity="powertrain.air_filter")


def _emit_turbo_hardware(engine, nodes, edges, node, edge, node_by_id) -> None:
    fi = engine.forced_induction
    if fi.kind != "turbo":
        return
    turbos = [n for n in nodes if n["identity"].startswith("powertrain.turbocharger")
              and n.get("kind") == "rotating-mass"]
    if not turbos:
        return
    half = engine_geometry.block_half_yz_m(engine)
    collectors = [n for n in nodes if n.get("kind") == "exhaust-collector"]
    plenum = node_by_id.get("powertrain.intake_plenum")
    x_min, _ = engine_geometry.crank_extent(engine_geometry.cylinder_sites(engine))
    # the charge cooler: one air-to-air core ahead of the engine, fed by
    # every compressor, feeding the plenum -- present whenever the intake
    # resolved to the piped placement (dressing) or any turbo exists
    cooler_c = np.array([x_min - half * 2.2, half * 0.9, 0.0])
    node("powertrain.charge_cooler", [float(v) for v in cooler_c], "engine-block-component", mass_kg=engine.mass_kg * 0.02,
         exchanger_kind="air-to-air-charge-cooler", body_half_extent_m=[half * 0.25, half * 1.1, half * 1.9])
    if plenum is not None:
        edge("powertrain.cold_side_charge_pipe", "powertrain.charge_cooler", "powertrain.intake_plenum", "low-pressure-air-line",
             radius=0.03, circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")

    # Real ordering for a serial (compound) chain: the FIRST-listed unit
    # is the small, quick-spooling stage nearest the collector; every
    # later one moves progressively further along the same swoop (more
    # forward, higher) and takes its hot feed from the PREVIOUS stage's
    # turbine outlet instead of straight off the collector. Parallel
    # (default) treats every turbo independently, one per its own
    # nearest collector, each with its own single-stage swoop.
    serial = fi.turbo_layout == "serial" and len(turbos) > 1
    turbos_sorted = sorted(turbos, key=lambda t: t["identity"]) if serial else turbos
    prev_stage_outlet_id: str | None = None
    prev_stage_pos: np.ndarray | None = None

    for stage_idx, t in enumerate(turbos_sorted):
        base = t["identity"]
        if serial and stage_idx > 0 and prev_stage_pos is not None:
            coll_pos = prev_stage_pos
            feed_from_id = prev_stage_outlet_id
        else:
            coll_pos = None
            feed_from_id = None
        if coll_pos is None:
            tp0 = _pos(t)
            side = 1.0 if tp0[2] >= 0 else -1.0
            if collectors:
                coll = min(collectors, key=lambda c: abs(_pos(c)[2] - tp0[2]) + abs(_pos(c)[0] - tp0[0]) * 0.3)
                coll_pos = _pos(coll)
                feed_from_id = coll["identity"]
            else:
                coll_pos = tp0
                feed_from_id = None
        else:
            side = 1.0 if coll_pos[2] >= 0 else -1.0

        stage = stage_idx if serial else 0
        pts = _turbo_swoop_points(coll_pos, x_min, half, side, stage=stage)
        tp = pts[-1]   # the turbo's own real position IS the swoop's arrival point -- derived, not independently placed
        tp0 = _pos(t)
        delta = tp - tp0
        t["reference_position"] = [float(v) for v in tp]
        # every real child port the production graph hung off this
        # turbo's OWN original position (the oil feed/drain bosses --
        # "powertrain.turbocharger.oil_feed"/".oil_drain") has to move
        # by the same delta, or its oil lines stay pinned to the old
        # spot while the housings/wastegate/BOV (all built fresh off
        # `tp` below, never off the old position) swoop away from them
        for n in nodes:
            if n["identity"].startswith(base + ".") and n["identity"] not in (
                    f"{base}.compressor_housing", f"{base}.turbine_housing",
                    f"{base}.wastegate", f"{base}.blow_off_valve", f"{base}.downpipe"):
                n["reference_position"] = [float(v) for v in (_pos(n) + delta)]

        # real housings around the production point mass: compressor
        # (cold, forward toward the charge pipe) and turbine (hot,
        # rearward toward the swoop it just arrived on), the cartridge
        # between them
        r_c = half * 0.55
        node(f"{base}.compressor_housing", [float(v) for v in (tp + np.array([-r_c * 0.9, 0.0, 0.0]))], "engine-block-component",
             mass_kg=2.5, drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(r_c), drum_length_m=float(r_c * 0.9))
        node(f"{base}.turbine_housing", [float(v) for v in (tp + np.array([r_c * 0.9, 0.0, 0.0]))], "exhaust-component",
             mass_kg=3.5, drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(r_c * 0.85), drum_length_m=float(r_c * 0.9))
        edge(f"{base}.chra_compressor", base, f"{base}.compressor_housing", "rigid-bolted-joint", radius=0.005)
        edge(f"{base}.chra_turbine", base, f"{base}.turbine_housing", "rigid-bolted-joint", radius=0.005)
        # wastegate on the turbine housing, blow-off on the compressor side
        node(f"{base}.wastegate", [float(v) for v in (tp + np.array([r_c * 0.9, r_c * 1.05, 0.0]))], "exhaust-component",
             mass_kg=0.8, wastegate_frac=float(fi.wastegate_frac), body_half_extent_m=[0.03, 0.03, 0.025])
        edge(f"{base}.wastegate_to_turbine", f"{base}.wastegate", f"{base}.turbine_housing", "rigid-bolted-joint", radius=0.004)
        node(f"{base}.blow_off_valve", [float(v) for v in (tp + np.array([-r_c * 0.9, r_c * 1.05, 0.0]))], "engine-block-component",
             mass_kg=0.4, drum_axis=[0.0, 1.0, 0.0], drum_radius_m=0.02, drum_length_m=0.05)
        edge(f"{base}.bov_to_compressor", f"{base}.blow_off_valve", f"{base}.compressor_housing", "rigid-bolted-joint", radius=0.004)

        # bearing oil: production only wires a turbo's feed/drain when it
        # made the pump itself; a dry-sump two-stroke's pump and tank come
        # from the toy side (drivetrain_graph / dressing), so wire them here
        if "powertrain.oil_pump" in node_by_id and not any(e["identity"] == f"powertrain.turbo_oil_feed{base.replace('powertrain.turbocharger', '')}"
                                                            for e in edges):
            suffix = base.replace("powertrain.turbocharger", "")
            if f"{base}.oil_feed" in node_by_id:
                edge(f"powertrain.turbo_oil_feed{suffix}", "powertrain.oil_pump", f"{base}.oil_feed", "oil-line", radius=.003,
                     circuit_identity="oil", medium_rate_state="oil-flow-and-temperature-and-pressure")
            if f"{base}.oil_drain" in node_by_id:
                drain_to = "powertrain.oil_reserve_tank" if "powertrain.oil_reserve_tank" in node_by_id else "powertrain.oil_pump"
                edge(f"powertrain.turbo_oil_drain{suffix}", f"{base}.oil_drain", drain_to, "oil-line", radius=.005,
                     circuit_identity="oil", medium_rate_state="oil-flow-and-temperature-and-pressure", gravity_drain=True)
        # hot side: the real swept multi-segment up-pipe from the feed
        # point (a collector for stage 0, the previous stage's turbine
        # outlet for a later serial stage) up, forward, and back into
        # this turbine's own inlet -- duct_routing's shared waypoint
        # helper, the same one a remote-mounted supercharger's ducts
        # and a remote air box/snorkel use
        if feed_from_id is not None:
            duct_routing.lay_routed_pipe(
                node, edge, f"{base}.up_pipe", pts, radius=0.028,
                circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature",
                start_identity=feed_from_id, end_identity=f"{base}.turbine_housing", kind="exhaust-flow-path")

        if serial and stage_idx < len(turbos_sorted) - 1:
            # this stage's turbine OUTLET feeds the next stage's inlet
            # (compound), not the tailpipe -- no downpipe of its own
            prev_stage_outlet_id = f"{base}.turbine_housing"
            prev_stage_pos = tp
            edge(f"{base}.hot_side_charge_pipe", f"{base}.compressor_housing", "powertrain.charge_cooler", "low-pressure-air-line",
                 radius=0.028, circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
            continue

        # last (or only) stage: a real downpipe leaves the turbine
        # outlet rearward toward the tailpipe
        dp_end = tp + np.array([r_c * 0.9 + 0.30, -half * 0.6, side * half * 0.4])
        node(f"{base}.downpipe", [float(v) for v in dp_end], "exhaust-component", mass_kg=2.0, chassis_side=False,
             drum_axis=[float(v) for v in _unit(dp_end - (tp + np.array([r_c * 0.9, 0.0, 0.0])))], drum_radius_m=0.032, drum_length_m=0.30)
        edge(f"{base}.turbine_to_downpipe", f"{base}.turbine_housing", f"{base}.downpipe", "exhaust-flow-path", radius=0.032,
             circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature")
        # cold side: compressor outlet to the shared charge cooler
        edge(f"{base}.hot_side_charge_pipe", f"{base}.compressor_housing", "powertrain.charge_cooler", "low-pressure-air-line",
             radius=0.028, circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")


# ---------------------------------------------------------------------
# aircraft / industrial / air-cooled families
# ---------------------------------------------------------------------

def _purge(nodes, edges, identities: set[str]) -> None:
    nodes[:] = [n for n in nodes if n["identity"] not in identities]
    edges[:] = [e for e in edges if e["a"] not in identities and e["b"] not in identities]


def _emit_aircraft_drive(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """An aircraft engine has no clutch, gearbox or transfer case: the
    crank drives a propeller through a fixed reduction gear, and the
    propeller's own constant-speed governor sits on that gearcase. The
    production driveline chain (a car's) is replaced, mounts and all;
    the dyno stays coupled to powertrain.engine directly, so nothing the
    solver reads changes."""
    if getattr(engine, "ignition_profile", "") != "aircraft-dual-magneto":
        return
    fit = crank_end_fittings(layout)
    if fit is None:
        return
    gone = {"powertrain.clutch", "powertrain.transmission", "powertrain.transfer_case",
            "powertrain.direct_drive_bypass", "powertrain.pre_clutch_flywheel_wrench", "powertrain.bellhousing",
            "mount.transmission_left", "mount.transmission_right", "mount.transfer_case_left", "mount.transfer_case_right"}
    _purge(nodes, edges, gone)
    half = engine_geometry.block_half_yz_m(engine)
    bore = fit["bore_m"]
    # the prop end is the FRONT (timing-cover) end of the crank on both
    # real engines here; the reduction gearcase bolts to the front face
    # and the prop shaft leaves it on the crank axis, raised on the Merlin's
    # spur reduction, on-axis on the Wasp's planetary
    face_x = _block_front_face_x(engine, nodes)
    trans = getattr(engine, "transmission", None)
    ratio = float(getattr(trans, "final_drive_ratio", 0.0) or 0.0)
    ratio = ratio if 1.1 <= ratio <= 3.5 else 2.0
    case_len = bore * 0.9
    case_c = np.array([face_x - 0.03 - case_len / 2.0, fit["y0"], fit["z0"]])
    node("powertrain.prop_reduction_gearbox", [float(v) for v in case_c], "engine-block-component",
         mass_kg=engine.mass_kg * 0.06, reduction_ratio=ratio,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(half * 0.9), drum_length_m=float(case_len))
    edge("powertrain.crank_to_prop_reduction", "powertrain.engine", "powertrain.prop_reduction_gearbox",
         "geared-timing-drive", radius=0.02, ratio=1.0 / ratio)
    shaft_len = bore * 1.2
    shaft_c = case_c + np.array([-(case_len / 2.0 + shaft_len / 2.0), 0.0, 0.0])
    node("powertrain.prop_shaft", [float(v) for v in shaft_c], "rotating-mass", mass_kg=engine.mass_kg * 0.02,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(bore * 0.22), drum_length_m=float(shaft_len))
    edge("powertrain.prop_reduction_to_shaft", "powertrain.prop_reduction_gearbox", "powertrain.prop_shaft", "rigid-keyed-hub", radius=0.02)
    hub_c = shaft_c + np.array([-(shaft_len / 2.0 + bore * 0.25), 0.0, 0.0])
    node("powertrain.prop_hub", [float(v) for v in hub_c], "rotating-mass", mass_kg=engine.mass_kg * 0.03,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(bore * 0.9), drum_length_m=float(bore * 0.5))
    edge("powertrain.prop_shaft_to_hub", "powertrain.prop_shaft", "powertrain.prop_hub", "rigid-keyed-hub", radius=0.02)
    # the constant-speed governor rides the gearcase, driven off it
    gov_c = case_c + np.array([0.0, half * 0.9 + 0.04, half * 0.3])
    node("powertrain.propeller_governor", [float(v) for v in gov_c], "engine-block-component", mass_kg=2.5,
         drum_axis=[0.0, 1.0, 0.0], drum_radius_m=0.04, drum_length_m=0.09)
    edge("powertrain.propeller_governor_drive", "powertrain.prop_reduction_gearbox", "powertrain.propeller_governor",
         "geared-timing-drive", radius=0.006)


def _emit_governor(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """A hit-and-miss engine's flyball governor and its latch-out
    linkage to the exhaust rocker -- the mechanism `governor_mode ==
    "hit_and_miss"` already simulates every revolution, with no part."""
    if getattr(engine, "governor_mode", "throttle") != "hit_and_miss":
        return
    fit = crank_end_fittings(layout)
    if fit is None:
        return
    bore = fit["bore_m"]
    face_x = _block_front_face_x(engine, nodes)
    gov_c = np.array([face_x - 0.02 - bore * 0.35, fit["y0"] + bore * 0.7, fit["z0"] + bore * 0.6])
    node("powertrain.flyball_governor", [float(v) for v in gov_c], "engine-block-component", mass_kg=3.0,
         governor_kind="flyball", drum_axis=[0.0, 1.0, 0.0], drum_radius_m=float(bore * 0.25), drum_length_m=float(bore * 0.6))
    edge("powertrain.flyball_governor_drive", "powertrain.engine", "powertrain.flyball_governor", "geared-timing-drive",
         radius=0.008, ratio=1.0)
    node("powertrain.governor_latch_linkage", [float(v) for v in (gov_c + np.array([bore * 0.4, bore * 0.3, 0.0]))],
         "engine-block-component", mass_kg=0.4, body_half_extent_m=[bore * 0.4, 0.008, 0.008])
    edge("powertrain.governor_to_latch", "powertrain.flyball_governor", "powertrain.governor_latch_linkage", "rigid-bolted-joint", radius=0.005)


def _emit_injection_pump(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """A compression-ignition engine's injection pump -- inline or
    distributor type, cam-driven off the block's flank, feeding the
    rail the direct injectors already hang from. Its own governor is
    part of the pump body (the `diesel-injection-governor` profile)."""
    if not getattr(engine, "compression_ignition", False):
        return
    rail = node_by_id.get("powertrain.fuel_rail")
    fit = crank_end_fittings(layout)
    if rail is None or fit is None:
        return
    half = engine_geometry.block_half_yz_m(engine)
    bore = fit["bore_m"]
    rp = _pos(rail)
    n_cyl = max(1, engine.architecture.cylinders)
    pump_len = min(bore * 0.55 * n_cyl, abs(float(rail.get("body_half_extent_m", [bore, 0, 0])[0])) * 1.6)
    pump_c = np.array([rp[0], fit["y0"] + half * 0.55, fit["z0"] + half * 1.15 + bore * 0.25])
    node("powertrain.injection_pump", [float(v) for v in pump_c], "engine-block-component", mass_kg=engine.mass_kg * 0.03,
         pump_kind="inline-injection-pump" if n_cyl >= 4 else "distributor-injection-pump",
         body_half_extent_m=[float(pump_len / 2.0), float(bore * 0.3), float(bore * 0.25)])
    edge("powertrain.injection_pump_drive", "powertrain.engine", "powertrain.injection_pump", "geared-timing-drive",
         radius=0.01, ratio=0.5)
    edge("powertrain.injection_pump_to_rail", "powertrain.injection_pump", "powertrain.fuel_rail", "fuel-supply-line",
         radius=0.004, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure", high_pressure=True)
    if "fuel.pump" in node_by_id:
        # the lift pump feeds the injection pump's gallery, not the rail
        for e in edges:
            if e["identity"] == "fuel.pump_to_rail":
                e["b"] = "powertrain.injection_pump"


def _emit_dry_sump_extras(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """A real dry sump is more than a tank and a scavenge pump: the
    multi-stage pump is belt-driven off the crank, and the oil goes
    through a cooler on its way back to the tank."""
    from dressing import derive_dressing
    if derive_dressing(engine).lube != "dry-sump":
        return
    tank = node_by_id.get("powertrain.oil_reserve_tank")
    scav = node_by_id.get("powertrain.scavenge_pump")
    if tank is None:
        return
    half = engine_geometry.block_half_yz_m(engine)
    tp = _pos(tank)
    cooler_c = tp + np.array([0.0, half * 0.9, 0.0])
    node("powertrain.oil_cooler", [float(v) for v in cooler_c], "engine-block-component", mass_kg=2.5,
         exchanger_kind="oil-to-air-cooler", body_half_extent_m=[half * 0.5, half * 0.35, half * 0.12])
    edge("powertrain.scavenge_to_oil_cooler", scav["identity"] if scav is not None else "powertrain.oil_pump",
         "powertrain.oil_cooler", "oil-line", radius=0.008, circuit_identity="oil",
         medium_rate_state="oil-flow-and-temperature-and-pressure")
    edge("powertrain.oil_cooler_to_tank", "powertrain.oil_cooler", "powertrain.oil_reserve_tank", "oil-line", radius=0.008,
         circuit_identity="oil", medium_rate_state="oil-flow-and-temperature-and-pressure")
    if scav is not None and "powertrain.harmonic_balancer" in node_by_id:
        sp = _pos(scav)
        face_x = _block_front_face_x(engine, nodes)
        pulley_c = np.array([face_x - 0.054, sp[1], sp[2]])
        node("powertrain.scavenge_pump.pulley", [float(v) for v in pulley_c], "rotating-mass", mass_kg=0.5,
             drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(half * 0.24), drum_length_m=0.022)
        edge("powertrain.scavenge_pump.pulley.hub", scav["identity"], "powertrain.scavenge_pump.pulley", "rigid-keyed-hub", radius=0.006)
        edge("powertrain.dry_sump_belt", "powertrain.harmonic_balancer", "powertrain.scavenge_pump.pulley", "cosmetic-belt-wrap", radius=0.005)


def _emit_air_cooling(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """An air-cooled engine's cooling hardware -- the whole cooling
    system, which used to be absent because the production cooling
    stack is gated on a water pump. A boxer/inline gets the fan, its
    housing and the tin shrouds ducting air over the fins; a radial
    gets per-cylinder baffles and cowl flaps at the rear."""
    acc = engine.accessories
    if acc.water_pump or engine.kind != "combustion" or not engine.architecture.cylinders:
        return
    half = engine_geometry.block_half_yz_m(engine)
    fit = crank_end_fittings(layout)
    if fit is None:
        return
    bore = fit["bore_m"]
    if engine.architecture.radial:
        # a radial folds its heads into one block node (drivetrain_graph),
        # so the per-cylinder positions come from the real cylinder sites;
        # baffles between the cylinders, cowl flaps in a ring behind them
        sites = engine_geometry.cylinder_sites(engine)
        for s_ in sites:
            hp = np.array(s_.position, dtype=np.float64)
            radial_dir = _unit(np.array([0.0, hp[1] - fit["y0"], hp[2] - fit["z0"]]))
            node(f"powertrain.cylinder_{s_.number}.baffle", [float(v) for v in (hp + radial_dir * bore * 0.15 + CRANK_AXIS * bore * 0.35)],
                 "engine-block-component", mass_kg=0.3, body_half_extent_m=[bore * 0.05, bore * 0.45, bore * 0.45])
        ring_r = float(np.mean([np.hypot(s_.position[1] - fit["y0"], s_.position[2] - fit["z0"]) for s_ in sites])) if sites else half * 2.0
        n_flaps = max(4, len(sites))
        for i in range(n_flaps):
            a = 2.0 * np.pi * i / n_flaps
            fc = np.array([fit["flywheel_centre"][0] - bore * 0.2, fit["y0"] + ring_r * 1.05 * np.cos(a), fit["z0"] + ring_r * 1.05 * np.sin(a)])
            node(f"powertrain.cowl_flap_{i + 1}", [float(v) for v in fc], "engine-block-component", mass_kg=0.4,
                 body_half_extent_m=[bore * 0.3, bore * 0.04 + abs(np.sin(a)) * bore * 0.2, bore * 0.04 + abs(np.cos(a)) * bore * 0.2])
        return
    if not acc.mechanical_fan:
        return
    # boxer / inline air-cooled: the fan on the crank/alternator pulley in
    # its housing, tins over each bank ducting the blast across the fins
    face_x = _block_front_face_x(engine, nodes)
    fan_c = np.array([face_x - 0.09, fit["y0"] + half * 1.1, fit["z0"]])
    node("powertrain.cooling_fan_housing", [float(v) for v in fan_c], "engine-block-component", mass_kg=2.0,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=float(half * 1.3), drum_length_m=0.06)
    fan = node_by_id.get("powertrain.cooling_fan")
    if fan is None:
        node("powertrain.cooling_fan", [float(v) for v in fan_c], "rotating-mass", mass_kg=1.2,
             fan_disk_radius_m=float(half * 1.15))
        edge("powertrain.cooling_fan_belt", "powertrain.harmonic_balancer" if "powertrain.harmonic_balancer" in node_by_id else "powertrain.engine",
             "powertrain.cooling_fan", "cosmetic-belt-wrap", radius=0.005)
    else:
        fan["reference_position"] = [float(v) for v in fan_c]
    edge("powertrain.fan_housing_to_block", "powertrain.cooling_fan_housing", "powertrain.engine", "rigid-bolted-joint", radius=0.006)
    from head_mesh import banks
    for bi, gs in enumerate(banks(layout)):
        xs = [float(g.base[0]) for g in gs]
        c = np.mean([np.array(g.base) + np.array(g.axis) * g.length_m for g in gs], axis=0)
        axis = _unit(gs[0].axis)
        tin_c = c - axis * (bore * 0.15)
        node(f"powertrain.cooling_tin_bank{bi + 1}", [float(v) for v in tin_c], "engine-block-component", mass_kg=1.5,
             body_half_extent_m=[float((max(xs) - min(xs)) / 2.0 + bore * 0.6), 0.006, float(bore * 0.9)])
        edge(f"powertrain.cooling_tin_bank{bi + 1}_duct", "powertrain.cooling_fan_housing", f"powertrain.cooling_tin_bank{bi + 1}",
             "low-pressure-air-line", radius=float(half * 0.35), circuit_identity="cooling-air",
             medium_rate_state="cooling-air-flow-and-temperature")


# ---------------------------------------------------------------------
# the non-piston kinds: their own real parts, sized from their own
# declared specs, around the single block body each already has
# ---------------------------------------------------------------------

def _drum(node, identity, centre, axis, radius, length, kind="engine-block-component", **attrs) -> None:
    node(identity, [float(v) for v in centre], kind, drum_axis=[float(v) for v in _unit(axis)],
         drum_radius_m=float(radius), drum_length_m=float(length), **attrs)


def _emit_turbine_gas_path(engine, nodes, edges, node, edge, node_by_id) -> None:
    """gas_turbine.py models compressor, combustor and turbine
    thermodynamically; the graph declared none of them. Sized from the
    same TurbineSpec: design mass flow sets the annulus (flow area at a
    disclosed axial Mach), pressure ratio the compressor's length."""
    spec = getattr(engine, "turbine", None)
    body = node_by_id.get("powertrain.engine_block_body")
    if engine.kind != "turbine" or spec is None or body is None:
        return
    bp = _pos(body)
    # annulus radius from design mass flow: mdot = rho * A * V at a real
    # inlet velocity class (~150 m/s), rho ~1.2 -- a disclosed sizing
    # relation, not a fit
    area = spec.mdot_design_kg_s / (1.2 * 150.0)
    r_in = max(0.06, float(np.sqrt(area / np.pi)))
    r_out = r_in * 0.55                                  # centrifugal impeller exit is wider than its eye
    l_comp = r_in * (0.9 + 0.12 * min(spec.design_pressure_ratio, 12.0))
    x0 = bp[0] - l_comp * 0.9
    # inlet plenum + screen, compressor (impeller + diffuser), combustor
    # annulus, NGVs + turbine wheel, power turbine, exhaust duct/eductor
    _drum(node, "powertrain.inlet_plenum", [x0 - r_in * 0.7, bp[1], bp[2]], CRANK_AXIS, r_in * 1.15, r_in * 0.6, mass_kg=4.0)
    node("powertrain.inlet_screen", [float(x0 - r_in * 1.05), float(bp[1]), float(bp[2])], "engine-block-port", port_kind="inlet-screen",
         port_direction=[-1.0, 0.0, 0.0], port_radius_m=float(r_in * 1.1), fluid_role="intake-air", mating=False, connected=True, plugged=False,
         part="powertrain.inlet_plenum", bung=True)
    _drum(node, "powertrain.compressor_impeller", [x0 + l_comp * 0.35, bp[1], bp[2]], CRANK_AXIS, r_in * 0.95, l_comp * 0.6,
          kind="rotating-mass", mass_kg=engine.mass_kg * 0.05, inertia_kg_m2=float(spec.shaft_inertia_kg_m2) * 0.4)
    _drum(node, "powertrain.compressor_diffuser", [x0 + l_comp * 0.75, bp[1], bp[2]], CRANK_AXIS, r_in * 1.45, l_comp * 0.3, mass_kg=6.0)
    xc = x0 + l_comp * 1.15
    _drum(node, "powertrain.combustor", [xc, bp[1], bp[2]], CRANK_AXIS, r_in * 1.5, r_in * 1.3, mass_kg=engine.mass_kg * 0.08,
          combustor_kind="annular-can")
    _drum(node, "powertrain.turbine_ngv", [xc + r_in * 0.9, bp[1], bp[2]], CRANK_AXIS, r_in * 1.1, r_in * 0.25, mass_kg=3.0)
    _drum(node, "powertrain.turbine_wheel", [xc + r_in * 1.25, bp[1], bp[2]], CRANK_AXIS, r_in * 1.0, r_in * 0.35,
          kind="rotating-mass", mass_kg=engine.mass_kg * 0.05, inertia_kg_m2=float(spec.shaft_inertia_kg_m2) * 0.6)
    _drum(node, "powertrain.power_turbine", [xc + r_in * 1.75, bp[1], bp[2]], CRANK_AXIS, r_in * 1.1, r_in * 0.35,
          kind="rotating-mass", mass_kg=engine.mass_kg * 0.04, inertia_kg_m2=float(spec.shaft_inertia_kg_m2) * 0.5)
    _drum(node, "powertrain.exhaust_duct", [xc + r_in * 2.9, bp[1] + r_in * 0.3, bp[2]], _unit(np.array([1.0, 0.35, 0.0])), r_in * 1.05, r_in * 1.8,
          kind="exhaust-component", mass_kg=5.0)
    for a, b, k, r in (("powertrain.inlet_plenum", "powertrain.compressor_impeller", "low-pressure-air-line", r_in),
                       ("powertrain.compressor_impeller", "powertrain.compressor_diffuser", "low-pressure-air-line", r_in * 0.6),
                       ("powertrain.compressor_diffuser", "powertrain.combustor", "low-pressure-air-line", r_in * 0.6),
                       ("powertrain.combustor", "powertrain.turbine_ngv", "exhaust-flow-path", r_in * 0.8),
                       ("powertrain.turbine_ngv", "powertrain.turbine_wheel", "exhaust-flow-path", r_in * 0.8),
                       ("powertrain.turbine_wheel", "powertrain.power_turbine", "exhaust-flow-path", r_in * 0.9),
                       ("powertrain.power_turbine", "powertrain.exhaust_duct", "exhaust-flow-path", r_in)):
        edge(f"{a}.to_{b.split('.')[-1]}", a, b, k, radius=float(r),
             circuit_identity="intake-air" if k == "low-pressure-air-line" else "exhaust",
             medium_rate_state="intake-air-flow-and-temperature" if k == "low-pressure-air-line" else "exhaust-pulse-pressure-and-temperature")
    # the gas-generator shaft ties impeller to turbine wheel; the power
    # turbine drives the output through the reduction gearbox the spec
    # already declares (reduction_ratio)
    edge("powertrain.gas_generator_shaft", "powertrain.compressor_impeller", "powertrain.turbine_wheel", "rigid-keyed-hub", radius=0.015)
    _drum(node, "powertrain.power_turbine_reduction", [xc + r_in * 2.2, bp[1] - r_in * 1.4, bp[2]], CRANK_AXIS, r_in * 0.7, r_in * 0.8,
          mass_kg=engine.mass_kg * 0.06, reduction_ratio=float(spec.reduction_ratio))
    edge("powertrain.power_turbine_to_reduction", "powertrain.power_turbine", "powertrain.power_turbine_reduction", "geared-timing-drive",
         radius=0.012, ratio=1.0 / float(spec.reduction_ratio))
    edge("powertrain.reduction_to_output", "powertrain.power_turbine_reduction", "powertrain.engine", "rigid-keyed-hub", radius=0.015)
    # fuel: FCU on the accessory side feeding the nozzles in the combustor
    # from the rail the graph already has; light-off igniter + exciter
    rail = node_by_id.get("powertrain.fuel_rail")
    _drum(node, "powertrain.fuel_control_unit", [x0 + l_comp * 0.5, bp[1] - r_in * 1.6, bp[2] + r_in * 0.6], UP, r_in * 0.35, r_in * 0.7, mass_kg=4.0)
    if rail is not None:
        edge("powertrain.rail_to_fcu", "powertrain.fuel_rail", "powertrain.fuel_control_unit", "fuel-supply-line", radius=0.004,
             circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
    n_nozzles = 6
    for i in range(n_nozzles):
        a = 2.0 * np.pi * i / n_nozzles
        p = np.array([xc - r_in * 0.55, bp[1] + r_in * 1.1 * np.cos(a), bp[2] + r_in * 1.1 * np.sin(a)])
        nid = f"powertrain.combustor.nozzle_{i + 1}"
        node(nid, [float(v) for v in p], "engine-block-port", port_kind="fuel-nozzle", port_direction=[0.0, float(np.cos(a)), float(np.sin(a))],
             port_radius_m=0.006, fluid_role="fuel", mating=False, connected=True, plugged=False, part="powertrain.combustor", bung=True)
        edge(f"{nid}.feed", "powertrain.fuel_control_unit", nid, "fuel-supply-line", radius=0.003, circuit_identity="fuel",
             medium_rate_state="fuel-flow-and-pressure")
    node("powertrain.combustor.igniter", [float(xc - r_in * 0.3), float(bp[1] + r_in * 1.4), float(bp[2])], "engine-block-port",
         port_kind="light-off-igniter", port_direction=[0.0, 1.0, 0.0], port_radius_m=0.008, fluid_role="ignition", mating=False,
         connected=True, plugged=False, part="powertrain.combustor", bung=True)
    node("powertrain.ignition_exciter", [float(x0 + l_comp * 0.5), float(bp[1] - r_in * 1.6), float(bp[2] - r_in * 0.6)], "engine-block-component",
         mass_kg=1.2, body_half_extent_m=[0.05, 0.035, 0.04])
    edge("powertrain.exciter_to_igniter", "powertrain.ignition_exciter", "powertrain.combustor.igniter", "ignition-lead", radius=0.004, circuit_identity="ignition")
    # accessory gearbox on the compressor end with the starter-generator
    _drum(node, "powertrain.accessory_gearbox", [x0 - r_in * 0.2, bp[1] - r_in * 1.5, bp[2]], CRANK_AXIS, r_in * 0.6, r_in * 0.5, mass_kg=engine.mass_kg * 0.05)
    edge("powertrain.gas_generator_to_accessory_gearbox", "powertrain.compressor_impeller", "powertrain.accessory_gearbox", "geared-timing-drive", radius=0.01)
    _drum(node, "powertrain.starter_generator", [x0 - r_in * 0.2, bp[1] - r_in * 1.5, bp[2] - r_in * 1.0], CRANK_AXIS, r_in * 0.35, r_in * 0.9,
          mass_kg=8.0, starter_kind="starter-generator")
    edge("powertrain.starter_generator_to_gearbox", "powertrain.starter_generator", "powertrain.accessory_gearbox", "rigid-keyed-hub", radius=0.008)
    node("powertrain.bleed_valve", [float(x0 + l_comp * 0.9), float(bp[1] + r_in * 1.6), float(bp[2])], "engine-block-component", mass_kg=0.8,
         body_half_extent_m=[0.03, 0.025, 0.03])
    edge("powertrain.bleed_valve_to_diffuser", "powertrain.bleed_valve", "powertrain.compressor_diffuser", "rigid-bolted-joint", radius=0.005)


def _emit_electric_drive_unit(engine, nodes, edges, node, edge, node_by_id) -> None:
    """The inverter, the motor itself (stator, rotor, housing, resolver),
    its cooling plate, the reduction (a single-speed gearset + diff on
    the EV, a planetary gearhead with an output flange on the servo) and
    the HV switchgear at the battery."""
    if engine.kind not in ("electric", "servo-electric"):
        return
    body = node_by_id.get("powertrain.engine_block_body")
    if body is None:
        return
    bp = _pos(body)
    half = engine_geometry.block_half_yz_m(engine)
    servo = engine.kind == "servo-electric"
    r_motor = half * (0.75 if servo else 1.0)
    l_motor = half * (1.4 if servo else 2.2)
    # motor housing on the block body's own centre, rotor inside, stator
    # around it, resolver on the tail
    _drum(node, "powertrain.motor_housing", bp, CRANK_AXIS, r_motor * 1.1, l_motor, mass_kg=engine.mass_kg * 0.25)
    _drum(node, "powertrain.stator", bp, CRANK_AXIS, r_motor, l_motor * 0.8, mass_kg=engine.mass_kg * 0.3)
    _drum(node, "powertrain.rotor_pack", bp, CRANK_AXIS, r_motor * 0.62, l_motor * 0.8, kind="rotating-mass",
          mass_kg=engine.mass_kg * 0.15, inertia_kg_m2=float(engine.inertia_kg_m2) * 0.8)
    edge("powertrain.rotor_to_shaft", "powertrain.rotor_pack", "powertrain.engine", "rigid-keyed-hub", radius=0.015)
    _drum(node, "powertrain.resolver", bp + CRANK_AXIS * (l_motor / 2.0 + 0.02), CRANK_AXIS, r_motor * 0.25, 0.03, mass_kg=0.3)
    edge("powertrain.resolver_to_shaft", "powertrain.resolver", "powertrain.engine", "rigid-keyed-hub", radius=0.006)
    # inverter on top (the front position engine_baker already names for
    # its switching whine), DC link and phase busbars to the stator
    inv_c = bp + UP * (r_motor * 1.1 + half * 0.35)
    node("powertrain.inverter", [float(v) for v in inv_c], "engine-block-component", mass_kg=engine.mass_kg * 0.12,
         body_half_extent_m=[l_motor * 0.45, half * 0.3, r_motor * 0.9])
    for ph in ("u", "v", "w"):
        edge(f"powertrain.phase_busbar_{ph}", "powertrain.inverter", "powertrain.stator", "insulated-copper-wire", radius=0.006, circuit_identity="hv")
    # coolant plate under the inverter, plumbed to the existing stack
    plate_c = inv_c - UP * (half * 0.3 + 0.008)
    node("powertrain.inverter_coolant_plate", [float(v) for v in plate_c], "engine-block-component", mass_kg=2.0,
         body_half_extent_m=[l_motor * 0.45, 0.008, r_motor * 0.9])
    if "powertrain.water_pump" in node_by_id:
        edge("powertrain.coolant_to_inverter_plate", "powertrain.water_pump", "powertrain.inverter_coolant_plate", "coolant-line", radius=0.008,
             circuit_identity="coolant", medium_rate_state="coolant-flow-and-temperature")
    # reduction: EV single-speed gearset + differential between motor and
    # the driveline chain the dyno couples through; servo planetary gearhead
    # with an output flange
    red_c = bp - CRANK_AXIS * (l_motor / 2.0 + half * 0.5)
    if servo:
        _drum(node, "powertrain.planetary_gearhead", red_c, CRANK_AXIS, r_motor * 0.9, half * 0.7, mass_kg=engine.mass_kg * 0.15, gearhead_kind="planetary")
        _drum(node, "powertrain.output_flange", red_c - CRANK_AXIS * (half * 0.5), CRANK_AXIS, r_motor * 0.7, 0.02, kind="rotating-mass", mass_kg=1.5)
        edge("powertrain.motor_to_gearhead", "powertrain.engine", "powertrain.planetary_gearhead", "geared-timing-drive", radius=0.01)
        edge("powertrain.gearhead_to_flange", "powertrain.planetary_gearhead", "powertrain.output_flange", "rigid-keyed-hub", radius=0.01)
    else:
        node("powertrain.reduction_gearset", [float(v) for v in red_c], "engine-block-component", mass_kg=engine.mass_kg * 0.12,
             body_half_extent_m=[half * 0.5, r_motor * 0.9, r_motor * 0.7], gearset_kind="single-speed-helical")
        edge("powertrain.motor_to_reduction", "powertrain.engine", "powertrain.reduction_gearset", "geared-timing-drive", radius=0.012)
        node("powertrain.drive_unit_differential", [float(v) for v in (red_c - UP * (r_motor * 0.6))], "engine-block-component",
             mass_kg=engine.mass_kg * 0.08, body_half_extent_m=[half * 0.45, r_motor * 0.5, r_motor * 0.9])
        edge("powertrain.reduction_to_differential", "powertrain.reduction_gearset", "powertrain.drive_unit_differential", "geared-timing-drive", radius=0.012)
    # HV switchgear at the battery: contactors, pre-charge resistor, DC-DC
    batt = node_by_id.get("electrical.battery")
    if batt is not None:
        bpp = _pos(batt)
        node("electrical.hv_contactor_box", [float(v) for v in (bpp + UP * 0.12)], "electrical-junction", mass_kg=3.0,
             body_half_extent_m=[0.08, 0.04, 0.10], contents=["main+ contactor", "main- contactor", "pre-charge relay + resistor", "HV interlock loop"])
        edge("electrical.battery_to_contactors", "electrical.battery", "electrical.hv_contactor_box", "insulated-copper-wire", radius=0.008, circuit_identity="hv")
        edge("electrical.contactors_to_inverter", "electrical.hv_contactor_box", "powertrain.inverter", "insulated-copper-wire", radius=0.008, circuit_identity="hv")
        node("electrical.dc_dc_converter", [float(v) for v in (bpp + UP * 0.12 + np.array([0.0, 0.0, 0.25]))], "electrical-junction", mass_kg=4.0,
             body_half_extent_m=[0.09, 0.04, 0.07])
        edge("electrical.contactors_to_dcdc", "electrical.hv_contactor_box", "electrical.dc_dc_converter", "insulated-copper-wire", radius=0.006, circuit_identity="hv")
        if "electrical.fusebox" in node_by_id:
            edge("electrical.dcdc_to_12v", "electrical.dc_dc_converter", "electrical.fusebox", "insulated-copper-wire", radius=0.006, circuit_identity="12v")


def _emit_expander_plant(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """What sits upstream of an expander's admission: for steam a boiler
    (barrel, firebox, steam dome, safety valve, water gauge, regulator,
    feedwater injector and tank, chimney with the blast nozzle in it);
    for compressed air the reservoir bank, its charging and check
    valves, gauge, and the inter-stage reheater. The valve gear gets a
    real node on the eccentric cylinder_ports already draws."""
    if engine.kind != "expander":
        return
    src = node_by_id.get("fuel.source")
    if src is None:
        return
    sp = _pos(src)
    spec = getattr(engine, "expander", None)
    bore = float(getattr(spec, "bore_m", 0.2))
    fluid = str(getattr(getattr(engine, "fuel_network", None), "fluid", "") or "").lower()
    steam = "steam" in fluid or "steam" in engine.identity
    if steam:
        barrel_r = bore * 2.2
        barrel_l = bore * 7.0
        bc = sp + np.array([-barrel_l * 0.5, barrel_r * 0.6, 0.0])
        _drum(node, "fuel.boiler_barrel", bc, CRANK_AXIS, barrel_r, barrel_l, mass_kg=engine.mass_kg * 0.35, fluid_volume_l=float(np.pi * barrel_r ** 2 * barrel_l * 700.0))
        node("fuel.firebox", [float(v) for v in (bc - CRANK_AXIS * (barrel_l * 0.55))], "engine-block-component", mass_kg=engine.mass_kg * 0.12,
             body_half_extent_m=[barrel_r * 0.6, barrel_r * 1.1, barrel_r * 1.0])
        edge("fuel.firebox_to_barrel", "fuel.firebox", "fuel.boiler_barrel", "rigid-bolted-joint", radius=0.02)
        _drum(node, "fuel.steam_dome", bc + UP * (barrel_r * 1.15), UP, barrel_r * 0.35, barrel_r * 0.5, mass_kg=25.0)
        edge("fuel.dome_to_barrel", "fuel.steam_dome", "fuel.boiler_barrel", "rigid-bolted-joint", radius=0.02)
        node("fuel.safety_valve", [float(v) for v in (bc + UP * (barrel_r * 1.5) + CRANK_AXIS * (barrel_r * 0.5))], "engine-block-port", port_kind="safety-valve",
             port_direction=[0.0, 1.0, 0.0], port_radius_m=0.02, fluid_role="steam", mating=False, connected=True, plugged=False, part="fuel.boiler_barrel", bung=True)
        node("fuel.water_gauge", [float(v) for v in (bc + CRANK_AXIS * (barrel_l * 0.48) + UP * (barrel_r * 0.3))], "engine-block-component", mass_kg=1.5,
             body_half_extent_m=[0.02, barrel_r * 0.35, 0.02])
        _drum(node, "fuel.regulator_valve", bc + UP * (barrel_r * 1.15) + CRANK_AXIS * (barrel_r * 0.4), CRANK_AXIS, 0.04, 0.10, mass_kg=6.0)
        # the dome feeds the regulator, the regulator feeds the existing
        # source/admission chain
        edge("fuel.dome_to_regulator", "fuel.steam_dome", "fuel.regulator_valve", "fuel-supply-line", radius=0.03, circuit_identity="fuel",
             medium_rate_state="fuel-flow-and-pressure")
        edge("fuel.regulator_to_source", "fuel.regulator_valve", "fuel.source", "fuel-supply-line", radius=0.03, circuit_identity="fuel",
             medium_rate_state="fuel-flow-and-pressure")
        _drum(node, "fuel.chimney", bc - CRANK_AXIS * (barrel_l * 0.45) + UP * (barrel_r * 2.2), UP, barrel_r * 0.3, barrel_r * 2.2, mass_kg=30.0, kind="exhaust-component")
        node("fuel.blast_nozzle", [float(v) for v in (bc - CRANK_AXIS * (barrel_l * 0.45) + UP * (barrel_r * 1.2))], "engine-block-port", port_kind="blast-nozzle",
             port_direction=[0.0, 1.0, 0.0], port_radius_m=0.02, fluid_role="steam", mating=False, connected=True, plugged=False, part="fuel.chimney", bung=True)
        for e in edges:
            if e["identity"].endswith(".exhaust_primary") or e["identity"].endswith(".to_downpipe"):
                pass
        node("fuel.water_tank", [float(v) for v in (sp + np.array([barrel_l * 0.35, -barrel_r * 0.4, barrel_r * 1.6]))], "high-pressure-canister",
             mass_kg=40.0, capacity_kg=300.0, chassis_side=True)
        _drum(node, "fuel.feedwater_injector", sp + np.array([barrel_l * 0.1, -barrel_r * 0.6, barrel_r * 1.2]), CRANK_AXIS, 0.03, 0.12, mass_kg=4.0)
        edge("fuel.tank_to_injector", "fuel.water_tank", "fuel.feedwater_injector", "coolant-line", radius=0.012, circuit_identity="feedwater",
             medium_rate_state="coolant-flow-and-temperature")
        edge("fuel.injector_to_barrel", "fuel.feedwater_injector", "fuel.boiler_barrel", "coolant-line", radius=0.012, circuit_identity="feedwater",
             medium_rate_state="coolant-flow-and-temperature")
    else:
        # compressed air: a bank of reservoirs is the loco's dominant part
        res_r = bore * 1.6
        res_l = bore * 6.0
        for i in range(2):
            rc = sp + np.array([-res_l * 0.5, res_r * 0.8 + i * (res_r * 2.2), 0.0])
            rid = f"fuel.air_reservoir_{i + 1}"
            _drum(node, rid, rc, CRANK_AXIS, res_r, res_l, mass_kg=engine.mass_kg * 0.15, kind="high-pressure-canister",
                  capacity_kg=float(getattr(src, "capacity_kg", 0.0) or 0.0) / 2.0)
            edge(f"{rid}_to_source", rid, "fuel.source", "fuel-supply-line", radius=0.02, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
        node("fuel.charging_valve", [float(v) for v in (sp + np.array([-res_l, res_r * 0.8, res_r * 1.1]))], "engine-block-port", port_kind="charging-connection",
             port_direction=[0.0, 0.0, 1.0], port_radius_m=0.02, fluid_role="compressed-air", mating=False, connected=False, plugged=True, part="fuel.air_reservoir_1", bung=True)
        node("fuel.check_valve", [float(v) for v in (sp + np.array([-res_l * 0.98, res_r * 0.8, res_r * 0.9]))], "engine-block-component", mass_kg=1.0,
             body_half_extent_m=[0.03, 0.03, 0.03])
        edge("fuel.charging_to_check", "fuel.charging_valve", "fuel.check_valve", "fuel-supply-line", radius=0.012, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
        edge("fuel.check_to_reservoir", "fuel.check_valve", "fuel.air_reservoir_1", "fuel-supply-line", radius=0.012, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
        node("fuel.pressure_gauge", [float(v) for v in (sp + np.array([0.0, res_r * 2.2, res_r * 0.5]))], "engine-block-component", mass_kg=0.5,
             drum_axis=[0.0, 0.0, 1.0], drum_radius_m=0.05, drum_length_m=0.03)
        # the inter-stage reheater: fights the adiabatic freeze-out a real
        # two-stage air loco suffers, between the regulator stages
        stage0 = next((i for i in node_by_id if i.startswith("fuel.stage_0_")), None)
        if stage0 is not None:
            _drum(node, "fuel.reheater", _pos(node_by_id[stage0]) + UP * (bore * 0.9), CRANK_AXIS, bore * 0.5, bore * 1.6, mass_kg=20.0, exchanger_kind="air-reheater")
            edge("fuel.stage0_to_reheater", stage0, "fuel.reheater", "fuel-supply-line", radius=0.015, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
    # valve gear: a real node on the eccentric the cylinder layout already
    # draws (an eccentric per cylinder), plus the reverser lever
    for g, _ports in layout:
        if getattr(g, "kind", "") != "expander":
            continue
        cc = np.array(g.crank_centre, dtype=np.float64)
        node(f"powertrain.cylinder_{g.number}.valve_gear_eccentric", [float(v) for v in cc], "rotating-mass", mass_kg=4.0,
             drawn_by="cylinder_ports:eccentric_shaft", gear_kind="stephenson-link")
        edge(f"powertrain.cylinder_{g.number}.eccentric_on_crank", "powertrain.engine", f"powertrain.cylinder_{g.number}.valve_gear_eccentric", "rigid-keyed-hub", radius=0.01)
    node("powertrain.reverser_lever", [float(v) for v in (sp + np.array([bore * 1.5, bore * 2.2, -bore * 1.5]))], "engine-block-component", mass_kg=3.0,
         body_half_extent_m=[0.015, bore * 1.2, 0.015])


def _emit_atmospheric_mechanism(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """The Otto-Langen's whole power take-off -- rack, pinion, freewheel
    -- plus the slide valve on its eccentric, the pilot burner at the
    flame port, and the captive-ball governor governor.py already
    simulates, on geometry cylinder_ports already draws."""
    if engine.kind != "atmospheric":
        return
    for g, ports in layout:
        if getattr(g, "kind", "") != "atmospheric":
            continue
        cc = np.array(g.crank_centre, dtype=np.float64)
        bore = float(g.bore_m)
        tag = f"powertrain.cylinder_{g.number}"
        node(f"{tag}.pinion", [float(v) for v in cc], "rotating-mass", mass_kg=6.0, drawn_by="cylinder_ports:pinion")
        node(f"{tag}.freewheel", [float(v) for v in (cc + CRANK_AXIS * (bore * 0.26))], "rotating-mass", mass_kg=9.0, drawn_by="cylinder_ports:freewheel_drum",
             clutch_kind="roller-freewheel")
        edge(f"{tag}.freewheel_to_shaft", f"{tag}.freewheel", "powertrain.engine", "rigid-keyed-hub", radius=0.012)
        edge(f"{tag}.pinion_to_freewheel", f"{tag}.pinion", f"{tag}.freewheel", "one-way-clutch", radius=0.012)
        # the rack is the piston rod's toothed extension: a real part, tall
        base = np.array(g.base, dtype=np.float64); axis = _unit(g.axis)
        rack_c = base + axis * (float(g.length_m) * 1.1)
        node(f"{tag}.rack", [float(v) for v in rack_c], "rotating-mass", mass_kg=float(getattr(engine.atmospheric, "piston_mass_kg", 10.0)) * 0.4 if getattr(engine, "atmospheric", None) else 5.0,
             body_half_extent_m=[bore * 0.06, float(g.length_m) * 0.55, bore * 0.06])
        edge(f"{tag}.rack_meshes_pinion", f"{tag}.rack", f"{tag}.pinion", "rack-and-pinion", radius=0.01)
        # slide valve + its eccentric on the shaft, the burner at the flame port
        flame = next((p for p in ports if getattr(p, "port_kind", "") == "flame-port" or "flame" in getattr(p, "name", "")), None)
        gas = next((p for p in ports if "gas" in getattr(p, "name", "")), None)
        anchor = np.array(gas.position if gas is not None else base, dtype=np.float64)
        node(f"{tag}.slide_valve", [float(v) for v in (anchor + np.array([0.0, 0.0, bore * 0.5]))], "engine-block-component", mass_kg=4.0,
             body_half_extent_m=[bore * 0.35, bore * 0.12, bore * 0.18], valve_kind="slide-valve")
        _drum(node, f"{tag}.valve_eccentric", cc + CRANK_AXIS * (-bore * 0.5), CRANK_AXIS, bore * 0.12, bore * 0.1, kind="rotating-mass", mass_kg=2.0)
        edge(f"{tag}.valve_eccentric_on_shaft", "powertrain.engine", f"{tag}.valve_eccentric", "rigid-keyed-hub", radius=0.008)
        edge(f"{tag}.eccentric_rod", f"{tag}.valve_eccentric", f"{tag}.slide_valve", "cosmetic-belt-wrap", radius=0.006)
        if flame is not None:
            fp = np.array(flame.position, dtype=np.float64)
            _drum(node, f"{tag}.pilot_burner", fp + np.array([0.0, 0.0, -bore * 0.3]), np.array([0.0, 0.0, 1.0]), 0.012, bore * 0.25, mass_kg=0.5)
            gm = "fuel.gas_main_connection" if "fuel.gas_main_connection" in node_by_id else ("fuel.gasholder" if "fuel.gasholder" in node_by_id else None)
            if gm:
                edge(f"{tag}.pilot_gas_line", gm, f"{tag}.pilot_burner", "fuel-supply-line", radius=0.004, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
    gov = getattr(engine, "atmospheric_governor", None)
    if gov is not None:
        cc0 = np.array(layout[0][0].crank_centre, dtype=np.float64) if layout else np.zeros(3)
        r_gov = float(getattr(gov, "r_max_m", 0.08)) * 1.3
        _drum(node, "powertrain.captive_ball_governor", cc0 + CRANK_AXIS * (-0.35) + UP * 0.15, UP, r_gov, 0.06, mass_kg=6.0,
              governor_kind="captive-ball", drive_ratio=float(getattr(gov, "drive_ratio", 1.0)))
        edge("powertrain.governor_belt", "powertrain.engine", "powertrain.captive_ball_governor", "cosmetic-belt-wrap", radius=0.005)


def _emit_small_engine_kit(engine, layout, nodes, edges, node, edge, node_by_id) -> None:
    """Combustion singles that are not cars: the trimmer's flywheel
    magneto (magnet ring + fan vanes on the flywheel), muffler with spark
    arrestor, on-engine tank with primer and pulse line, diaphragm carb;
    the hit-and-miss engine's hopper (a node on the drawn hopper) and
    igniter trip lever."""
    if engine.kind != "combustion" or engine.architecture.cylinders != 1:
        return
    fit = crank_end_fittings(layout)
    if fit is None:
        return
    bore = fit["bore_m"]
    two_stroke = engine.architecture.two_stroke
    mag = node_by_id.get("powertrain.magneto")
    if two_stroke and mag is not None:
        # the magneto IS the flywheel: coil against a magnet ring cast into
        # the flywheel, whose vanes are also the cooling fan
        flywheel = node_by_id.get("powertrain.flywheel_front") or node_by_id.get("powertrain.flywheel")
        if flywheel is not None:
            fw = _pos(flywheel)
            mag["reference_position"] = [float(fw[0] - bore * 0.2), float(fw[1] + fit["flywheel_radius_m"] * 1.05), float(fw[2])]
            mag["magneto_kind"] = "flywheel-magneto-coil"
            flywheel["magnet_ring"] = True
            flywheel["fan_vanes"] = True
            edge("powertrain.flywheel_magnet_ring_to_coil", flywheel["identity"], "powertrain.magneto", "magnetic-coupling", radius=0.004)
    if two_stroke:
        # muffler with spark-arrestor screen right on the exhaust port side
        coll = next((n for n in nodes if n.get("kind") == "exhaust-collector"), None)
        if coll is not None:
            cp = _pos(coll); d = _unit(coll.get("outlet_direction", [0.0, -1.0, 0.0]))
            node("powertrain.small_engine_muffler", [float(v) for v in (cp + d * (bore * 0.9))], "exhaust-component", mass_kg=0.35,
                 body_half_extent_m=[bore * 0.9, bore * 0.5, bore * 0.45], spark_arrestor=True)
            edge("powertrain.collector_to_small_muffler", coll["identity"], "powertrain.small_engine_muffler", "exhaust-flow-path", radius=0.01,
                 circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature")
        # the tank bolts to the engine (it is the handle's counterweight),
        # with a primer bulb and the carb's crankcase pulse line
        tank = node_by_id.get("fuel.tank")
        if tank is not None:
            tank["reference_position"] = [float(fit["flywheel_centre"][0] + bore * 0.4), float(fit["y0"] - bore * 1.6), float(fit["z0"])]
            tank["chassis_side"] = False
            tank["body_half_extent_m"] = [bore * 1.2, bore * 0.7, bore * 0.9]
            node("fuel.primer_bulb", [float(v) for v in (_pos(tank) + np.array([-bore * 1.25, bore * 0.3, 0.0]))], "engine-block-component", mass_kg=0.02,
                 drum_axis=[1.0, 0.0, 0.0], drum_radius_m=0.012, drum_length_m=0.02)
            edge("fuel.primer_to_tank", "fuel.primer_bulb", "fuel.tank", "fuel-supply-line", radius=0.002, circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
        bowl = node_by_id.get("powertrain.fuel_bowl")
        if bowl is not None:
            bowl["kind"] = "diaphragm-carburetor"
            bowl["carburetor_kind"] = "diaphragm-metering"
            breather = next((i for i in node_by_id if i.endswith("crankcase.breather")), None)
            if breather:
                edge("fuel.carb_pulse_line", breather, "powertrain.fuel_bowl", "low-pressure-air-line", radius=0.002, circuit_identity="crankcase",
                     medium_rate_state="intake-air-flow-and-temperature")
    if getattr(engine, "governor_mode", "") == "hit_and_miss":
        # hopper cooling: a node on the hopper cylinder_ports already draws
        for g, _p in layout:
            if getattr(g, "cooling", "") == "hopper":
                base = np.array(g.base, dtype=np.float64); axis = _unit(g.axis)
                hc = base + axis * (float(g.length_m) * 0.85)
                node(f"powertrain.cylinder_{g.number}.water_hopper", [float(v) for v in hc], "engine-block-component", mass_kg=25.0,
                     drawn_by="cylinder_ports:water_hopper", fluid_volume_l=float(np.pi * (g.bore_m * 1.4) ** 2 * g.length_m * 300.0))
        # low-tension igniter trip lever off the cam-driven side rod
        plug = next((i for i in node_by_id if i.endswith(".spark_plug")), None)
        if plug:
            pp = _pos(node_by_id[plug])
            node("powertrain.igniter_trip_lever", [float(v) for v in (pp + np.array([bore * 0.35, 0.0, bore * 0.25]))], "engine-block-component", mass_kg=0.6,
                 body_half_extent_m=[bore * 0.3, 0.01, 0.01])
            edge("powertrain.igniter_trip_to_plug", "powertrain.igniter_trip_lever", plug, "rigid-bolted-joint", radius=0.004)


def emit_universal_parts(engine, layout, nodes, edges, node, edge) -> None:
    """Called once per graph after dressing and headers, so every anchor
    node it hangs a part off already sits at its final real position."""
    node_by_id = {n["identity"]: n for n in nodes}
    if engine.kind != "combustion":
        # the non-piston kinds carry their own real parts, sized from
        # their own declared specs; none of the crank-engine bolt-ons
        # below apply to them
        _emit_turbine_gas_path(engine, nodes, edges, node, edge, node_by_id)
        _emit_electric_drive_unit(engine, nodes, edges, node, edge, node_by_id)
        _emit_expander_plant(engine, layout, nodes, edges, node, edge, node_by_id)
        _emit_atmospheric_mechanism(engine, layout, nodes, edges, node, edge, node_by_id)
        return
    _emit_exhaust_downstream(engine, nodes, edges, node, edge)
    _emit_crank_end_hardware(engine, layout, nodes, edges, node, edge, node_by_id)
    _emit_belt_drive(engine, nodes, edges, node, edge, node_by_id)
    if engine.architecture.has_poppet_valves and not engine.architecture.rotary:
        _emit_timing_drive(engine, nodes, edges, node, edge, node_by_id)
    _emit_coolant_plumbing(engine, nodes, edges, node, edge, node_by_id)
    _emit_emissions(engine, nodes, edges, node, edge, node_by_id)
    _emit_pneumatic_idle_assist(engine, nodes, edges, node, edge, node_by_id)
    _emit_compression_brake(engine, layout, nodes, edges, node, edge, node_by_id)
    _emit_bellhousing(engine, nodes, edges, node, edge, node_by_id)
    # forced induction -- after the crank-end hardware so the blower belt
    # can reach the damper, and after the exhaust chain so a turbine can
    # take its up-pipe off a real collector
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_supercharger(engine, nodes, edges, node, edge, node_by_id)
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_mechanical_injection(engine, nodes, edges, node, edge, node_by_id)
    _emit_turbo_hardware(engine, nodes, edges, node, edge, node_by_id)
    # after any forced-induction hat/scoop relocation has settled the
    # air_filter's own final on-block position, so a remote-box/snorkel
    # relocation always starts from wherever the filter actually ended
    # up, blown or not
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_remote_air_intake(engine, nodes, edges, node, edge, node_by_id)
    # aircraft / industrial / air-cooled families -- the aircraft drive
    # last of the driveline work (it removes the car's clutch/gearbox chain
    # and the bellhousing built above, so it must see them)
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_aircraft_drive(engine, layout, nodes, edges, node, edge, node_by_id)
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_governor(engine, layout, nodes, edges, node, edge, node_by_id)
    _emit_injection_pump(engine, layout, nodes, edges, node, edge, node_by_id)
    _emit_dry_sump_extras(engine, layout, nodes, edges, node, edge, node_by_id)
    _emit_air_cooling(engine, layout, nodes, edges, node, edge, node_by_id)
    # singles that are not cars: flywheel magneto, spark-arrestor muffler,
    # on-engine tank, diaphragm carb, hopper, igniter trip -- last, since
    # it relocates parts the stages above placed
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_small_engine_kit(engine, layout, nodes, edges, node, edge, node_by_id)
