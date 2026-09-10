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
                 body_half_extent_m=([seg.length_m / 2.0, r * (2.2 if can else 1.0), r * (2.2 if can else 1.0)]
                                     if abs(run_dir[0]) > 0.5 else
                                     [r * (2.2 if can else 1.0), seg.length_m / 2.0, r * (2.2 if can else 1.0)]),
                 drum_axis=None if can else [float(v) for v in run_dir],
                 drum_radius_m=None if can else r, drum_length_m=None if can else seg.length_m)
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

def _emit_timing_drive(engine, nodes, edges, node, edge, node_by_id) -> None:
    cam = node_by_id.get("powertrain.camshaft")
    if cam is None:
        return
    face_x = _block_front_face_x(engine, nodes)
    half = engine_geometry.block_half_yz_m(engine)
    cam_p = _pos(cam)
    # the cover bolts flush to the block's real front face; the sprockets
    # and their run sit just inside it
    cover_c = np.array([face_x - 0.012, cam_p[1] * 0.5, 0.0])
    node("powertrain.timing_cover", [float(v) for v in cover_c], "engine-block-component", mass_kg=1.5,
         body_half_extent_m=[0.012, max(abs(cam_p[1]) * 0.5 + half * 0.3, half * 0.6), half * 0.7])
    edge("powertrain.timing_cover_to_block", "powertrain.timing_cover", "powertrain.engine", "rigid-bolted-joint", radius=0.006)
    sx = face_x - 0.006
    node("powertrain.timing_drive.crank_sprocket", [float(sx), 0.0, 0.0], "rotating-mass", mass_kg=0.3,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=half * 0.22, drum_length_m=0.012)
    node("powertrain.timing_drive.cam_sprocket", [float(sx), float(cam_p[1]), float(cam_p[2])], "rotating-mass", mass_kg=0.4,
         drum_axis=[1.0, 0.0, 0.0], drum_radius_m=half * 0.44, drum_length_m=0.012)
    edge("powertrain.timing_crank_sprocket_hub", "powertrain.engine", "powertrain.timing_drive.crank_sprocket", "rigid-keyed-hub", radius=0.006)
    edge("powertrain.timing_cam_sprocket_hub", "powertrain.camshaft", "powertrain.timing_drive.cam_sprocket", "rigid-keyed-hub", radius=0.006)
    edge("powertrain.timing_chain_run", "powertrain.timing_drive.crank_sprocket", "powertrain.timing_drive.cam_sprocket",
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
    node("powertrain.coolant_expansion_bottle", [float(v) for v in bottle], "engine-block-component", mass_kg=0.8,
         fluid_volume_l=1.5, body_half_extent_m=[half * 0.35, half * 0.5, half * 0.35])
    edge("powertrain.expansion_bottle_overflow", "powertrain.radiator", "powertrain.coolant_expansion_bottle", "coolant-line",
         radius=0.005, circuit_identity="coolant", medium_rate_state="coolant-flow-and-temperature")
    # heater core: a small exchanger tapped off the head outlet, returning
    # to the pump suction -- present on any liquid-cooled road engine
    core = tp + np.array([0.0, half * 0.6, -half * 2.2])
    node("powertrain.heater_core", [float(v) for v in core], "engine-block-component", mass_kg=1.2,
         exchanger_kind="fin-and-tube-heater-core", body_half_extent_m=[half * 0.5, half * 0.35, half * 0.12])
    edge("powertrain.heater_supply_hose", "powertrain.coolant_thermostat", "powertrain.heater_core", "coolant-line",
         radius=0.008, circuit_identity="coolant", medium_rate_state="coolant-flow-and-temperature")
    pump = node_by_id.get("powertrain.water_pump")
    if pump is not None:
        edge("powertrain.heater_return_hose", "powertrain.heater_core", "powertrain.water_pump", "coolant-line",
             radius=0.008, circuit_identity="coolant", medium_rate_state="coolant-flow-and-temperature")


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
    trans = node_by_id.get("powertrain.transmission")
    if rear is None or clutch is None or trans is None:
        return
    a, b = _pos(rear), _pos(trans)
    mid = (a + b) / 2.0
    half = engine_geometry.block_half_yz_m(engine)
    node("powertrain.bellhousing", [float(v) for v in mid], "engine-block-component", mass_kg=engine.mass_kg * 0.04,
         body_half_extent_m=[max(abs(b[0] - a[0]) / 2.0, 0.03), half * 0.9, half * 0.9])
    edge("powertrain.bellhousing_to_block", "powertrain.engine", "powertrain.bellhousing", "rigid-bolted-joint", radius=0.008)
    edge("powertrain.bellhousing_to_transmission", "powertrain.bellhousing", "powertrain.transmission", "rigid-bolted-joint", radius=0.008)


def emit_universal_parts(engine, layout, nodes, edges, node, edge) -> None:
    """Called once per graph after dressing and headers, so every anchor
    node it hangs a part off already sits at its final real position."""
    if engine.kind not in ("combustion",):
        return   # electric/servo/turbine/atmospheric/expander have their own real parts (later stages)
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_exhaust_downstream(engine, nodes, edges, node, edge)
    _emit_crank_end_hardware(engine, layout, nodes, edges, node, edge, node_by_id)
    _emit_belt_drive(engine, nodes, edges, node, edge, node_by_id)
    if engine.architecture.has_poppet_valves and not engine.architecture.rotary:
        _emit_timing_drive(engine, nodes, edges, node, edge, node_by_id)
    _emit_coolant_plumbing(engine, nodes, edges, node, edge, node_by_id)
    _emit_emissions(engine, nodes, edges, node, edge, node_by_id)
    _emit_bellhousing(engine, nodes, edges, node, edge, node_by_id)
