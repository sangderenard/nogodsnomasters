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


# ---------------------------------------------------------------------
# forced induction: the blower on its manifold with the hat on top; each
# turbo as real housings with wastegate, blow-off and downpipe; the
# charge cooler and its pipes; mechanical injection's barrel valve and
# hat nozzles
# ---------------------------------------------------------------------

def _emit_supercharger(engine, nodes, edges, node, edge, node_by_id) -> None:
    fi = engine.forced_induction
    rotor = node_by_id.get("supercharger_rotor")
    plenum = node_by_id.get("powertrain.intake_plenum")
    if fi.kind != "supercharger" or rotor is None or plenum is None:
        return
    half = engine_geometry.block_half_yz_m(engine)
    pp = _pos(plenum)
    # the case is sized by the real declared rotor pack: lobe count sets
    # the rotor diameter class, boost fraction the case length (a 14-71
    # is longer than a 6-71; both real Roots sizes), belt ratio nothing
    # geometric -- disclosed proportions of the block's own half-width
    lobes = max(2, int(fi.lobe_count))
    r_rotor = half * (0.34 + 0.05 * (lobes - 2))
    case_half = np.array([half * (1.3 + 0.6 * min(fi.max_boost_frac, 1.5)), r_rotor * 1.15, r_rotor * 2.25])
    plate_c = pp + UP * (float(plenum.get("body_half_extent_m", [0, half * 0.3, 0])[1]) + 0.015)
    case_c = plate_c + UP * (0.015 + case_half[1])
    node("powertrain.blower_manifold", [float(v) for v in plate_c], "engine-block-component", mass_kg=engine.mass_kg * 0.03,
         body_half_extent_m=[float(case_half[0]), 0.015, float(case_half[2])])
    edge("powertrain.blower_manifold_to_plenum", "powertrain.blower_manifold", "powertrain.intake_plenum", "rigid-bolted-joint", radius=0.008)
    node("powertrain.blower_case", [float(v) for v in case_c], "engine-block-component", mass_kg=engine.mass_kg * 0.09,
         lobe_count=lobes, body_half_extent_m=[float(v) for v in case_half])
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
    # the charge cooler: one air-to-air core ahead of the engine, fed by
    # every compressor, feeding the plenum -- present whenever the intake
    # resolved to the piped placement (dressing) or any turbo exists
    x_min, _ = engine_geometry.crank_extent(engine_geometry.cylinder_sites(engine))
    cooler_c = np.array([x_min - half * 2.2, half * 0.9, 0.0])
    node("powertrain.charge_cooler", [float(v) for v in cooler_c], "engine-block-component", mass_kg=engine.mass_kg * 0.02,
         exchanger_kind="air-to-air-charge-cooler", body_half_extent_m=[half * 0.25, half * 1.1, half * 1.9])
    if plenum is not None:
        edge("powertrain.cold_side_charge_pipe", "powertrain.charge_cooler", "powertrain.intake_plenum", "low-pressure-air-line",
             radius=0.03, circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
    for t in turbos:
        tp = _pos(t)
        side = 1.0 if tp[2] >= 0 else -1.0
        base = t["identity"]
        # real housings around the production point mass: compressor
        # (cold, inboard toward the charge pipe) and turbine (hot,
        # outboard toward the header), the cartridge between them
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
        # hot side: the nearest collector feeds this turbine; a downpipe
        # leaves the turbine outlet rearward
        if collectors:
            coll = min(collectors, key=lambda c: abs(_pos(c)[2] - tp[2]) + abs(_pos(c)[0] - tp[0]) * 0.3)
            edge(f"{base}.up_pipe", coll["identity"], f"{base}.turbine_housing", "exhaust-flow-path", radius=0.028,
                 circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature")
        dp_end = tp + np.array([r_c * 0.9 + 0.30, -half * 0.6, side * half * 0.4])
        node(f"{base}.downpipe", [float(v) for v in dp_end], "exhaust-component", mass_kg=2.0, chassis_side=False,
             drum_axis=[float(v) for v in _unit(dp_end - (tp + np.array([r_c * 0.9, 0.0, 0.0])))], drum_radius_m=0.032, drum_length_m=0.30)
        edge(f"{base}.turbine_to_downpipe", f"{base}.turbine_housing", f"{base}.downpipe", "exhaust-flow-path", radius=0.032,
             circuit_identity="exhaust", medium_rate_state="exhaust-pulse-pressure-and-temperature")
        # cold side: compressor outlet to the charge cooler
        edge(f"{base}.hot_side_charge_pipe", f"{base}.compressor_housing", "powertrain.charge_cooler", "low-pressure-air-line",
             radius=0.028, circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")


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
    # forced induction -- after the crank-end hardware so the blower belt
    # can reach the damper, and after the exhaust chain so a turbine can
    # take its up-pipe off a real collector
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_supercharger(engine, nodes, edges, node, edge, node_by_id)
    node_by_id = {n["identity"]: n for n in nodes}
    _emit_mechanical_injection(engine, nodes, edges, node, edge, node_by_id)
    _emit_turbo_hardware(engine, nodes, edges, node, edge, node_by_id)
