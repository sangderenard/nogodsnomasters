"""Engine DRESSING: the rest of the engine that hangs on the castings
-- lubrication (wet sump, dry-sump reserve tank and scavenge, splash
bath, pump, spin-on filter), air and fuel filtration (paper/foam/
gauze element, oil-bath cleaner, sediment bowl, inline filter, diesel
water separator), the fuel rail (port EFI rail, common rail), the
ignition wiring (distributor and coil with plug leads, coil-on-plug
packs, magneto on the crank nose, glow-plug harness) and the intake
side where it applies (plenum log with procedurally routed runners
into each intake port, a central throttle body or a carburetor on a
riser, or individual throttle bodies with stacks; nothing for a
diesel, which meters fuel, not air).

Everything is DERIVED from what the catalogue already declares
(`derive_dressing`): ignition_profile picks distributor / magneto /
glow; carburetor.is_carbureted, injector and compression_ignition
pick carburetor / rail / common rail; intake_system.filter_material
picks the cleaner (velocity_stack = individual throttle bodies with
open stacks; an antique slow single gets an oil-bath cleaner); race
fuels get a dry sump; an antique single gets a splash bath. Every
choice is a real installation, and every one is a graph node/edge
the existing mesh, circuit and part systems already understand:
oil parts join the oil circuit, air parts the intake circuit, leads
run to the real spark-plug/glow-plug bosses the cylinder layout put
on the head, the rail feeds the real injector bosses.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from exhaust_header import _quarter_bend, BEND_RADIUS_FRAC_OF_DIAMETER

CRANK_AXIS = np.array([1.0, 0.0, 0.0])


def _unit(v):
    a = np.array(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class DressingSpec:
    lube: str          # "wet-sump" | "dry-sump" | "splash-bath" | "drip"
    oil_filter: str    # "spin-on" | "screen" | "none"
    air_filter: str    # "paper" | "foam" | "cotton_gauze" | "oil-bath" | "stacks" | "none"
    fuel_filter: str   # "inline" | "sediment-bowl" | "water-separator" | "none"
    rail: str          # "port-rail" | "common-rail" | "none"
    ignition: str      # "distributor" | "coil-on-plug" | "magneto" | "glow" | "flame" | "none"
    throttle: str      # "central" | "carburetor" | "individual" | "hat" (injector hat on a blower) | "none"


def derive_dressing(engine) -> DressingSpec:
    arch = engine.architecture
    kind = engine.kind
    layout_name = arch.layout.lower()
    antique_single = arch.cylinders == 1 and engine.redline_rpm < 1000.0
    race_fuel = engine.preferred_fuel_profile in ("nitromethane-race", "methanol-race")
    ci = bool(getattr(engine, "compression_ignition", False))
    carb = getattr(engine, "carburetor", None)
    carbureted = bool(carb is not None and carb.is_carbureted) and not ci
    if kind in ("expander", "atmospheric"):
        return DressingSpec("drip", "none", "none", "none", "none", "flame" if kind == "atmospheric" else "none", "none")
    if kind in ("electric", "servo-electric", "turbine"):
        return DressingSpec("wet-sump" if kind == "turbine" else "none", "none", "none", "none", "none", "none", "none")
    declared = getattr(engine, "lubrication", "auto")
    lube = declared if declared != "auto" else ("splash-bath" if antique_single else ("dry-sump" if race_fuel else "wet-sump"))
    oil_filter = "screen" if antique_single else "spin-on"
    fm = engine.intake_system.filter_material
    air_filter = "stacks" if fm == "velocity_stack" else ("oil-bath" if (antique_single or "hit-and-miss" in layout_name) else fm)
    fuel_filter = "water-separator" if ci else ("sediment-bowl" if (carbureted and antique_single) else "inline")
    rail = "common-rail" if ci else ("none" if carbureted else "port-rail")
    prof = getattr(engine, "ignition_profile", "gasoline-distributor")
    if arch.rotary:
        ignition = "coil-on-plug"
    elif ci:
        ignition = "glow"
    elif "magneto" in prof:
        ignition = "magneto"
    elif "coil" in prof:
        # a declared coil-pack / coil-on-plug system -- no distributor
        ignition = "coil-on-plug"
    elif fm == "velocity_stack" or (engine.redline_rpm >= 9000.0 and not carbureted):
        ignition = "coil-on-plug"
    else:
        ignition = "distributor"
    # a blown engine with open stacks is not ITBs on the ports: the
    # throttle is the injector HAT (bugcatcher) on top of the blower,
    # butterflies in the hat, nozzles in the hat and at the ports
    blown = engine.forced_induction.kind == "supercharger"
    throttle = ("none" if ci else "hat" if (blown and fm == "velocity_stack")
                else "individual" if fm == "velocity_stack" else ("carburetor" if carbureted else "central"))
    return DressingSpec(lube, oil_filter, air_filter, fuel_filter, rail, ignition, throttle)


# ---------------------------------------------------------------------
# Intake side: plenum log + runners, throttle / carburetor / ITBs
# ---------------------------------------------------------------------

def _intake_ports(layout):
    out = []
    for g, ports in layout:
        if g.kind not in ("spark-piston", "compression-piston"):
            continue
        for p in ports:
            if p.port_kind == "intake-valve-port":
                out.append((g, p))
    return out


@dataclass(frozen=True)
class IntakeHardware:
    """The intake resolved to real, distinct hardware facts -- what one
    "barrel count" used to conflate. units are SPATIALLY separate carbs/
    throttle bodies (a 2- or 4-barrel carb is ONE unit; dual quads two;
    ITBs one per port). barrels_per_unit is how many bores share that one
    casting. planes is how many firing-order groups share a DIVIDED
    plenum under one unit (dual-plane = 2) -- co-located, never spread to
    each group's own centroid. placement is where the chamber sits."""
    units: int
    barrels_per_unit: int
    planes: int
    placement: str        # "valley" | "inboard" | "piped"
    label: str


def derive_intake_hardware(engine, spec: DressingSpec, n_ports: int, n_banks: int) -> IntakeHardware:
    """Declared on IntakeSystem where the catalogue declares it; every
    None resolves to the typical real build for this engine's own
    declared architecture, throttle assembly and forced induction."""
    intake = engine.intake_system
    fi = engine.forced_induction
    cyl = max(1, engine.architecture.cylinders)

    if n_ports == 0:
        return IntakeHardware(0, 0, 1, "inboard", "none")

    # placement: the valley only exists between the banks of a V/flat
    # engine; a straight engine's manifold sits inboard above its head;
    # a turbo's compressor outlet is always plumbed to the plenum by a
    # real pipe. A supercharger keeps the on-block placement unless the
    # build declares "piped" (a remote/centrifugal blower).
    placement = intake.plenum_placement
    if placement == "auto":
        placement = "piped" if fi.kind == "turbo" else ("valley" if n_banks >= 2 else "inboard")

    if spec.throttle == "none":
        # a diesel still has a manifold: one open inlet, no butterfly
        return IntakeHardware(intake.inlet_units or 1, 1, intake.plenum_planes or 1, placement, "unthrottled manifold")
    if spec.throttle == "individual":
        return IntakeHardware(n_ports, 1, 1, placement, "individual throttle bodies")
    if spec.throttle == "hat":
        # one unit: the hat on top of the blower; its barrels are the hat's
        # butterflies (a declared assembly if any, else the classic pair);
        # one open plenum -- the blower case IS the manifold
        tb = getattr(engine, "throttle_body", None)
        barrels = len(tb.barrels) if (tb is not None and getattr(tb, "barrels", None)) else 2
        return IntakeHardware(intake.inlet_units or 1, barrels, 1, placement, "injector hat over blower")

    tb = getattr(engine, "throttle_body", None)
    declared_barrels = len(tb.barrels) if (tb is not None and getattr(tb, "barrels", None)) else None

    if spec.throttle == "carburetor":
        if declared_barrels is not None:
            # a real 8-barrel declaration is two 4-barrel carbs (dual quads),
            # never one casting with eight bores
            units = intake.inlet_units or (2 if declared_barrels >= 8 else 1)
            barrels = max(1, declared_barrels // units)
        elif cyl == 1:
            units, barrels = intake.inlet_units or 1, 1
        elif cyl <= 6:
            units, barrels = intake.inlet_units or 1, 2
        elif engine.preferred_fuel_profile in ("nitromethane-race", "methanol-race"):
            units, barrels = intake.inlet_units or 2, 4
        else:
            units, barrels = intake.inlet_units or 1, 4
        # dual-plane is the classic carbureted manifold on 4..8 cylinders;
        # a single is one plane by construction
        planes = intake.plenum_planes or (2 if 4 <= cyl <= 8 else 1)
        total = units * barrels
        label = ("double quad" if (units == 2 and barrels == 4)
                 else f"single {barrels}-barrel carburetor" if units == 1 and barrels > 1
                 else "single carburetor" if units == 1
                 else f"{units}x {barrels}-barrel carburetors")
        if planes > 1:
            label += " (dual-plane)" if planes == 2 else f" ({planes}-plane)"
        return IntakeHardware(units, barrels, planes, placement, label)

    # central EFI throttle body: one unit, one open plenum unless declared
    units = intake.inlet_units or 1
    barrels = declared_barrels // units if declared_barrels else 1
    planes = intake.plenum_planes or 1
    return IntakeHardware(units, max(1, barrels), planes, placement,
                          "single throttle body" if units == 1 else f"{units} throttle bodies")


def intake_inlet_count(engine, spec: DressingSpec, n_ports: int) -> tuple[int, str]:
    """Kept for callers that only want the unit count and a label."""
    hw = derive_intake_hardware(engine, spec, n_ports, n_banks=1)
    return hw.units, hw.label


def distribute_ports(layout, ports, n_inlets: int) -> list[int]:
    """Group index per port: consecutive cylinders in the FIRING order
    go to different inlets, so every inlet sees evenly spaced pulses
    -- the real dual-plane / balanced-manifold rule. 1:1 when the
    inlet count equals the port count."""
    if n_inlets >= len(ports):
        return list(range(len(ports)))
    order = sorted({g.number: g.throw_angle_deg for g, _ in layout}.items(), key=lambda kv: kv[1])
    slot = {cyl: k for k, (cyl, _) in enumerate(order)}
    return [slot.get(g.number, k) % n_inlets for k, (g, _) in enumerate(ports)]


def plan_intake(layout, spec: DressingSpec, runner_radius_m: float, engine=None):
    """The intake as a DISTRIBUTOR: N intake ports gathered into the
    chambers of M spatially distinct inlet UNITS (the mirror of the
    exhaust collector). A unit is one real casting -- a carburetor or
    throttle body with however many barrels it has side by side -- so a
    straight six with a two-barrel is 6-to-1: ONE centre chamber, ONE
    carb, two bores a few cm apart on it. Dual quads are 6/8-to-2. ITBs
    are N-to-N with the body right on the short runner.

    Within a unit the plenum may be DIVIDED into planes: cylinders that
    fire consecutively go to different planes (distribute_ports), so each
    half sees evenly spaced pulses -- the real dual-plane manifold. The
    planes are halves of the same chamber under the same carb, tagged on
    each runner and offset to their own half of the chamber; they are
    never separate chambers at each group's own centroid (that is what
    put two "throttle bodies" 35 cm apart on a two-barrel straight six).

    Placement: the chamber sits in the valley of a V/flat engine, or
    inboard above the head on a straight one; "piped" keeps the chamber
    where the runners meet but feeds it through a real pipe from the
    compressor outlet instead of a throttle body sitting on top."""
    from head_mesh import banks
    ports = _intake_ports(layout)
    if not ports:
        return None
    bank_list = banks(layout)
    axes = [_unit(gs[0].axis) for gs in bank_list]
    mean_up = _unit(sum(axes))
    bore = max(g.bore_m for g, _ in ports)
    bend_r = 2.0 * runner_radius_m * BEND_RADIUS_FRAC_OF_DIAMETER
    hw = (derive_intake_hardware(engine, spec, len(ports), len(bank_list)) if engine is not None
          else IntakeHardware(1, 1, 1, "valley" if len(bank_list) >= 2 else "inboard", "single"))
    n_units = max(1, min(hw.units, len(ports)))
    n_planes = max(1, hw.planes)
    unit_of = distribute_ports(layout, ports, n_units)
    positions = [np.array(p.position) for _, p in ports]
    centre = sum(positions) / len(positions)
    if len(bank_list) >= 2:
        rail_level = centre + mean_up * (bore * 1.1)
    else:
        inboard = _unit(sum(np.array(p.direction) for _, p in ports))
        inboard = inboard - mean_up * float(np.dot(inboard, mean_up))
        inboard = _unit(inboard) if np.linalg.norm(inboard) > 1e-6 else -_unit(np.cross(CRANK_AXIS, mean_up))
        rail_level = centre + inboard * (bore * 0.75) + mean_up * (bore * 0.8)
    individual = hw.label == "individual throttle bodies" and n_units == len(ports) and len(ports) > 1

    # plane within each unit: the same firing-order rule applied to that
    # unit's own members, so a dual-plane half never sees two consecutive
    # fires either
    plane_of = [0] * len(ports)
    for ui in range(n_units):
        members = [k for k, u in enumerate(unit_of) if u == ui]
        if n_planes > 1 and len(members) > 1:
            sub_layout = [(g, ps) for (g, ps) in layout if any(g is ports[k][0] for k in members)]
            sub_ports = [ports[k] for k in members]
            for k, pl in zip(members, distribute_ports(sub_layout, sub_ports, min(n_planes, len(members)))):
                plane_of[k] = pl

    chambers = []
    for ui in range(n_units):
        members = [k for k, u in enumerate(unit_of) if u == ui]
        xs = [positions[k][0] for k in members]
        c = rail_level.copy(); c[0] = float(np.mean(xs))
        # a divided plenum is one chamber, long enough to seat every
        # barrel side by side and both planes' runner joins
        span = (max(xs) - min(xs)) / 2.0 if len(members) > 1 else 0.0
        barrel_pitch = bore * 0.42 * 2.2
        half_len = max(bore * 0.45, span * 0.35, hw.barrels_per_unit * barrel_pitch / 2.0)
        chambers.append({"centre": c, "half_len": half_len, "radius": bore * 0.42, "ports": members,
                         "barrels": hw.barrels_per_unit, "planes": min(n_planes, max(1, len(members)))})
    plan = {"chambers": chambers, "runners": [], "itbs": [], "up": mean_up, "n_inlets": n_units,
            "label": hw.label, "plenum_radius": bore * 0.42, "individual": individual,
            "placement": hw.placement, "barrels_per_unit": hw.barrels_per_unit, "planes": n_planes}
    for k, (g, p) in enumerate(ports):
        pos = np.array(p.position); d = _unit(p.direction)
        pts = [pos, pos + d * (p.radius_m * 1.4)]
        if individual:
            body = pts[-1] + d * (bore * 0.35)
            pts.append(body)
            plan["itbs"].append({"cylinder": g.number, "port": p.name, "body": body, "direction": d,
                                 "stack_end": body + d * (bore * 0.45)})
        else:
            ch = chambers[unit_of[k]]
            # a plane's runners join their own half of the divided chamber:
            # plane 0 toward the front end, plane 1 toward the rear, etc.
            n_pl = ch["planes"]
            plane_shift = ((plane_of[k] + 0.5) / n_pl - 0.5) * (ch["half_len"] * 1.2) if n_pl > 1 else 0.0
            target = ch["centre"].copy(); target[0] = pos[0]
            to_t = _unit(target - pts[-1])
            pts.extend(_quarter_bend(pts[-1], d, to_t, bend_r))
            join_x = ch["centre"][0] + plane_shift
            if abs(pos[0] - join_x) > ch["half_len"] * 0.9:
                run_dir = CRANK_AXIS * (1.0 if join_x > pos[0] else -1.0)
                pts.append(target - to_t * (ch["radius"] * 1.2))
                pts.extend(_quarter_bend(pts[-1], to_t, run_dir, bend_r))
                join = pts[-1].copy(); join[0] = join_x - run_dir[0] * (ch["half_len"] * 0.9)
                pts.append(join)
                pts.extend(_quarter_bend(pts[-1], run_dir, to_t, bend_r))
                end = ch["centre"] - run_dir * (ch["half_len"] * 0.6) - to_t * (ch["radius"] * 0.3)
                end[0] = join_x - run_dir[0] * (ch["half_len"] * 0.3)
                pts.append(end)
            else:
                end = target - to_t * (ch["radius"] * 0.6)
                end[0] = join_x if n_pl > 1 else end[0]
                pts.append(end)
        plan["runners"].append({"cylinder": g.number, "port": p.name, "points": pts,
                                "group": unit_of[k], "plane": plane_of[k]})
    return plan


# ---------------------------------------------------------------------
# Graph emission
# ---------------------------------------------------------------------

def emit_dressing_graph(engine, layout, spec: DressingSpec, nodes, edges, node, edge, runner_radius_m: float) -> dict:
    """Everything as nodes/edges on the existing graph, reusing the
    production nodes that already exist (plenum, throttle body, fuel
    rail/bowl, oil pump, oil pan, ignition driver) by moving them to
    where the dressing actually puts them."""
    by_id = {n["identity"]: n for n in nodes}
    report = {"spec": spec}
    stations_bore = max((g.bore_m for g, _ in layout), default=0.08)
    from crank_mesh import crank_stations
    stations = crank_stations(layout)
    x0 = stations[0][0] if stations else 0.0
    x1 = stations[-1][0] if stations else 0.0
    y0 = float(stations[0][1][0].crank_centre[1]) if stations else 0.0
    bore = stations_bore

    # ---- intake: N ports -> M inlet chambers ----
    plan = plan_intake(layout, spec, runner_radius_m, engine) if _intake_ports(layout) else None
    if plan is not None:
        plenum = by_id.get("powertrain.intake_plenum")
        tb = by_id.get("powertrain.throttle_body")
        bowl = by_id.get("powertrain.fuel_bowl")
        chamber_ids = []
        placement = plan.get("placement", "inboard")
        barrels = int(plan.get("barrels_per_unit", 1))
        # a real pipe run needs a real source point: the compressor outlet
        # -- the same front-of-block reference the supercharger rotor is
        # placed from in drivetrain_graph.py (a turbo has no rotor node
        # yet; its compressor outlet is declared at that same reference
        # until a real turbo placement exists)
        compressor_outlet = None
        if placement == "piped":
            import engine_geometry
            f_ = np.array(engine_geometry.accessory_front_point(engine), dtype=np.float64)
            compressor_outlet = f_ + np.array([0.0, 0.06, 0.0]) + plan["up"] * (bore * 0.4)

        def _bung(identity: str, part: str, pos, direction, radius_m: float, port_kind: str, fluid: str) -> None:
            # a real, default-PLUGGED casting port in exactly the vocabulary
            # assembly_ports.emit_ports_graph already uses -- something a
            # sensor, injector or vacuum line can later plumb into, drawn
            # as a stub like every other casting port
            node(identity, [float(v) for v in pos], "engine-block-port", port_kind=port_kind,
                 port_direction=[float(v) for v in _unit(direction)], port_radius_m=radius_m,
                 fluid_role=fluid, mating=False, connected=False, plugged=True, part=part, bung=True)

        if not plan["individual"]:
            for gi, ch in enumerate(plan["chambers"]):
                cid = "powertrain.intake_plenum" if gi == 0 else f"powertrain.intake_plenum_{gi + 1}"
                n_ = plenum if (gi == 0 and plenum is not None) else None
                if n_ is None:
                    node(cid, [float(v) for v in ch["centre"]], "engine-block-component", mass_kg=2.5)
                    n_ = nodes[-1]
                n_["reference_position"] = [float(v) for v in ch["centre"]]
                n_["body_half_extent_m"] = [ch["half_len"], ch["radius"] * 0.7, ch["radius"]]
                n_["plenum_style"] = f"chamber {gi + 1}/{plan['n_inlets']}"
                n_["inlet_group"] = gi
                n_["plenum_planes"] = ch["planes"]
                n_["plenum_placement"] = placement
                chamber_ids.append(cid)
                # default-plugged sensor/vacuum bungs on the chamber: a MAP/
                # vacuum tap on the end wall and an IAT boss on the roof
                side = _unit(np.cross(plan["up"], CRANK_AXIS))
                _bung(f"{cid}.map_tap", cid, ch["centre"] + CRANK_AXIS * ch["half_len"], CRANK_AXIS,
                      0.004, "vacuum-tap", "intake-air")
                _bung(f"{cid}.iat_boss", cid, ch["centre"] + side * ch["radius"], side,
                      0.006, "sensor-boss", "intake-air")
                # the inlet on this chamber: ONE unit -- a carburetor / throttle
                # body casting -- with its barrels side by side on it
                inlet_pos = ch["centre"] + plan["up"] * (ch["radius"] * 1.3)
                tid = "powertrain.throttle_body" if gi == 0 else f"powertrain.throttle_body_{gi + 1}"
                t_ = tb if (gi == 0 and tb is not None) else None
                if t_ is None:
                    node(tid, [float(v) for v in inlet_pos], "engine-block-component", mass_kg=1.0)
                    t_ = nodes[-1]
                if compressor_outlet is not None:
                    # piped: the throttle/inlet sits at the compressor outlet
                    # and a real pipe carries the charge to the chamber
                    inlet_pos = compressor_outlet
                t_["reference_position"] = [float(v) for v in inlet_pos]
                barrel_pitch = ch["radius"] * 0.85 * 2.2
                t_["body_half_extent_m"] = [max(ch["radius"] * 0.6, barrels * barrel_pitch / 2.0),
                                            ch["radius"] * 0.55, ch["radius"] * 0.6]
                t_["barrels"] = barrels
                t_["barrel_pitch_m"] = barrel_pitch
                # a default-plugged vacuum tap at the throttle base (ported
                # vacuum -- what a distributor advance or PCV line taps)
                _bung(f"{tid}.vacuum_tap", tid, inlet_pos - plan["up"] * (ch["radius"] * 0.45) + side * (ch["radius"] * 0.6),
                      side, 0.004, "vacuum-tap", "intake-air")
                # the real bore/flow axis through this barrel -- metadata
                # only (not a new node/edge), read by vehicle_mesh.py to
                # build a real rotating butterfly-plate part live off
                # this same node instead of adding more graph topology
                # for what is purely a cosmetic sub-feature of the
                # throttle body that's already here.
                t_["flow_axis"] = [float(v) for v in plan["up"]]
                # bigger than the housing box's own half-extent
                # (radius*0.6/0.55/0.6 above) on purpose: a plate/lever
                # sized to match or sit inside the housing is fully
                # buried in its own opaque box from every angle --
                # never actually visible regardless of throttle
                # position. A real external lever does stick out past
                # the housing to reach its cable/rod anyway.
                t_["plate_radius_m"] = ch["radius"] * 0.85
                t_["inlet_kind"] = ("carburetor" if spec.throttle == "carburetor"
                                    else "open-inlet-elbow" if spec.throttle == "none" else "throttle-body")
                # every inlet needs this edge, including the first
                # (gi == 0): without it, that chamber's throttle body/
                # carburetor/inlet sits right on top of its own runner
                # union with no fluid-circuit connection down into it at
                # all -- the one visibly "floating," unwired inlet a
                # multi-chamber engine would otherwise always show.
                edge(f"{tid}_to_plenum", tid, cid, "low-pressure-air-line", radius=ch["radius"] * 0.7,
                     circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
            if bowl is not None and spec.throttle == "carburetor":
                c0 = plan["chambers"][0]["centre"]
                bowl["reference_position"] = [float(v) for v in (c0 + plan["up"] * (bore * 0.55) + np.array([bore * 0.45, 0.0, 0.0]))]
            # one air cleaner over the inlets (spanning them along the bank)
            # -- or, piped, at the compressor's own inlet: the filter sits
            # where air actually enters, which is no longer over the chamber
            xs = [ch["centre"][0] for ch in plan["chambers"]]
            if compressor_outlet is not None:
                fc = compressor_outlet + plan["up"] * (bore * 0.9)
                xs = [float(compressor_outlet[0])]
            else:
                fc = sum(ch["centre"] for ch in plan["chambers"]) / len(plan["chambers"]) + plan["up"] * (bore * 1.3)
            span = (max(xs) - min(xs)) / 2.0 + bore * 0.9
            node("powertrain.air_filter", [float(v) for v in fc], "air-filter", filter=spec.air_filter, mass_kg=1.5,
                 # a filter blocks because it is working: what it catches stays on it
                 # real dirt-holding capacity, in the mode's own units: an air
                 # filter sees a couple of hundred cubic metres of air an HOUR,
                 # so a service life of a few hundred hours is tens of thousands
                 # of cubic metres -- four orders off an oil filter
                 fouling_mode="filter-cake", blocked_frac=0.0, fouling_capacity=60_000.0,
                 body_half_extent_m=[span, bore * 0.22, bore * 0.9] if len(xs) > 1 else None,
                 drum_axis=None if len(xs) > 1 else [float(v) for v in plan["up"]],
                 drum_radius_m=bore * 1.0, drum_length_m=bore * 0.45)
            for gi in range(plan["n_inlets"]):
                tid = "powertrain.throttle_body" if gi == 0 else f"powertrain.throttle_body_{gi + 1}"
                edge(f"powertrain.air_filter_to_{tid.split('.')[-1]}", "powertrain.air_filter", tid, "low-pressure-air-line",
                     radius=plan["plenum_radius"] * 0.6, circuit_identity="intake-air",
                     medium_rate_state="intake-air-flow-and-temperature")
        # runners: re-route the production runner edges through bend waypoints
        runner_edges = {e["b"]: e for e in edges if e["identity"].endswith(".intake_runner")}
        tower_edges = [e for e in edges if e["identity"].endswith(".intake_runner_to_tower")]
        for r in plan["runners"]:
            port_id = f"powertrain.cylinder_{r['cylinder']}.{r['port']}"
            prev = port_id
            for k, pt in enumerate(r["points"][1:], start=1):
                wid = f"{port_id}.runner_{k}"
                node(wid, [float(v) for v in pt], "engine-block-port", port_kind="runner-waypoint")
                edge(f"{port_id}.runner_seg_{k}", wid, prev, "low-pressure-air-line", radius=runner_radius_m,
                     circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
                prev = wid
            re = runner_edges.get(port_id)
            if spec.rail == "none" and len(r["points"]) >= 2:
                # no port injection on this build: every runner still carries a
                # real, default-plugged injector bung just upstream of the port,
                # so the same casting can be converted to port injection later
                # by plumbing it rather than by inventing a new casting
                p0 = np.array(r["points"][0]); p1 = np.array(r["points"][1])
                boss_pos = p1 + (p1 - p0) * 0.5
                boss_dir = _unit(np.cross(_unit(p1 - p0), CRANK_AXIS))
                if np.linalg.norm(boss_dir) < 1e-6:
                    boss_dir = plan["up"]
                _bung(f"{port_id}.injector_bung", port_id, boss_pos + boss_dir * (runner_radius_m * 0.9),
                      boss_dir, 0.005, "injector-bung", "fuel")
            if plan["individual"]:
                itb = next(i for i in plan["itbs"] if i["cylinder"] == r["cylinder"] and i["port"] == r["port"])
                tb_id = f"powertrain.cylinder_{r['cylinder']}.{r['port']}.throttle_body"
                node(tb_id, [float(v) for v in itb["body"]], "engine-block-component", mass_kg=0.6,
                     body_half_extent_m=[runner_radius_m * 1.4] * 3, inlet_kind="individual-throttle-body",
                     flow_axis=[float(v) for v in itb["direction"]], plate_radius_m=runner_radius_m * 1.7)
                node(tb_id + ".stack", [float(v) for v in itb["stack_end"]], "engine-block-port", port_kind="velocity-stack",
                     port_direction=[float(v) for v in itb["direction"]], port_radius_m=runner_radius_m * 1.3)
                edge(tb_id + ".stack_to_body", tb_id + ".stack", tb_id, "low-pressure-air-line", radius=runner_radius_m * 1.2,
                     circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
                src = tb_id
            else:
                src = chamber_ids[r["group"]]
            if re is not None:
                re["a"] = src; re["b"] = prev
            else:
                edge(f"{port_id}.intake_runner", src, prev, "low-pressure-air-line", radius=runner_radius_m,
                     circuit_identity="intake-air", medium_rate_state="intake-air-flow-and-temperature")
        for e in tower_edges:
            edges.remove(e)
        nodes[:] = [n for n in nodes if n.get("port_kind") != "intake-riser-waypoint"]
        if plan["individual"] and tb is not None:
            nodes.remove(tb)
            edges[:] = [e for e in edges if tb["identity"] not in (e["a"], e["b"])]
        # Feed the runner's own real, procedurally-routed path length back
        # into the Helmholtz-resonance model (engines.IntakeSystem.
        # tuned_frequency_hz/resonance_gain) instead of leaving it locked
        # to whatever static catalog default the engine happened to
        # declare: this IS the neck length the graph actually built (bore
        # spacing, chamber routing and bends all folded in), the one real
        # number the resonance solve needs and previously never got.
        real_lengths = [float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a))
                                   for a, b in zip(r["points"], r["points"][1:]))) for r in plan["runners"]]
        if real_lengths and engine is not None and hasattr(engine, "intake_system"):
            engine.intake_system.runner_length_m = sum(real_lengths) / len(real_lengths)
        report["intake"] = f"{len(plan['runners'])}-to-{plan['n_inlets']} ({plan['label']}), filter {spec.air_filter}"

    # ---- fuel rail ----
    rail = by_id.get("powertrain.fuel_rail")
    if spec.rail != "none":
        bosses = [n for n in nodes if n["kind"] == "engine-block-port"
                  and n.get("port_kind") in ("port-injector-boss", "direct-injector-boss", "gas-injector-boss")]
        if bosses:
            xs = [n["reference_position"][0] for n in bosses]
            mean = np.mean([n["reference_position"] for n in bosses], axis=0)
            rail_pos = mean + np.array([0.0, bore * 0.25, 0.0])
            if rail is None:
                node("powertrain.fuel_rail", [float(v) for v in rail_pos], "fuel-rail", mass_kg=1.0)
                rail = nodes[-1]
            rail["reference_position"] = [float(v) for v in rail_pos]
            rail["body_half_extent_m"] = [(max(xs) - min(xs)) / 2.0 + bore * 0.2, bore * 0.06, bore * 0.06]
            rail["rail_kind"] = spec.rail
            for b in bosses:
                edge(f"{b['identity']}.rail_feed", "powertrain.fuel_rail", b["identity"], "fuel-supply-line",
                     radius=0.004 if spec.rail == "port-rail" else 0.003, circuit_identity="fuel",
                     medium_rate_state="fuel-flow-and-pressure", high_pressure=(spec.rail == "common-rail"))
            report["rail"] = f"{spec.rail} feeding {len(bosses)} injectors"
    # fuel filter on the tank line
    if spec.fuel_filter != "none":
        pump = by_id.get("fuel.pump")
        if pump is not None:
            fp = np.array(pump["reference_position"]) + np.array([0.08, 0.02, 0.0])
            node("fuel.filter", [float(v) for v in fp], "fuel-filter", filter=spec.fuel_filter, mass_kg=0.4,
                 body_half_extent_m=[0.03, 0.045 if spec.fuel_filter == "water-separator" else 0.03, 0.03])
            for e in edges:
                if e["identity"] == "fuel.pump_to_rail":
                    e["a"] = "fuel.filter"
            edge("fuel.pump_to_filter", "fuel.pump", "fuel.filter", "fuel-supply-line", radius=0.004,
                 circuit_identity="fuel", medium_rate_state="fuel-flow-and-pressure")
            report["fuel_filter"] = spec.fuel_filter

    # ---- lubrication ----
    pump = by_id.get("powertrain.oil_pump")
    if spec.lube in ("wet-sump", "dry-sump") and pump is not None:
        if spec.oil_filter == "spin-on":
            fpos = np.array(pump["reference_position"]) + np.array([0.0, bore * 0.6, bore * 1.1])
            node("powertrain.oil_filter", [float(v) for v in fpos], "oil-filter", mass_kg=0.8,
                 # an engine circulates a few cubic metres of oil an hour, so a
                 # few hundred cubic metres is one oil-change interval
                 fouling_mode="filter-cake", blocked_frac=0.0, fouling_capacity=450.0,
                 drum_axis=[0.0, 0.0, 1.0], drum_radius_m=bore * 0.38, drum_length_m=bore * 0.75)
            for e in edges:
                if e["identity"] == "powertrain.oil_pump_to_gallery":
                    e["a"] = "powertrain.oil_filter"
            edge("powertrain.oil_pump_to_filter", "powertrain.oil_pump", "powertrain.oil_filter", "oil-line",
                 radius=0.006, circuit_identity="oil")
        if spec.lube == "dry-sump":
            tank_pos = np.array([x0 - bore * 1.8, y0 + bore * 0.6, bore * 1.4])
            node("powertrain.oil_reserve_tank", [float(v) for v in tank_pos], "oil-reserve-tank", mass_kg=6.0,
                 capacity_l=8.0, drum_axis=[0.0, 1.0, 0.0], drum_radius_m=bore * 0.9, drum_length_m=bore * 2.6)
            node("powertrain.scavenge_pump", [float(v) for v in (np.array(pump["reference_position"]) + np.array([bore * 0.6, 0.0, 0.0]))],
                 "rotating-mass", mass_kg=1.5, body_half_extent_m=[bore * 0.2] * 3)
            # the scavenge pulls from the pan when there is one, else from
            # the case's own scavenge drain port (a dry-sump two-stroke)
            scav_src = ("powertrain.oil_pan" if by_id.get("powertrain.oil_pan") is not None
                        else "powertrain.crankcase.scavenge_drain" if by_id.get("powertrain.crankcase.scavenge_drain") is not None
                        else None)
            if scav_src is not None:
                edge("powertrain.pan_to_scavenge", scav_src, "powertrain.scavenge_pump", "oil-line", radius=0.008, circuit_identity="oil")
            edge("powertrain.scavenge_to_tank", "powertrain.scavenge_pump", "powertrain.oil_reserve_tank", "oil-line", radius=0.008, circuit_identity="oil")
            for e in edges:
                if e["identity"] == "powertrain.oil_pan_to_pump":
                    e["a"] = "powertrain.oil_reserve_tank"
            pan = by_id.get("powertrain.oil_pan")
            if pan is not None:
                pan["shallow"] = True
        report["lube"] = spec.lube + (" + spin-on filter" if spec.oil_filter == "spin-on" else "")
    elif spec.lube == "splash-bath":
        # no pump: the rod dips into a trough of oil in the case
        trough = np.array([(x0 + x1) / 2.0, y0 - bore * 0.55, 0.0])
        node("powertrain.oil_bath_trough", [float(v) for v in trough], "oil-bath", mass_kg=2.0,
             body_half_extent_m=[bore * 0.5, bore * 0.12, bore * 0.45], capacity_l=1.5)
        # no pump at all: the node and everything that drove or fed it
        # go; the solver treats the dipper (the splash edge below) as the
        # circuit's supply, in proportion to crank speed
        if pump is not None:
            nodes.remove(pump)
            edges[:] = [e for e in edges if pump["identity"] not in (e["a"], e["b"])]
        edge("powertrain.trough_splash", "powertrain.oil_bath_trough", "powertrain.engine", "oil-line", radius=0.004,
             circuit_identity="oil", splash=True)
        report["lube"] = "splash bath (no pump)"

    # ---- splash paths: the oil source to every cylinder's bore bottom ----
    # every crankcase splash-lubricates its bores; the pan (or the trough
    # on a pump-less engine) is joined to each bore bottom by a real
    # path the crankcase-state model computes participation over (oil
    # film, blow-by, dilution) -- not a pipe the circuit solver flows
    source = "powertrain.oil_bath_trough" if spec.lube == "splash-bath" else "powertrain.oil_pan"
    if spec.lube in ("wet-sump", "dry-sump", "splash-bath") and source in {n["identity"] for n in nodes}:
        n_paths = 0
        for g, _ports in layout:
            if g.kind not in ("spark-piston", "compression-piston"):
                continue
            bid = f"powertrain.cylinder_{g.number}.bore_bottom"
            node(bid, [float(v) for v in g.base], "engine-block-port", port_kind="bore-bottom", cylinder=g.number)
            edge(f"{bid}.splash", source, bid, "oil-splash-path", circuit_identity="oil-splash", cylinder=g.number)
            n_paths += 1
        report["splash"] = f"{source.split('.')[-1]} -> {n_paths} bore bottoms"

    # ---- ignition wiring ----
    plugs = [n for n in nodes if n["kind"] == "engine-block-port" and n.get("port_kind") == "spark-plug-boss"]
    glows = [n for n in nodes if n["kind"] == "engine-block-port" and n.get("port_kind") == "glow-plug-boss"]
    driver = by_id.get("electrical.ignition_driver")
    src = None
    if spec.ignition == "distributor" and plugs:
        # which END the distributor lives at is real hardware, not a
        # constant: a single-bank engine drives it off the FRONT of the
        # cam (behind the timing cover -- the Jeep 258, Ford 300, most
        # inline fours); a V engine's cam-in-block drive is at the REAR
        # of the valley (every American pushrod V8). One fixed rear
        # position had the six's distributor at the flywheel end.
        from head_mesh import banks as _banks
        rear_end = len(_banks(layout)) >= 2
        end_x = (x1 + bore * 1.2) if rear_end else (x0 - bore * 1.2)
        dpos = np.array([end_x, y0 + bore * 1.6, -bore * 0.3])
        node("powertrain.distributor", [float(v) for v in dpos], "distributor", mass_kg=1.2,
             drum_axis=[0.0, 1.0, 0.0], drum_radius_m=bore * 0.32, drum_length_m=bore * 0.6)
        node("powertrain.ignition_coil", [float(v) for v in (dpos + np.array([0.0, 0.0, -bore * 0.8]))], "ignition-coil",
             mass_kg=0.9, drum_axis=[0.0, 1.0, 0.0], drum_radius_m=bore * 0.18, drum_length_m=bore * 0.7)
        edge("powertrain.coil_to_distributor", "powertrain.ignition_coil", "powertrain.distributor", "ignition-lead",
             radius=0.004, circuit_identity="ignition")
        src = "powertrain.distributor"
        for p in plugs:
            edge(f"{p['identity']}.lead", src, p["identity"], "ignition-lead", radius=0.0035, circuit_identity="ignition")
    elif spec.ignition == "coil-on-plug" and plugs:
        for p in plugs:
            cp = np.array(p["reference_position"]) + np.array(p.get("port_direction", [0, 1, 0])) * 0.035
            cid = p["identity"].replace("spark_plug", "coil_pack").replace("leading_plug", "leading_coil").replace("trailing_plug", "trailing_coil")
            node(cid, [float(v) for v in cp], "ignition-coil", mass_kg=0.25, body_half_extent_m=[0.016, 0.03, 0.016])
            edge(f"{p['identity']}.lead", cid, p["identity"], "ignition-lead", radius=0.004, circuit_identity="ignition")
        src = "coil packs"
    elif spec.ignition == "magneto" and plugs:
        dual = getattr(engine, "ignition_profile", "") == "aircraft-dual-magneto"
        # an aircraft engine's magnetos live on the rear accessory case
        # (the anti-prop end, where the whole accessory drive is); a
        # race/stationary magneto rides the crank nose at the front
        mag_x = (x1 + bore * 1.4) if dual else (x0 - bore * 1.4)
        mpos = np.array([mag_x, y0 + bore * 0.5, 0.0 if not dual else -bore * 0.45])
        node("powertrain.magneto", [float(v) for v in mpos], "magneto", mass_kg=2.0,
             rotor_class="magneto",
             drum_axis=[1.0, 0.0, 0.0], drum_radius_m=bore * 0.28, drum_length_m=bore * 0.6)
        edge("powertrain.magneto_drive", "powertrain.engine", "powertrain.magneto", "geared-timing-drive", ratio=0.5)
        srcs = ["powertrain.magneto"]
        if dual:
            # TWO independent magnetos, each firing its own plug in every
            # cylinder -- the real redundancy an aircraft certification
            # demands: lose one system, keep flying on the other
            m2 = mpos + np.array([0.0, 0.0, bore * 0.9])
            node("powertrain.magneto_2", [float(v) for v in m2], "magneto", mass_kg=2.0,
                 rotor_class="magneto",
                 drum_axis=[1.0, 0.0, 0.0], drum_radius_m=bore * 0.28, drum_length_m=bore * 0.6)
            edge("powertrain.magneto_2_drive", "powertrain.engine", "powertrain.magneto_2", "geared-timing-drive", ratio=0.5)
            srcs.append("powertrain.magneto_2")
        for p in plugs:
            # plug 1 (spark_plug) on magneto 1, plug 2 (spark_plug_2, the
            # second boss cylinder_ports adds on dual-magneto engines) on
            # magneto 2; a single-magneto engine wires every plug to it
            second = p["identity"].endswith("spark_plug_2")
            src = srcs[1] if (dual and second) else srcs[0]
            edge(f"{p['identity']}.lead", src, p["identity"], "ignition-lead", radius=0.0035, circuit_identity="ignition")
    elif spec.ignition == "glow" and glows:
        bpos = np.array([(x0 + x1) / 2.0, y0 + bore * 2.6, -bore * 0.9])
        node("powertrain.glow_plug_bus", [float(v) for v in bpos], "electrical-junction", mass_kg=0.3,
             body_half_extent_m=[abs(x1 - x0) / 2.0 + bore * 0.3, 0.008, 0.008])
        src = "powertrain.glow_plug_bus"
        for g_ in glows:
            edge(f"{g_['identity']}.lead", src, g_["identity"], "insulated-copper-wire", radius=0.003, circuit_identity="glow")
    if src and driver is not None and spec.ignition in ("distributor", "magneto", "glow"):
        edge("ignition_driver_to_source", driver["identity"], src, "crank-cam-timed-ignition")
    report["ignition"] = f"{spec.ignition}: {len(plugs) or len(glows)} leads"
    return report
