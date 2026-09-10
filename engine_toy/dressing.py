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
    throttle: str      # "central" | "carburetor" | "individual" | "none"


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
    lube = "splash-bath" if antique_single else ("dry-sump" if race_fuel else "wet-sump")
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
    elif fm == "velocity_stack" or (engine.redline_rpm >= 9000.0 and not carbureted):
        ignition = "coil-on-plug"
    else:
        ignition = "distributor"
    throttle = "none" if ci else ("individual" if fm == "velocity_stack" else ("carburetor" if carbureted else "central"))
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


def intake_inlet_count(engine, spec: DressingSpec, n_ports: int) -> tuple[int, str]:
    """How many inlets (throttle barrels / carburetor barrels / throttle
    bodies) the intake distributes N ports into, and what that is
    called. Declared where the catalogue declares it (a
    ThrottleBodyAssembly's barrels); otherwise the typical build."""
    if n_ports == 0:
        return 0, "none"
    if spec.throttle == "none":
        # a diesel still has a manifold: one open inlet with a filter, no butterfly
        return 1, "unthrottled manifold"
    if spec.throttle == "individual":
        return n_ports, "individual throttle bodies"
    tb = getattr(engine, "throttle_body", None)
    if tb is not None and getattr(tb, "barrels", None):
        m = len(tb.barrels)
        return m, f"{m}-barrel" + (" (double quad)" if m == 8 else "")
    cyl = max(1, engine.architecture.cylinders)
    if spec.throttle == "carburetor":
        if cyl == 1:
            return 1, "single carburetor"
        if cyl <= 6:
            return 2, "two-barrel carburetor"
        if engine.preferred_fuel_profile in ("nitromethane-race", "methanol-race"):
            return 8, "double quad"
        return 4, "single four-barrel"
    return 1, "single throttle body"


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
    """The intake as a DISTRIBUTOR: N intake ports gathered into M
    inlet chambers (the mirror of the exhaust collector). Each chamber
    sits at the mean position of its group along the bank -- in the
    valley of a V/flat engine, inboard and above the head on a
    straight one -- and every runner leaves its port, bends up to the
    rail level, runs along the bank to its chamber and bends in; a
    barrel/throttle body sits on each chamber (or, with M == N, the
    throttle body sits right on the short runner with an open stack).
    So a straight six is 6-to-1 with one carb on a centre chamber, a
    V8 with a four-barrel is 8-to-4, a double quad or ITBs 8-to-8."""
    from head_mesh import banks
    ports = _intake_ports(layout)
    if not ports:
        return None
    bank_list = banks(layout)
    axes = [_unit(gs[0].axis) for gs in bank_list]
    mean_up = _unit(sum(axes))
    bore = max(g.bore_m for g, _ in ports)
    bend_r = 2.0 * runner_radius_m * BEND_RADIUS_FRAC_OF_DIAMETER
    n_inlets, label = intake_inlet_count(engine, spec, len(ports)) if engine is not None else (1, "single")
    n_inlets = max(1, min(n_inlets, len(ports)))
    groups = distribute_ports(layout, ports, n_inlets)
    positions = [np.array(p.position) for _, p in ports]
    centre = sum(positions) / len(positions)
    if len(bank_list) >= 2:
        rail_level = centre + mean_up * (bore * 1.1)
    else:
        inboard = _unit(sum(np.array(p.direction) for _, p in ports))
        inboard = inboard - mean_up * float(np.dot(inboard, mean_up))
        inboard = _unit(inboard) if np.linalg.norm(inboard) > 1e-6 else -_unit(np.cross(CRANK_AXIS, mean_up))
        rail_level = centre + inboard * (bore * 0.75) + mean_up * (bore * 0.8)
    individual = n_inlets == len(ports) and spec.throttle in ("individual", "central") and len(ports) > 1 and label == "individual throttle bodies"
    chambers = []
    for gi in range(n_inlets):
        members = [k for k, gk in enumerate(groups) if gk == gi]
        xs = [positions[k][0] for k in members]
        c = rail_level.copy(); c[0] = float(np.mean(xs))
        half_len = max(bore * 0.45, (max(xs) - min(xs)) / 2.0 * 0.35) if len(members) > 1 else bore * 0.35
        chambers.append({"centre": c, "half_len": half_len, "radius": bore * 0.42, "ports": members})
    plan = {"chambers": chambers, "runners": [], "itbs": [], "up": mean_up, "n_inlets": n_inlets, "label": label,
            "plenum_radius": bore * 0.42, "individual": individual}
    for k, (g, p) in enumerate(ports):
        pos = np.array(p.position); d = _unit(p.direction)
        pts = [pos, pos + d * (p.radius_m * 1.4)]
        if individual:
            body = pts[-1] + d * (bore * 0.35)
            pts.append(body)
            plan["itbs"].append({"cylinder": g.number, "port": p.name, "body": body, "direction": d,
                                 "stack_end": body + d * (bore * 0.45)})
        else:
            ch = chambers[groups[k]]
            target = ch["centre"].copy(); target[0] = pos[0]
            to_t = _unit(target - pts[-1])
            pts.extend(_quarter_bend(pts[-1], d, to_t, bend_r))
            if abs(pos[0] - ch["centre"][0]) > ch["half_len"] * 0.9:
                run_dir = CRANK_AXIS * (1.0 if ch["centre"][0] > pos[0] else -1.0)
                pts.append(target - to_t * (ch["radius"] * 1.2))
                pts.extend(_quarter_bend(pts[-1], to_t, run_dir, bend_r))
                join = pts[-1].copy(); join[0] = ch["centre"][0] - run_dir[0] * (ch["half_len"] * 0.9)
                pts.append(join)
                pts.extend(_quarter_bend(pts[-1], run_dir, to_t, bend_r))
                pts.append(ch["centre"] - run_dir * (ch["half_len"] * 0.6) - to_t * (ch["radius"] * 0.3))
            else:
                pts.append(target - to_t * (ch["radius"] * 0.6))
        plan["runners"].append({"cylinder": g.number, "port": p.name, "points": pts, "group": groups[k]})
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
                chamber_ids.append(cid)
                # the inlet on this chamber: a throttle barrel / carburetor barrel
                inlet_pos = ch["centre"] + plan["up"] * (ch["radius"] * 1.3)
                tid = "powertrain.throttle_body" if gi == 0 else f"powertrain.throttle_body_{gi + 1}"
                t_ = tb if (gi == 0 and tb is not None) else None
                if t_ is None:
                    node(tid, [float(v) for v in inlet_pos], "engine-block-component", mass_kg=1.0)
                    t_ = nodes[-1]
                t_["reference_position"] = [float(v) for v in inlet_pos]
                t_["body_half_extent_m"] = [ch["radius"] * 0.6, ch["radius"] * 0.55, ch["radius"] * 0.6]
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
            xs = [ch["centre"][0] for ch in plan["chambers"]]
            fc = sum(ch["centre"] for ch in plan["chambers"]) / len(plan["chambers"]) + plan["up"] * (bore * 1.3)
            span = (max(xs) - min(xs)) / 2.0 + bore * 0.9
            node("powertrain.air_filter", [float(v) for v in fc], "air-filter", filter=spec.air_filter, mass_kg=1.5,
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
            if plan["individual"]:
                itb = next(i for i in plan["itbs"] if i["cylinder"] == r["cylinder"] and i["port"] == r["port"])
                tb_id = f"powertrain.cylinder_{r['cylinder']}.{r['port']}.throttle_body"
                node(tb_id, [float(v) for v in itb["body"]], "engine-block-component", mass_kg=0.6,
                     body_half_extent_m=[runner_radius_m * 1.4] * 3, inlet_kind="individual-throttle-body")
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
            edge("powertrain.pan_to_scavenge", "powertrain.oil_pan", "powertrain.scavenge_pump", "oil-line", radius=0.008, circuit_identity="oil")
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
        dpos = np.array([x1 + bore * 1.2, y0 + bore * 1.6, -bore * 0.3])
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
        mpos = np.array([x0 - bore * 1.4, y0 + bore * 0.5, 0.0])
        node("powertrain.magneto", [float(v) for v in mpos], "magneto", mass_kg=2.0,
             drum_axis=[1.0, 0.0, 0.0], drum_radius_m=bore * 0.28, drum_length_m=bore * 0.6)
        edge("powertrain.magneto_drive", "powertrain.engine", "powertrain.magneto", "geared-timing-drive", ratio=0.5)
        src = "powertrain.magneto"
        for p in plugs:
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
