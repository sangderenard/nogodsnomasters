"""Part PORTS and the port-to-port mating solver -- the assembly rule
that lets any head go on any block, any pan under any case, and tells
you exactly what connected and what was left open.

Every major casting declares its ports from the cylinder layout:

  head (per bank)   oil_fill (a line port on top: the filler, open by
                    default -- nothing plugs into it until a cap or a
                    catch-can line does); oil_feed_face and two
                    oil_return_face ports on the DECK (mating ports:
                    holes in the gasket face that line up with the
                    block's when the head is bolted on); coolant_face
                    ports on the deck for a jacketed engine
  crankcase/block   the matching deck-face ports under each head;
                    main_gallery (line: to a filter/cooler), breather
                    (line), dipstick (line), pump_pickup_face (mating,
                    down into the pan); pan_rim_face (mating)
  oil pan           pan_rim_face (mating, up to the case), pickup_face
                    (mating: the pump's pickup sits in the pan's oil),
                    drain_plug (line, closed by a plug part)

`mate_ports(ports)` pairs mating ports across DIFFERENT parts that are
compatible in kind, face each other (opposite directions), and sit
within a tolerance of each other -- a real gasket-face match. It is
O(n log n): ports are bucketed by kind, sorted along the crank axis,
and each port only looks at the few neighbours within tolerance. The
result is a list of seals (graph edges of kind "port-face-seal") plus
the OPEN ports: everything that found no partner. An open mating port
on a head means the head's gallery is fed nothing (starved
valvetrain) or drains onto the floor (a leak) -- the consequences a
"catastrophic parts mixing" needs to be able to reach, and the same
report a creative interchange uses to see what still needs a hose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

CRANK_AXIS = np.array([1.0, 0.0, 0.0])
MATE_TOLERANCE_M = 0.012        # gasket-face registration: holes within this line up
MATE_ANGLE_COS = -0.5           # facing each other: directions at least ~120 deg apart


def _unit(v):
    a = np.array(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


@dataclass
class PartPort:
    identity: str
    part: str                 # the casting this hole is in
    kind: str                 # "oil-feed" | "oil-return" | "oil-fill" | "coolant" | "pump-pickup" | "pan-rim" | "breather" | "dipstick" | "drain-plug" | "main-gallery"
    position: np.ndarray
    direction: np.ndarray     # outward
    radius_m: float
    mating: bool              # True: a gasket-face hole that mates when parts bolt up; False: a line port (needs a hose/pipe/plug)
    connected_to: str | None = None
    fluid: str = "oil"


# kinds that mate with each other across a joint face
MATE_PAIRS = {("oil-feed", "oil-feed"), ("oil-return", "oil-return"), ("coolant", "coolant"),
              ("pump-pickup", "pump-pickup"), ("pan-rim", "pan-rim")}


def _perp(axis):
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = _unit(np.cross(axis, helper)); v = _unit(np.cross(axis, u))
    return u, v


def part_ports(layout, wet_sump: bool = True) -> list[PartPort]:
    """Every casting's ports for this layout: heads per bank, the
    crankcase, the pan. Expander and atmospheric layouts have no oil
    galleries in this sense (open cranks, drip lubricators) and get
    only a lubricator line port each, already in their cylinder ports.

    wet_sump=False (a total-loss two-stroke: oil is mixed into the fuel)
    keeps only the crankcase breather -- there is no pan, no drain plug,
    no dipstick, no main gallery, no head oil fill to declare. These
    used to be emitted unconditionally, putting a dipstick on a 25 cc
    string trimmer."""
    from head_mesh import banks
    from crank_mesh import crank_stations, _pitch
    ports: list[PartPort] = []
    bank_list = banks(layout)
    stations = crank_stations(layout)
    if not bank_list or not stations:
        return ports
    gs_all = [g for gs in bank_list for g in gs]
    bore = max(g.bore_m for g in gs_all)
    y0 = float(stations[0][1][0].crank_centre[1]); z0 = float(stations[0][1][0].crank_centre[2])
    jacketed = any(g.cooling in ("jacket", "hopper") for g in gs_all)
    for b_index, gs in enumerate(bank_list):
        gs = sorted(gs, key=lambda g: g.base[0])
        axis = _unit(gs[0].axis)
        side = _unit(np.cross(CRANK_AXIS, axis))
        wall = bore * 0.10
        xs = [float(g.base[0]) for g in gs]
        pitch = (xs[1] - xs[0]) if len(xs) > 1 else bore * 1.3
        deck_pts = [np.array(g.base) + axis * (g.length_m + wall * 1.5) for g in gs]
        deck_c = sum(deck_pts) / len(deck_pts)
        head = f"head_bank{b_index + 1}"
        block = "crankcase"
        deck = f"crankcase.deck{b_index + 1}"   # the block's deck face under THIS bank
        # deck-face holes: feed gallery at the rear, returns at both ends, coolant passages between cylinders
        feed = deck_c.copy(); feed[0] = max(xs) + pitch * 0.30; feed = feed + side * (bore * 0.45)
        if wet_sump:
            # pressure-fed head oiling: a feed gallery up through the deck
            # and returns at both ends -- a total-loss two-stroke has no
            # oil galleries between case and head at all
            for part, dirn in ((head, -axis), (deck, axis)):
                ports.append(PartPort(f"{part}.oil_feed_face", "crankcase" if part == deck else part, "oil-feed", feed.copy(), dirn, 0.004, True))
            for k, xr in enumerate((min(xs) - pitch * 0.30, max(xs) + pitch * 0.30)):
                ret = deck_c.copy(); ret[0] = xr; ret = ret - side * (bore * 0.45)
                for part, dirn in ((head, -axis), (deck, axis)):
                    ports.append(PartPort(f"{part}.oil_return_face_{k + 1}", "crankcase" if part == deck else part, "oil-return", ret.copy(), dirn, 0.007, True))
        if jacketed:
            for k in range(len(xs) + 1):
                xc = (xs[0] - pitch / 2.0) + k * pitch
                for sgn in (-1.0, 1.0):
                    cp = deck_c.copy(); cp[0] = xc; cp = cp + side * (sgn * bore * 0.58)
                    for part, dirn in ((head, -axis), (deck, axis)):
                        ports.append(PartPort(f"{part}.coolant_face_{k + 1}{'a' if sgn < 0 else 'b'}", "crankcase" if part == deck else part, "coolant",
                                              cp.copy(), dirn, 0.006, True, fluid="coolant"))
        if wet_sump:
            # the FILL: a line port on top of the head/cover, open until something is put on it
            top = deck_c + axis * (bore * 0.9); top[0] = min(xs) + pitch * 0.2
            ports.append(PartPort(f"{head}.oil_fill", head, "oil-fill", top, axis, 0.016, False))
    # crankcase line ports and the pan joint
    x_a = stations[0][0] - _pitch(stations, bore) / 2.0
    x_b = stations[-1][0] + _pitch(stations, bore) / 2.0
    r_throw = max(g.crank_radius_m for _, gs in stations for g in gs)
    tunnel_r = r_throw + bore * 0.32
    rim_y = y0 - tunnel_r * 0.6
    ports.append(PartPort("crankcase.breather", "crankcase", "breather", np.array([x_a + bore * 0.4, y0 + tunnel_r, z0 - bore * 0.3]),
                          np.array([0.0, 1.0, 0.0]), 0.009, False, fluid="crankcase-gas"))
    if not wet_sump:
        return ports
    ports.append(PartPort("crankcase.main_gallery", "crankcase", "main-gallery", np.array([x_b + bore * 0.1, y0 + bore * 0.2, z0 + tunnel_r]),
                          np.array([0.0, 0.0, 1.0]), 0.008, False))
    ports.append(PartPort("crankcase.dipstick", "crankcase", "dipstick", np.array([(x_a + x_b) / 2.0, y0 + tunnel_r * 0.8, z0 + tunnel_r * 0.9]),
                          np.array([0.0, 0.7, 0.7]), 0.005, False))
    for k, xr in enumerate((x_a + bore * 0.2, (x_a + x_b) / 2.0, x_b - bore * 0.2)):
        for part, dirn in (("crankcase", np.array([0.0, -1.0, 0.0])), ("oil_pan", np.array([0.0, 1.0, 0.0]))):
            ports.append(PartPort(f"{part}.pan_rim_face_{k + 1}", part, "pan-rim", np.array([xr, rim_y, z0 + tunnel_r * 0.8]),
                                  dirn, 0.005, True))
    pick = np.array([(x_a + x_b) / 2.0, rim_y, z0])
    ports.append(PartPort("crankcase.pump_pickup_face", "crankcase", "pump-pickup", pick.copy(), np.array([0.0, -1.0, 0.0]), 0.012, True))
    ports.append(PartPort("oil_pan.pickup_face", "oil_pan", "pump-pickup", pick.copy(), np.array([0.0, 1.0, 0.0]), 0.012, True))
    ports.append(PartPort("oil_pan.drain_plug", "oil_pan", "drain-plug", np.array([x_b - bore * 0.3, rim_y - bore * 0.65, z0]),
                          np.array([0.0, -1.0, 0.0]), 0.008, False))
    return ports


@dataclass
class MateResult:
    seals: list = field(default_factory=list)      # (port_a, port_b)
    open_ports: list = field(default_factory=list)  # mating ports with no partner
    line_ports: list = field(default_factory=list)  # ports that need a hose/pipe/plug regardless

    def summary(self) -> list[str]:
        out = [f"{len(self.seals)} seals, {len(self.open_ports)} open mating ports, {len(self.line_ports)} line ports"]
        for a, b in self.seals[:6]:
            out.append(f"  seal  {a.identity:34s} <-> {b.identity}")
        if len(self.seals) > 6:
            out.append(f"  ... {len(self.seals) - 6} more seals")
        for p in self.open_ports:
            out.append(f"  OPEN  {p.identity:34s} ({p.kind}, {p.fluid})")
        for p in self.line_ports:
            out.append(f"  line  {p.identity:34s} ({p.kind}, {p.fluid}) -> {p.connected_to or 'nothing'}")
        return out


def mate_ports(ports: list[PartPort], tolerance_m: float = MATE_TOLERANCE_M) -> MateResult:
    """Pair every mating port with the nearest compatible, facing port
    on another part within tolerance. Buckets by kind, sorts along
    the crank axis, and scans only the neighbours whose x is within
    tolerance -- n log n, and each port mates at most once."""
    result = MateResult()
    mating = [p for p in ports if p.mating]
    result.line_ports = [p for p in ports if not p.mating]
    by_kind: dict[str, list] = {}
    for p in mating:
        by_kind.setdefault(p.kind, []).append(p)
    taken: set = set()
    for kind, group in by_kind.items():
        if (kind, kind) not in MATE_PAIRS:
            continue
        group.sort(key=lambda p: float(p.position[0]))
        xs = np.array([float(p.position[0]) for p in group])
        for i, a in enumerate(group):
            if a.identity in taken:
                continue
            lo = int(np.searchsorted(xs, xs[i] - tolerance_m)); hi = int(np.searchsorted(xs, xs[i] + tolerance_m, side="right"))
            best = None; best_d = tolerance_m
            for j in range(lo, hi):
                b = group[j]
                if b is a or b.identity in taken or b.part == a.part:
                    continue
                if float(np.dot(_unit(a.direction), _unit(b.direction))) > MATE_ANGLE_COS:
                    continue
                d = float(np.linalg.norm(a.position - b.position))
                if d < best_d:
                    best, best_d = b, d
            if best is not None:
                taken.add(a.identity); taken.add(best.identity)
                a.connected_to = best.identity; best.connected_to = a.identity
                result.seals.append((a, best))
    result.open_ports = [p for p in mating if p.identity not in taken]
    return result


def emit_ports_graph(ports: list[PartPort], result: MateResult, node, edge) -> None:
    """Ports as engine-block-port nodes (connected flag = mated or
    plumbed), seals as zero-length 'port-face-seal' edges in the
    fluid's own circuit."""
    for p in ports:
        node(f"powertrain.{p.identity}", [float(v) for v in p.position], "engine-block-port", port_kind=p.kind,
             port_direction=[float(v) for v in _unit(p.direction)], port_radius_m=p.radius_m, fluid_role=p.fluid,
             mating=p.mating, connected=p.connected_to is not None, part=p.part)
    for a, b in result.seals:
        circuit = {"oil": "oil", "coolant": "coolant", "crankcase-gas": "crankcase"}.get(a.fluid, a.fluid)
        edge(f"powertrain.seal.{a.identity}__{b.identity}", f"powertrain.{a.identity}", f"powertrain.{b.identity}",
             "port-face-seal", radius=a.radius_m, circuit_identity=circuit, gasket=True)


def transplant(ports_on_block: list[PartPort], donor_head_ports: list[PartPort], x_offset_m: float = 0.0,
               tolerance_m: float = MATE_TOLERANCE_M, deck_offset_m: float = 0.0, bank_axis=(0.0, 1.0, 0.0)) -> MateResult:
    """A hot-rod: try a donor head's ports against this block's, with
    the donor shifted along the crank by x_offset_m (how it was set
    on the deck). What mates, mates; what doesn't is an open hole."""
    import copy
    block = [copy.copy(p) for p in ports_on_block if not p.part.startswith("head_")]
    donor = []
    for p in donor_head_ports:
        if p.part.startswith("head_"):
            q = copy.copy(p)
            q.position = p.position + np.array([x_offset_m, 0.0, 0.0]) + _unit(bank_axis) * deck_offset_m
            q.connected_to = None
            donor.append(q)
    for p in block:
        p.connected_to = None
    return mate_ports(block + donor, tolerance_m)
