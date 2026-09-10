"""Cylinder heads as castings, the cam case, the valve covers, and the
crankcase top (valley) cover -- the parts that close the block ABOVE
the bores, built per BANK from the cylinder layout the way crank_mesh
builds the crank train per station below them.

  head casting   one casting per bank spanning that bank's cylinders,
                 sitting on the bore heads, oriented along the bank's
                 own bore axis (so a V or flat bank's head is tilted
                 with its bank)
  cam case       where the camshaft actually lives, by valvetrain:
                   pushrod -- the cam runs in the BLOCK: a cam tunnel
                              beside the crank, pushrod tubes up each
                              cylinder to a rocker cover
                   sohc     -- one camshaft in a cam box on the head
                   dohc     -- two camshafts in a wider cam box
                 (a two-stroke with no poppet valves has none of this)
  valve cover    the pressed/cast cover over the cam box or rockers
  valley cover   multi-bank engines: the plate that closes the crank
                 tunnel between the bank skirts (a V/W valley cover;
                 the top of a flat engine's case), without which the
                 crankcase is open between the banks

The valvetrain layout is not yet a catalogue field, so it is derived
here from what IS declared and disclosed as such: 3+ valves per
cylinder means twin cams (dohc); a two-valve head with a redline over
7000 rpm means an overhead cam (sohc); any other two-valve head is a
pushrod engine, which is what every catalogue engine with a cam in
its production graph (`powertrain.camshaft` near the crank) actually
is. Declaring `EngineArchitecture.valvetrain` later replaces the
guess, not this module.
"""
from __future__ import annotations

import math

import numpy as np

from mesh_primitives import tube_mesh, cuboid_mesh, capped_tube_mesh

CRANK_AXIS = np.array([1.0, 0.0, 0.0])


def _unit(v):
    a = np.array(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


def _tube(a, b, radius, sides=14):
    return capped_tube_mesh(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), radius, sides=sides)


def oriented_box(center, half_extent, up):
    """A cuboid whose local +y is `up` (its local x stays the crank
    axis): an axis-aligned cuboid rotated about x, so a tilted bank's
    head/cam box/cover tilts with the bank instead of staying square
    to the world."""
    up = _unit(up)
    x = CRANK_AXIS
    z = _unit(np.cross(x, up))
    y = _unit(np.cross(z, x))
    rot = np.stack([x, y, z], axis=1)          # local -> world
    vtx, nrm = cuboid_mesh(np.zeros(3), np.array(half_extent, dtype=np.float64))
    vtx = vtx @ rot.T + np.array(center, dtype=np.float64)
    nrm = nrm @ rot.T
    return vtx, nrm


def valvetrain_layout(geometry, redline_rpm: float = 0.0) -> str:
    """The geometry carries the (derived, see cylinder_ports) layout."""
    if geometry.wall_ports == "loop" or geometry.kind not in ("spark-piston", "compression-piston"):
        return "none"
    return getattr(geometry, "valvetrain", "pushrod")


def banks(layout) -> list[list]:
    """Cylinders grouped by bore axis direction -- one group per bank."""
    groups: dict[tuple, list] = {}
    for g, _ports in layout:
        if g.kind not in ("spark-piston", "compression-piston"):
            continue
        key = tuple(np.round(_unit(g.axis), 3))
        groups.setdefault(key, []).append(g)
    return list(groups.values())


def head_oil_ports(layout, wet_sump: bool = True) -> list[dict]:
    """Kept for the mesh: the ports themselves live in assembly_ports
    (heads: a FILL on top plus deck-face feed/return/coolant holes;
    crankcase and pan: their own). Drawn as stubs by build_head_parts.
    wet_sump must match what drivetrain_graph declared, or the mesh
    draws stubs for ports the graph no longer has."""
    from assembly_ports import part_ports
    out = []
    for p in part_ports(layout, wet_sump=wet_sump):
        out.append({"identity": f"powertrain.{p.identity}", "position": [float(v) for v in p.position],
                    "direction": [float(v) for v in p.direction], "radius_m": p.radius_m, "port_kind": p.kind,
                    "fluid_role": p.fluid, "mating": p.mating, "part": p.part})
    return out


def build_head_parts(layout, covers_off: bool = False, engine=None, crank_angle_deg: float = 0.0,
                     spring_style: str = "helix") -> list:
    from vehicle_mesh import SolidPart
    parts: list = []
    bank_list = banks(layout)
    wet_sump = engine is None or not engine.architecture.two_stroke
    for port in head_oil_ports(layout, wet_sump=wet_sump):
        d = np.array(port["direction"]); pos = np.array(port["position"])
        depth = 0.006 if port["mating"] else 0.018
        vtx, nrm = _tube(pos - d * 0.004, pos + d * depth, port["radius_m"], sides=10)
        grp = "coolant" if port["fluid_role"] == "coolant" else "oil"
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"port_{port['identity'].replace('powertrain.', '').replace('.', '_')}"))
    if covers_off:
        from valvetrain_parts import build_valvetrain_parts
        parts.extend(build_valvetrain_parts(layout, engine, crank_angle_deg, spring_style=spring_style))
    for b_index, gs in enumerate(bank_list):
        gs = sorted(gs, key=lambda g: g.base[0])
        axis = _unit(gs[0].axis)
        bore = max(g.bore_m for g in gs)
        wall = bore * 0.10
        xs = [float(g.base[0]) for g in gs]
        pitch = (xs[1] - xs[0]) if len(xs) > 1 else bore * 1.3
        x_a, x_b = min(xs) - pitch / 2.0, max(xs) + pitch / 2.0
        # the head deck: on top of the bore heads of this bank
        head_pts = [np.array(g.base) + axis * (g.length_m + wall * 1.5) for g in gs]
        deck = sum(head_pts) / len(head_pts)
        deck[0] = (x_a + x_b) / 2.0
        head_t = bore * 0.45
        grp = f"block_cyl_{gs[len(gs) // 2].number}"
        vtx, nrm = oriented_box(deck + axis * (head_t / 2.0), [(x_b - x_a) / 2.0, head_t / 2.0, bore * 0.72], axis)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"head_casting_bank{b_index + 1}"))
        vt = valvetrain_layout(gs[0])
        top = deck + axis * head_t
        if vt in ("sohc", "dohc"):
            n_cams = 2 if vt == "dohc" else 1
            box_h = bore * 0.35
            box_w = bore * (0.72 if n_cams == 2 else 0.5)
            vtx, nrm = oriented_box(top + axis * (box_h / 2.0), [(x_b - x_a) / 2.0 - bore * 0.05, box_h / 2.0, box_w], axis)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"cam_box_bank{b_index + 1}"))
            side = _unit(np.cross(CRANK_AXIS, axis))
            for c in range(n_cams):
                off = side * ((c - (n_cams - 1) / 2.0) * bore * 0.45)
                a = top + axis * (box_h * 0.55) + off
                vtx, nrm = _tube(a + CRANK_AXIS * (x_a - deck[0]), a + CRANK_AXIS * (x_b - deck[0]), bore * 0.07, sides=10)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"camshaft_bank{b_index + 1}_{c + 1}"))
                # a lobe per cylinder
                for g in gs:
                    lc = a + CRANK_AXIS * (float(g.base[0]) - deck[0])
                    vtx, nrm = _tube(lc - CRANK_AXIS * (bore * 0.06), lc + CRANK_AXIS * (bore * 0.06), bore * 0.11, sides=10)
                    parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"cam_lobe_bank{b_index + 1}_{c + 1}_cyl{g.number}"))
            cover_base = top + axis * box_h
            cover_h = bore * 0.12
        elif vt == "pushrod":
            # the cam tunnel sits in the block beside the crank; pushrods run up the side of each bore
            side = _unit(np.cross(CRANK_AXIS, axis))
            crank_c = np.array(gs[0].crank_centre); crank_c[0] = deck[0]
            cam_c = crank_c + axis * (gs[0].crank_radius_m * 2.2) - side * (bore * 0.75)
            vtx, nrm = _tube(cam_c + CRANK_AXIS * (x_a - deck[0]), cam_c + CRANK_AXIS * (x_b - deck[0]), bore * 0.09, sides=10)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"cam_in_block_bank{b_index + 1}"))
            rocker_h = bore * 0.28
            for g in gs:
                lc = cam_c + CRANK_AXIS * (float(g.base[0]) - deck[0])
                vtx, nrm = _tube(lc, lc + axis * (bore * 0.16), bore * 0.11, sides=10)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"cam_lobe_bank{b_index + 1}_cyl{g.number}"))
                rocker = top + CRANK_AXIS * (float(g.base[0]) - deck[0]) + axis * (rocker_h * 0.5) - side * (bore * 0.15)
                vtx, nrm = _tube(lc + axis * (bore * 0.16), rocker, bore * 0.03, sides=8)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"pushrod_cyl{g.number}"))
                vtx, nrm = _tube(rocker - side * (bore * 0.2), rocker + side * (bore * 0.25), bore * 0.035, sides=8)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"rocker_cyl{g.number}"))
            cover_base = top
            cover_h = rocker_h + bore * 0.12
        else:
            continue
        if not covers_off:
            vtx, nrm = oriented_box(cover_base + axis * (cover_h / 2.0), [(x_b - x_a) / 2.0 - bore * 0.08, cover_h / 2.0, bore * 0.55], axis)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"valve_cover_bank{b_index + 1}"))

    # valley / top cover between banks
    if len(bank_list) >= 2:
        axes = [_unit(gs[0].axis) for gs in bank_list]
        mean_up = _unit(sum(axes))
        gs_all = [g for gs in bank_list for g in gs]
        bore = max(g.bore_m for g in gs_all)
        xs = [float(g.base[0]) for g in gs_all]
        pitch = bore * 1.3
        x_a, x_b = min(xs) - pitch / 2.0, max(xs) + pitch / 2.0
        crank_c = np.array(gs_all[0].crank_centre); crank_c[0] = (x_a + x_b) / 2.0
        r_throw = max(g.crank_radius_m for g in gs_all)
        # the cover sits at the height where the bank skirts meet the tunnel top
        spread = max(abs(float(np.dot(a, mean_up))) for a in axes)
        opposed = spread < 0.2   # a flat engine: banks are opposite, the "valley" is the case top
        if opposed:
            up = _unit(np.cross(CRANK_AXIS, axes[0]))
            if up[1] < 0: up = -up
            h = r_throw + bore * 0.34
            vtx, nrm = oriented_box(crank_c + up * (h + bore * 0.05), [(x_b - x_a) / 2.0, bore * 0.05, bore * 0.6], up)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group="oil", name="case_top_cover"))
        else:
            h = r_throw + bore * 0.34
            half_angle = math.acos(max(-1.0, min(1.0, float(np.dot(axes[0], mean_up)))))
            width = (h + bore * 0.6) * math.tan(half_angle) if half_angle > 1e-3 else bore * 0.6
            vtx, nrm = oriented_box(crank_c + mean_up * (h + bore * 0.55), [(x_b - x_a) / 2.0, bore * 0.05, max(width, bore * 0.3)], mean_up)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group="oil", name="valley_cover"))
    return parts
