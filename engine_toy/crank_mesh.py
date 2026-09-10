"""The crank train and the crankcase, built from the cylinder layout
(cylinder_ports.CylinderGeometry) rather than per cylinder: a real
crankshaft is ONE part whose main journals sit BETWEEN the throws at
the cylinder pitch, and the crankcase is one casting whose bulkheads
carry those mains -- neither is a thing any single cylinder owns.

Crankshaft (any crank kind: piston engines, expanders):
  - throw stations: cylinders grouped by crank-axis position; a V pair
    (same station, two bank axes) shares one crankpin, a radial row
    shares a single throw among all its rods
  - main journals between stations and at both ends, at the station
    pitch; a nose ahead of the front main (pulley/accessory drive) and
    a flywheel disc behind the rear main
  - each throw: two webs from the main axis to the crankpin at that
    station's phase (crank angle + throw phase), the pin, and a
    counterweight opposite it
Crankcase (piston kinds):
  - a crank tunnel along the crank axis sized to clear the throw
    swing, split into one segment per station so each carries its
    cylinder's own block temperature (block_cyl_N)
  - a bulkhead disc at every main journal
  - a skirt from the tunnel up to each bore's foot along that bore's
    own axis (this is what makes a V or flat case a V or flat case)
  - front cover and rear main-seal bosses on the end walls
  - a wet sump below (thermal group "oil") -- unless the engine is a
    loop-scavenged two-stroke, whose crankcase is a SEALED pump
    chamber per cylinder with its own intake (reed/piston-port) boss
    and no sump at all
Expanders get a bedplate with a main-bearing pedestal at each station
and an open crank (no case, no sump); the Otto-Langen has no crank
(its pinion is drawn with the cylinder).
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


def crank_stations(layout) -> list[tuple[float, list]]:
    """[(x, [CylinderGeometry, ...]), ...] sorted along the crank axis --
    cylinders that share a crankpin share a station."""
    groups: dict[float, list] = {}
    for g, _ports in layout:
        if g.crank_radius_m <= 0.0 or g.rod_length_m <= 0.0 or g.kind == "atmospheric":
            continue
        key = round(float(g.crank_centre[0]), 4)
        groups.setdefault(key, []).append(g)
    return sorted(groups.items(), key=lambda kv: kv[0])


def _pitch(stations, bore_m: float) -> float:
    xs = [x for x, _ in stations]
    if len(xs) >= 2:
        diffs = sorted(b - a for a, b in zip(xs, xs[1:]))
        return max(diffs[len(diffs) // 2], bore_m * 1.05)
    return bore_m * 1.3


def _pin_position(g, crank_angle_deg: float) -> np.ndarray:
    axis = _unit(g.axis)
    side = _unit(np.cross(CRANK_AXIS, axis))
    th = math.radians(crank_angle_deg + g.throw_angle_deg)
    c = np.array(g.crank_centre)
    return c + axis * (g.crank_radius_m * math.cos(th)) + side * (g.crank_radius_m * math.sin(th))


def crank_end_fittings(layout) -> dict | None:
    """The real positions/radii of the crank-end hardware this module
    already DRAWS (the pulley/damper on the nose, the flange and
    flywheel on the tail, a second flywheel on a twin-flywheel single)
    -- the SAME math build_crankshaft_parts uses, exposed so the graph
    can declare a real node for each at the exact place the mesh
    already puts it, instead of a second position that could drift."""
    stations = crank_stations(layout)
    if not stations:
        return None
    bore = max(g.bore_m for _, gs in stations for g in gs)
    pitch = _pitch(stations, bore)
    r_main = bore * 0.11
    web_t = bore * 0.075
    radial = len(stations) == 1 and len(stations[0][1]) > 2
    pin_len = (bore * 0.5 if radial else (bore * 0.42 if len(stations[0][1]) > 1 else bore * 0.26))
    y0 = float(stations[0][1][0].crank_centre[1]); z0 = float(stations[0][1][0].crank_centre[2])
    xs = [x for x, _ in stations]
    main_xs = [xs[0] - pitch / 2.0] + [(a + b) / 2.0 for a, b in zip(xs, xs[1:])] + [xs[-1] + pitch / 2.0]
    main_len = max(pitch - pin_len - 2.0 * web_t, bore * 0.12)
    nose_x = main_xs[0] - main_len / 2.0
    tail_x = main_xs[-1] + main_len / 2.0
    twin = any(getattr(g, "twin_flywheels", False) for _, gs in stations for g in gs)
    out = {
        "bore_m": bore, "r_main_m": r_main, "y0": y0, "z0": z0, "twin_flywheels": twin,
        "nose_end": [nose_x - bore * 0.55, y0, z0],
        "flywheel_centre": [tail_x + bore * 0.30, y0, z0], "flywheel_radius_m": bore * 1.15,
        "flywheel_half_len_m": bore * 0.12,
        "flange_centre": [tail_x + bore * 0.09, y0, z0], "flange_radius_m": r_main * 1.3,
    }
    if twin:
        out["front_flywheel_centre"] = [nose_x - bore * 0.50, y0, z0]
        out["front_flywheel_radius_m"] = bore * 1.15
    else:
        out["pulley_centre"] = [nose_x - bore * 0.485, y0, z0]
        out["pulley_radius_m"] = bore * 0.45
        out["pulley_half_len_m"] = bore * 0.065
    return out


def build_crankshaft_parts(layout, crank_angle_deg: float = 0.0) -> list:
    from vehicle_mesh import SolidPart
    parts: list = []
    stations = crank_stations(layout)
    if not stations:
        return parts
    bore = max(g.bore_m for _, gs in stations for g in gs)
    r_throw = max(g.crank_radius_m for _, gs in stations for g in gs)
    pitch = _pitch(stations, bore)
    r_main = bore * 0.11
    r_pin = bore * 0.095
    web_t = bore * 0.075
    radial = len(stations) == 1 and len(stations[0][1]) > 2
    pin_len = (bore * 0.5 if radial else (bore * 0.42 if len(stations[0][1]) > 1 else bore * 0.26))
    y0 = float(stations[0][1][0].crank_centre[1]); z0 = float(stations[0][1][0].crank_centre[2])
    centre_of = lambda x: np.array([x, y0, z0])

    # main journals: between stations and at both ends
    xs = [x for x, _ in stations]
    main_xs = [xs[0] - pitch / 2.0] + [(a + b) / 2.0 for a, b in zip(xs, xs[1:])] + [xs[-1] + pitch / 2.0]
    main_len = max(pitch - pin_len - 2.0 * web_t, bore * 0.12)
    for i, mx in enumerate(main_xs):
        a = centre_of(mx - main_len / 2.0); b = centre_of(mx + main_len / 2.0)
        vtx, nrm = _tube(a, b, r_main, sides=14)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"crank_main_{i + 1}"))
    # nose and flywheel
    nose_a = centre_of(main_xs[0] - main_len / 2.0)
    vtx, nrm = _tube(nose_a - CRANK_AXIS * (bore * 0.55), nose_a, r_main * 0.8, sides=12)
    parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="crank_nose"))
    twin = any(getattr(g, "twin_flywheels", False) for _, gs in stations for g in gs)
    if twin:
        # a stationary/hit-and-miss single: a second flywheel on the
        # nose instead of a pulley (the belt runs off one of the rims)
        vtx, nrm = _tube(nose_a - CRANK_AXIS * (bore * 0.62), nose_a - CRANK_AXIS * (bore * 0.38), bore * 1.15, sides=28)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="flywheel_front"))
    else:
        vtx, nrm = _tube(nose_a - CRANK_AXIS * (bore * 0.55), nose_a - CRANK_AXIS * (bore * 0.42), bore * 0.45, sides=20)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="crank_pulley"))
    tail = centre_of(main_xs[-1] + main_len / 2.0)
    vtx, nrm = _tube(tail, tail + CRANK_AXIS * (bore * 0.18), r_main * 1.3, sides=12)
    parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="crank_flange"))
    vtx, nrm = _tube(tail + CRANK_AXIS * (bore * 0.18), tail + CRANK_AXIS * (bore * 0.42), bore * 1.15, sides=28)
    parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="flywheel"))

    # throws
    for k, (x, gs) in enumerate(stations):
        g = gs[0]
        pin = _pin_position(g, crank_angle_deg)
        c = np.array(g.crank_centre)
        half = pin_len / 2.0
        for j, dx in enumerate((-half - web_t / 2.0, half + web_t / 2.0)):
            off = CRANK_AXIS * dx
            vtx, nrm = _tube(c + off, pin + off, bore * 0.16, sides=10)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"crank_web_{k + 1}_{j + 1}"))
            # counterweight opposite the pin
            away = c + (c - pin) * 0.75
            vtx, nrm = _tube(c + off, away + off, bore * 0.19, sides=10)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"crank_counterweight_{k + 1}_{j + 1}"))
        vtx, nrm = _tube(pin - CRANK_AXIS * half, pin + CRANK_AXIS * half, r_pin, sides=12)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"crankpin_{k + 1}"))
    return parts


def build_crankcase_parts(layout) -> list:
    from vehicle_mesh import SolidPart
    parts: list = []
    stations = crank_stations(layout)
    if not stations:
        return parts
    kinds = {g.kind for _, gs in stations for g in gs}
    bore = max(g.bore_m for _, gs in stations for g in gs)
    r_throw = max(g.crank_radius_m for _, gs in stations for g in gs)
    pitch = _pitch(stations, bore)
    xs = [x for x, _ in stations]
    main_xs = [xs[0] - pitch / 2.0] + [(a + b) / 2.0 for a, b in zip(xs, xs[1:])] + [xs[-1] + pitch / 2.0]
    y0 = float(stations[0][1][0].crank_centre[1]); z0 = float(stations[0][1][0].crank_centre[2])
    centre_of = lambda x: np.array([x, y0, z0])
    radial = len(stations) == 1 and len(stations[0][1]) > 2

    if "expander" in kinds:
        # open crank on a bedplate: a pedestal at every main
        x_a, x_b = main_xs[0] - bore * 0.3, main_xs[-1] + bore * 0.3
        depth = bore * 0.35
        vtx, nrm = cuboid_mesh(np.array([(x_a + x_b) / 2.0, y0 - r_throw - bore * 0.55 - depth / 2.0, z0]),
                               np.array([(x_b - x_a) / 2.0, depth / 2.0, r_throw + bore * 0.6]))
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="bedplate"))
        for i, mx in enumerate(main_xs):
            h = r_throw + bore * 0.55
            vtx, nrm = cuboid_mesh(np.array([mx, y0 - h / 2.0, z0]), np.array([bore * 0.12, h / 2.0, bore * 0.3]))
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"main_pedestal_{i + 1}"))
        # What carries the open-bottomed cylinder over an open crank is
        # the FRAME: a side plate either side of the crank plane (a
        # traction engine's hornplates, a locomotive's frame plates)
        # rising from the bedplate to the cylinder's crank-end cover,
        # with the crosshead guide bars bolted between them. That is the
        # real structure -- the cylinder hangs from the frame, not from
        # thin air over the crank.
        top = max(float(g.base[1]) for _, gs in stations for g in gs)
        plate_t = bore * 0.05
        for sgn, name in ((-1.0, "frame_plate_left"), (1.0, "frame_plate_right")):
            zc = z0 + sgn * (r_throw + bore * 0.62)
            vtx, nrm = cuboid_mesh(np.array([(x_a + x_b) / 2.0, (y0 - r_throw - bore * 0.55 + top) / 2.0, zc]),
                                   np.array([(x_b - x_a) / 2.0, (top - (y0 - r_throw - bore * 0.55)) / 2.0, plate_t]))
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=name))
        for x, gs in stations:
            g = gs[0]
            if not g.crosshead:
                continue
            # guide bars either side of the crosshead's travel
            lo = y0 + g.rod_length_m - g.crank_radius_m - bore * 0.2
            hi = y0 + g.rod_length_m + g.crank_radius_m + bore * 0.2
            for sgn in (-1.0, 1.0):
                zc = z0 + sgn * (bore * 0.36)
                vtx, nrm = cuboid_mesh(np.array([x, (lo + hi) / 2.0, zc]), np.array([bore * 0.10, (hi - lo) / 2.0, bore * 0.04]))
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"guide_bar_{g.number}_{'l' if sgn < 0 else 'r'}"))
        return parts

    sealed = any(g.wall_ports == "loop" for _, gs in stations for g in gs)
    if radial:
        # a radial's case is a short drum the cylinders bolt around,
        # sized to the ring they sit on, with the single throw inside
        gs0 = stations[0][1]
        ring_r = max(float(np.linalg.norm(np.array(g.base) - np.array(g.crank_centre))) for g in gs0)
        drum_r = ring_r * 0.92
        x = stations[0][0]
        vtx, nrm = _tube(centre_of(x - bore * 0.55), centre_of(x + bore * 0.55), drum_r, sides=36)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=f"block_cyl_{gs0[0].number}", name="crankcase_drum"))
        vtx, nrm = _tube(centre_of(x - bore * 0.55 - bore * 0.1), centre_of(x - bore * 0.55), drum_r * 0.7, sides=30)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="front_cover"))
        vtx, nrm = _tube(centre_of(x + bore * 0.55), centre_of(x + bore * 0.55 + bore * 0.35), drum_r * 0.8, sides=30)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group="oil", name="rear_accessory_case"))
        return parts
    tunnel_r = r_throw + bore * 0.32
    # crank tunnel, one segment per station (that cylinder's own block metal)
    for k, (x, gs) in enumerate(stations):
        seg_a = centre_of(main_xs[k]); seg_b = centre_of(main_xs[k + 1])
        grp = f"block_cyl_{gs[0].number}"
        vtx, nrm = _tube(seg_a, seg_b, tunnel_r, sides=22)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"crankcase_tunnel_{k + 1}"))
        if sealed:
            # a loop-scavenged two-stroke's crankcase is its scavenge
            # pump: the intake (reed/piston port) enters the case
            g = gs[0]
            axis = _unit(g.axis)
            side = _unit(np.cross(CRANK_AXIS, axis))
            mouth = centre_of(x) - side * tunnel_r
            vtx, nrm = _tube(mouth, mouth - side * (bore * 0.25), bore * 0.13, sides=10)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group="intake", name=f"crankcase_intake_{k + 1}"))
    # bulkheads at every main
    for i, mx in enumerate(main_xs):
        grp = f"block_cyl_{stations[min(i, len(stations) - 1)][1][0].number}"
        vtx, nrm = _tube(centre_of(mx - bore * 0.035), centre_of(mx + bore * 0.035), tunnel_r * 1.06, sides=22)
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"crankcase_bulkhead_{i + 1}"))
    # skirts from the tunnel up to each bore's foot, along that bore's own axis
    for x, gs in stations:
        for g in gs:
            axis = _unit(g.axis)
            c = np.array(g.crank_centre)
            foot = np.array(g.base)
            wall = g.bore_m * 0.10
            vtx, nrm = _tube(c + axis * (tunnel_r * 0.55), foot + axis * (wall * 0.5), g.bore_m / 2.0 + wall * 2.2, sides=20)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=f"block_cyl_{g.number}",
                                   name=f"crankcase_skirt_{g.number}"))
    # end walls: front cover boss and rear main seal
    front = centre_of(main_xs[0] - bore * 0.06)
    vtx, nrm = _tube(front - CRANK_AXIS * (bore * 0.10), front, tunnel_r * 1.06, sides=22)
    parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="front_cover"))
    rear = centre_of(main_xs[-1] + bore * 0.06)
    vtx, nrm = _tube(rear, rear + CRANK_AXIS * (bore * 0.10), tunnel_r * 1.06, sides=22)
    parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="rear_main_seal_housing"))
    # wet sump below the tunnel (oil temperature) -- not on a sealed two-stroke case
    if not sealed:
        x_a, x_b = main_xs[0], main_xs[-1]
        depth = bore * 0.65
        vtx, nrm = cuboid_mesh(np.array([(x_a + x_b) / 2.0, y0 - tunnel_r * 0.6 - depth / 2.0, z0]),
                               np.array([(x_b - x_a) / 2.0 + bore * 0.05, depth / 2.0, tunnel_r * 0.85]))
        parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group="oil", name="sump"))
    return parts


def build_crank_train_parts(layout, crank_angle_deg: float = 0.0) -> list:
    return build_crankshaft_parts(layout, crank_angle_deg) + build_crankcase_parts(layout)
