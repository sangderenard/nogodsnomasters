"""Explicit fabricated connections for production graphs.

The graph used to call any fixed-ended centreline member a weld.  That says
how a beam end is released; it says nothing about what joins two surfaces.
This module authors the missing hardware: filler metal between weld toes and
a two-piece shaped strap around the outside of actual component profiles.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from fasteners import BOLT_CLASS, NUT_FACTOR, THREADS, CLAMP_FRICTION
from milspec import MATERIAL_BY_KEY
from welding import WeldQuality, shop_weld, weld_edge


@dataclass(frozen=True)
class WrapProfile:
    """One component's cross-section in the strap's local x/y plane."""
    body: str
    kind: str                    # circle | box
    centre: tuple[float, float]
    radius_m: float = 0.0
    half_extent_m: tuple[float, float] = (0.0, 0.0)

    def boundary(self, clearance_m: float,
                 facet_angle_deg: float) -> list[tuple[np.ndarray, np.ndarray]]:
        c = np.asarray(self.centre, float)
        if self.kind == "circle":
            if self.radius_m <= 0.0:
                raise ValueError(f"{self.body}: circle radius must be positive")
            count = max(8, int(math.ceil(360.0 / facet_angle_deg)))
            result = []
            for i in range(count):
                angle = 2.0 * math.pi * i / count
                normal = np.asarray((math.cos(angle), math.sin(angle)))
                result.append((c + normal * (self.radius_m + clearance_m),
                               c + normal * self.radius_m))
            return result
        if self.kind == "box":
            hx, hy = self.half_extent_m
            if hx <= 0.0 or hy <= 0.0:
                raise ValueError(f"{self.body}: box half extents must be positive")
            # Chamfered outside corners are machineable bends.  The convex
            # hull below bridges the inaccessible re-entrant gap between
            # components instead of forcing the strap into it.
            result = []
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                normal = np.asarray((sx, sy), float) / math.sqrt(2.0)
                surface = c + np.asarray((sx * hx, sy * hy))
                result.append((surface + normal * clearance_m, surface))
            return result
        raise ValueError(f"{self.body}: unsupported wrap profile {self.kind!r}")


@dataclass(frozen=True)
class _HullPoint:
    point: np.ndarray
    surface: np.ndarray
    body: str


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    u, v = a - o, b - o
    return float(u[0] * v[1] - u[1] * v[0])


def outside_perimeter(profiles: Iterable[WrapProfile], *, clearance_m: float,
                      facet_angle_deg: float = 22.5) -> list[_HullPoint]:
    """Convex machineable route around all profiles.

    A convex outside route deliberately skips sharp narrow interior spaces:
    those are not places a bent strap can be fitted or tightened against.
    Circle faceting limits each shop bend to ``facet_angle_deg``.
    """
    if clearance_m < 0.0:
        raise ValueError("strap clearance cannot be negative")
    if not 5.0 <= facet_angle_deg <= 60.0:
        raise ValueError("facet angle must be between 5 and 60 degrees")
    points = []
    for profile in profiles:
        points.extend(_HullPoint(p, surface, profile.body)
                      for p, surface in profile.boundary(
                          clearance_m, facet_angle_deg))
    if len(points) < 3:
        raise ValueError("a shaped strap needs at least three boundary points")
    ordered = sorted(points, key=lambda p: (float(p.point[0]),
                                            float(p.point[1]), p.body))
    lower: list[_HullPoint] = []
    for p in ordered:
        while len(lower) >= 2 and _cross(
                lower[-2].point, lower[-1].point, p.point) <= 1.0e-12:
            lower.pop()
        lower.append(p)
    upper: list[_HullPoint] = []
    for p in reversed(ordered):
        while len(upper) >= 2 and _cross(
                upper[-2].point, upper[-1].point, p.point) <= 1.0e-12:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("wrap profiles have no finite outside perimeter")
    return hull


def _rectangular_section(width_m: float, thickness_m: float,
                         material: str) -> dict:
    mat = MATERIAL_BY_KEY[material]
    area = width_m * thickness_m
    # local z is strap width; local y is thickness.
    iy = thickness_m * width_m ** 3 / 12.0
    iz = width_m * thickness_m ** 3 / 12.0
    a, b = max(width_m, thickness_m), min(width_m, thickness_m)
    torsion = a * b ** 3 * (1.0 / 3.0 - 0.21 * b / a
                            * (1.0 - b ** 4 / (12.0 * a ** 4)))
    return {
        "section_area_m2": area,
        "second_moment_y_m4": iy,
        "second_moment_z_m4": iz,
        "torsion_constant_m4": torsion,
        "section_outer_y_m": thickness_m / 2.0,
        "section_outer_z_m": width_m / 2.0,
        "youngs_modulus_pa": mat.youngs_pa,
        "shear_modulus_pa": mat.shear_pa,
        "yield_strength_pa": mat.yield_pa,
        "ultimate_strength_pa": mat.ultimate_pa,
        "fracture_strain": mat.elongation,
        "material": material,
    }


def _node(g, identity: str, xyz, *, kind="strap-fabrication-junction",
          **attributes) -> None:
    g.node(identity, xyz, kind, mass_kg=0.0, mass_in_total=False,
           fabrication_path_node=True, material=attributes.pop(
               "material", "steel-plate"), **attributes)


def emit_shaped_iron_strap(g, identity: str, profiles: Iterable[WrapProfile],
                           *, station_m: float, width_m: float = 0.065,
                           thickness_m: float = 0.010,
                           clearance_m: float = 0.003,
                           facet_angle_deg: float = 22.5,
                           material: str = "4130n", bolt: str = "M16",
                           bolt_class: str = "10.9",
                           set_torque_nm: float = 210.0,
                           contact: str = "steel-on-steel",
                           plane_axes: tuple[int, int] = (0, 1)) -> dict:
    """Emit two bent strap halves, two side screws and contact pads."""
    profiles = tuple(profiles)
    hull = outside_perimeter(
        profiles, clearance_m=clearance_m + thickness_m / 2.0,
        facet_angle_deg=facet_angle_deg)
    if width_m <= 0.0 or thickness_m <= 0.0:
        raise ValueError("strap width and thickness must be positive")
    if set_torque_nm <= 0.0:
        raise ValueError("strap tightening torque must be positive")
    thread = THREADS[bolt]
    diameter = thread.major_mm / 1000.0
    requested_preload = set_torque_nm / (NUT_FACTOR * diameter)
    proof_preload = thread.tensile_stress_area_m2 * BOLT_CLASS[bolt_class] * 0.9
    preload = min(requested_preload, proof_preload)
    achieved_torque = preload * NUT_FACTOR * diameter
    mu = CLAMP_FRICTION[contact]
    if len(set(plane_axes)) != 2 or any(i not in (0, 1, 2)
                                       for i in plane_axes):
        raise ValueError("plane_axes must name two distinct world axes")
    width_axis = next(i for i in (0, 1, 2) if i not in plane_axes)

    def world(point, width_coordinate=None):
        result = np.zeros(3, dtype=float)
        result[plane_axes[0]], result[plane_axes[1]] = point
        result[width_axis] = (float(station_m) if width_coordinate is None
                              else width_coordinate)
        return result

    section_up = np.zeros(3, dtype=float)
    section_up[width_axis] = 1.0

    right_i = max(range(len(hull)), key=lambda i: float(hull[i].point[0]))
    left_i = min(range(len(hull)), key=lambda i: float(hull[i].point[0]))
    if right_i == left_i:
        raise ValueError("strap side closures collapsed to one location")

    def cyclic_between(start, stop):
        out = []
        i = (start + 1) % len(hull)
        while i != stop:
            out.append(i)
            i = (i + 1) % len(hull)
        return out

    old_motion, old_assembly = g.motion_group, g.assembly
    g.assembly = "fabricated-shaped-strap"
    section = _rectangular_section(width_m, thickness_m, material)
    z = float(station_m)
    lug_gap = max(diameter * 1.5, thickness_m * 2.0)

    def split(side_i):
        before = hull[(side_i - 1) % len(hull)].point
        after = hull[(side_i + 1) % len(hull)].point
        tangent = after - before
        tangent /= max(float(np.linalg.norm(tangent)), 1.0e-12)
        centre = hull[side_i].point
        return (world(centre - tangent * lug_gap / 2.0),
                world(centre + tangent * lug_gap / 2.0), tangent)

    left_a, left_b, left_axis = split(left_i)
    right_a, right_b, right_axis = split(right_i)
    special = {"la": left_a, "lb": left_b,
               "ra": right_a, "rb": right_b}
    for tag, pos in special.items():
        _node(g, f"{identity}.{tag}", pos, material=material,
              part_role="strap-tensioner-lug")

    arcs = (("a", "rb", cyclic_between(right_i, left_i), "la"),
            ("b", "lb", cyclic_between(left_i, right_i), "ra"))
    perimeter_nodes = {}
    contact_count = 0
    for arc, first, indices, last in arcs:
        names = [f"{identity}.{first}"]
        for order, i in enumerate(indices):
            name = f"{identity}.{arc}.{order}"
            hp = hull[i]
            _node(g, name, world(hp.point), material=material,
                  part_role="shaped-strap-bend")
            names.append(name)
            perimeter_nodes[name] = hp
        names.append(f"{identity}.{last}")
        for i, (a, b) in enumerate(zip(names, names[1:])):
            g.edge(f"{identity}.{arc}.segment.{i}", a, b, "rigid-distance",
                   radius=math.sqrt(width_m * thickness_m / math.pi),
                   wall_m=thickness_m, alloy=material,
                   section_shape="flat-strap", section_width_m=width_m,
                   section_thickness_m=thickness_m,
                   section_up=tuple(float(v) for v in section_up),
                   section_properties=section,
                   palette="rollbar-silver", part_role="shaped-iron-strap",
                   load_path="bent-strap-around-the-outside-perimeter")

    screw_area = thread.tensile_stress_area_m2
    tightening_axes = []
    for side, a, b, axis in (("left", "la", "lb", left_axis),
                             ("right", "ra", "rb", right_axis)):
        screw_length = float(np.linalg.norm(special[b] - special[a]))
        screw_stiffness = (MATERIAL_BY_KEY["4340qt"].youngs_pa * screw_area
                           / max(screw_length, 1.0e-6))
        axis_world = np.zeros(3, dtype=float)
        axis_world[plane_axes[0]], axis_world[plane_axes[1]] = axis
        axis3 = tuple(float(v) for v in axis_world)
        g.edge(f"{identity}.tensioner.{side}", f"{identity}.{a}",
               f"{identity}.{b}", "strap-tensioner",
               radius=diameter / 2.0, alloy="4340qt",
               stiffness_n_per_m=screw_stiffness, preload_force_n=preload,
               bolt=bolt, bolt_class=bolt_class,
               set_torque_nm=achieved_torque,
               requested_set_torque_nm=set_torque_nm,
               tightening_axis=axis3, adjustable=True, removable=True,
               palette="chassis-grey", part_role="strap-tensioner")
        tightening_axes.append(axis3)

    # Bind every exposed hull support to its originating physical surface.
    # The linear beam solve uses the locked, below-slip state; capacities are
    # explicit so a contact/friction pass can release rather than fracture it.
    normal_share = preload * 4.0 / max(len(perimeter_nodes), 1)
    for strap_node, hp in perimeter_nodes.items():
        port = f"{identity}.contact.{contact_count}.surface"
        _node(g, port, world(hp.surface),
              kind="structural-mount-port", material=material,
              wrench_point=True, surface_of=hp.body,
              solver_condensed_into=hp.body, solver_condensed_mass=False,
              part_role="strap-contact-surface")
        g.edge(f"{identity}.contact.{contact_count}", port, strap_node,
               "strap-clamped-contact", radius=max(thickness_m / 2.0, 0.003),
               alloy=material, rigid=True, palette="chassis-grey",
               part_role="preloaded-strap-contact",
               normal_preload_n=normal_share,
               friction_coefficient=mu,
               axial_slip_capacity_n=mu * normal_share,
               contact_regime="locked-below-slip-capacity",
               failure_response="contact-slips-without-fracture")
        contact_count += 1
    g.motion_group, g.assembly = old_motion, old_assembly
    return {
        "identity": identity,
        "perimeter": [tuple(float(v) for v in hp.point) for hp in hull],
        "closures": 2,
        "tightening_axes": tuple(tightening_axes),
        "set_torque_nm": achieved_torque,
        "preload_force_n_per_closure": preload,
        "contact_count": contact_count,
    }


def weld_touching(g, identity: str, body_a: str, body_b: str,
                  toe_pairs: Iterable[tuple[tuple[float, float, float],
                                             tuple[float, float, float]]], *,
                  throat_m: float, quality: WeldQuality | None = None,
                  restraint: float = 1.0,
                  stress_relieved: bool = False) -> list[str]:
    """Weld nearby surface toes and reject a disguised centreline member."""
    quality = quality or shop_weld()
    if throat_m <= 0.0:
        raise ValueError("weld throat must be positive")
    node_ids = {node["identity"] for node in g.nodes}
    if body_a not in node_ids or body_b not in node_ids:
        raise KeyError("both welded parent bodies must already exist")
    made = []
    old_assembly = g.assembly
    g.assembly = "fabricated-weld"
    for i, (raw_a, raw_b) in enumerate(toe_pairs):
        pa, pb = np.asarray(raw_a, float), np.asarray(raw_b, float)
        gap = float(np.linalg.norm(pb - pa))
        if gap <= 1.0e-7:
            raise ValueError(f"{identity}.{i}: distinct weld toe points required")
        if gap > throat_m * 3.0:
            raise ValueError(
                f"{identity}.{i}: {gap * 1000:.1f} mm toe gap is not touching "
                f"for a {throat_m * 1000:.1f} mm throat")
        a, b = f"{identity}.{i}.toe.a", f"{identity}.{i}.toe.b"
        for name, point, parent in ((a, pa, body_a), (b, pb, body_b)):
            _node(g, name, point, kind="structural-mount-port",
                  material=quality.parent_alloy, wrench_point=True,
                  surface_of=parent, solver_condensed_into=parent,
                  solver_condensed_mass=False, part_role="weld-toe")
        perimeter = gap * 2.0
        g.edge(f"{identity}.{i}", a, b, "welded-joint-element",
               radius=max(throat_m / 2.0, 0.001),
               alloy=quality.parent_alloy, palette="weld-bead",
               part_role="explicit-weld-bead")
        edge = g.edges[-1]
        retained = {k: edge[k] for k in (
            "constraint_token", "freedoms_released", "release_ends",
            "routed", "assembly")}
        edge.update(weld_edge(
            edge["identity"], a, b, quality=quality, throat_m=throat_m,
            perimeter_m=perimeter, gap_m=gap, restraint=restraint,
            stress_relieved=stress_relieved))
        edge.update(retained)
        edge.update(section_shape="weld-bead", weld_throat_m=throat_m,
                    weld_perimeter_m=perimeter,
                    parent_bodies=(body_a, body_b),
                    load_path="filler-metal-between-explicit-touching-toes")
        made.append(edge["identity"])
    g.assembly = old_assembly
    if not made:
        raise ValueError("weld_touching requires at least one toe pair")
    return made
