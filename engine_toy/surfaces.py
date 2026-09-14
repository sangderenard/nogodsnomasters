"""SURFACES AS STRUCTURE: annuli and plates that actually carry.

Everything structural in this project has been a TUBE between two
points. That is most of a space frame and it is none of a deck, a
mantlet, a bulkhead or a slew-ring seat -- and those were being modelled
as either nothing at all or as a lattice of sticks pretending to be a
sheet.

CAN AN ANNULUS BE BEAM-SOLVED? Yes, and not by pretending it is a plate.
A closed circular ring IS a curved beam, and the classical result for
one is that an out-of-plane load -- exactly what a turret overhang is --
is carried by BENDING AND TORSION TOGETHER, in proportions set by the
arc between supports. That is why a ring seat works at all and why a
straight beam of the same section would not.

`frame_solver.element_stiffness` is a 3D Timoshenko element: it already
has J, it already has torsion, and it already has the shear coefficient
that matters for a section this stubby. So an annulus needs no new
solver. It needs to be EMITTED correctly -- as arc segments carrying the
annulus's own rectangular section, closed on itself -- and the existing
frame solve does the rest.

THE TWO SECOND MOMENTS ARE NOT INTERCHANGEABLE and getting them the
wrong way round is the whole difference between a ring that works and
one that does not:

    out of plane   I = b t^3 / 12     (radial width b, thickness t)
    in plane       I = t b^3 / 12

A ring seat is wide and thin, so b >> t and the in-plane stiffness is
enormous while the out-of-plane stiffness -- the one the overhang
loads -- is the small one. A solver handed a single "radius" for a
member cannot tell those apart, which is why an annulus declares its
section rather than borrowing a tube's.

A PLATE IS THE SAME IDEA WITHOUT THE CURVATURE: a surface carried by an
orthogonal grid of strips, each a beam of the plate's own section, tied
at every crossing. Welded to the members it lands on through DECLARED
SEAM JOINTS, so a plate welded along one edge and free along the others
is a different structure from one welded all round -- and the graph says
which, instead of both being drawn the same and solved the same.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def rectangular_section(width_m: float, thickness_m: float) -> dict:
    """The section properties of a flat strip, both ways round.

    `radius` is what the existing edge API wants, so an equivalent one
    is derived that reproduces the OUT-OF-PLANE second moment -- the
    one that governs -- rather than the area or the in-plane stiffness.
    A strip is not a tube and the equivalence can only be exact in one
    property; this says which one, instead of leaving a reader to find
    out by being surprised."""
    b, t = float(width_m), float(thickness_m)
    area = b * t
    i_out = b * t ** 3 / 12.0
    i_in = t * b ** 3 / 12.0
    # St Venant torsion of a thin rectangle
    a, c = max(b, t), min(b, t)
    j = a * c ** 3 * (1.0 / 3.0 - 0.21 * c / a * (1.0 - c ** 4
                                                  / (12.0 * a ** 4)))
    r_eq = (4.0 * i_out / math.pi) ** 0.25
    return {"area_m2": area, "second_moment_out_m4": i_out,
            "second_moment_in_m4": i_in, "torsion_constant_m4": j,
            "equivalent_radius_m": r_eq, "width_m": b, "thickness_m": t}


def ring_frame(axis) -> tuple:
    """The two in-plane directions a ring's angles are measured in.

    THE CONVENTION HAS TO MATCH THE GRAPH'S. Everything already in this
    project places a ring as (r cos a, y, r sin a) about +y. A frame
    built by the usual `cross(axis, helper)` recipe gives (-sin, 0,
    -cos) instead -- ninety degrees round and mirrored -- so station i
    of a new ring landed OPPOSITE station i of the old one, and every
    member between them ran through the middle of the machine. The
    clear-volume check caught it as eight members crossing the case
    ejection column, which is exactly what it is for.

    So u is a reference direction projected into the plane, and v is
    cross(u, axis) -- not cross(axis, u), which is the same two vectors
    in the handedness that mirrors."""
    ax = np.asarray(axis, float)
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    ref = (np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9
           else np.array([0.0, 0.0, 1.0]))
    u = ref - ax * float(np.dot(ref, ax))
    u = u / max(np.linalg.norm(u), 1e-12)
    return u, np.cross(u, ax)


@dataclass
class Annulus:
    """A flat ring, emitted as the closed curved beam it is."""
    identity: str
    centre: tuple
    inner_radius_m: float
    outer_radius_m: float
    thickness_m: float
    segments: int = 16
    axis: tuple = (0.0, 1.0, 0.0)
    material: str = "steel-plate"
    alloy: str = "a36"
    #: the outer band, which may be a different metal from the web
    rim_material: str = "steel-plate"
    attributes: dict = field(default_factory=dict)

    @property
    def mean_radius_m(self) -> float:
        return (self.inner_radius_m + self.outer_radius_m) / 2.0

    @property
    def radial_width_m(self) -> float:
        return self.outer_radius_m - self.inner_radius_m

    @property
    def section(self) -> dict:
        return rectangular_section(self.radial_width_m, self.thickness_m)

    @property
    def mass_kg(self) -> float:
        area = math.pi * (self.outer_radius_m ** 2 - self.inner_radius_m ** 2)
        return area * self.thickness_m * 7850.0

    def _ring_point(self, i: int, radius: float) -> tuple:
        a = 2.0 * math.pi * i / self.segments
        c = np.asarray(self.centre, float)
        u, v = ring_frame(self.axis)
        p = c + (u * math.cos(a) + v * math.sin(a)) * radius
        return tuple(float(x) for x in p)


def emit_annulus(g, ring: Annulus, *, motion_group: str | None = None,
                 assembly: str | None = None) -> dict:
    """Write the ring into a production graph and hand back its stations.

    Returns {"mean": [...], "outer": [...]} -- the mean-radius nodes the
    ring's own bending runs through, and the OUTER EDGE nodes, which is
    where a surface-to-surface captive joint belongs. Not the mean
    radius: the contact patch is the rim, and putting the joint on the
    centreline understates the lever every bearing on it works at."""
    if motion_group is not None:
        g.motion_group = motion_group
    if assembly is not None:
        g.assembly = assembly
    sec = ring.section
    per = ring.mass_kg / ring.segments
    # ---- THE RING DRAWS AS ITS OWN SEGMENTS ----
    # An earlier attempt gave the annulus one `shape="tube"` body and it
    # came out a square slab: `tube_outer_radius_m` is written by four
    # modules in this project and READ BY NO MESH BUILDER on this path,
    # so the primitive silently fell back to the box half-extent. Round
    # bodies do exist in the renderer -- vehicle_mesh has capped_tube and
    # beveled_drum keyed on `drum_radius_m` -- but engine_mesh, which is
    # what EngineGLView builds with, draws node bodies as boxes.
    #
    # So the ring is drawn by its stations, each a wedge of the real
    # plate: radial width by thickness by the chord it spans. At enough
    # segments that is a triangulated circle, which is what a renderer
    # draws anyway.
    # ONE PARAMETRIC DISC TO DRAW, N SEGMENTS TO SOLVE. Different
    # questions, and they were being answered by one object: the
    # stations are the curved-beam discretisation and drawing each of
    # them as its own box made the beam model visible as a cog.
    g.node(f"{ring.identity}.disc", tuple(float(v) for v in ring.centre),
           "load-bearing-structure", material=ring.material,
           mass_in_total=False, mass_kg=0.01,
           solver_condensed_into=f"{ring.identity}.station.0",
           solver_condensed_mass=False,
           part_role="annulus-disc", annulus=ring.identity,
           shape="annulus", annulus_axis=tuple(float(v) for v in ring.axis),
           inner_radius_m=ring.inner_radius_m,
           outer_radius_m=ring.outer_radius_m,
           thickness_m=ring.thickness_m, draw_segments=64,
           is_drawing_body=True,
           half_extent_m=(ring.outer_radius_m, ring.thickness_m / 2.0,
                          ring.outer_radius_m))
    mean, outer = [], []
    for i in range(ring.segments):
        mid = f"{ring.identity}.station.{i}"
        g.node(mid, ring._ring_point(i, ring.mean_radius_m),
               "load-bearing-structure", material=ring.material,
               mass_in_total=False, mass_kg=round(per, 2),
               part_role="annulus-station", annulus=ring.identity,
               in_view=False,
               inner_radius_m=ring.inner_radius_m,
               outer_radius_m=ring.outer_radius_m,
               thickness_m=ring.thickness_m,
               second_moment_out_m4=sec["second_moment_out_m4"],
               second_moment_in_m4=sec["second_moment_in_m4"],
               torsion_constant_m4=sec["torsion_constant_m4"],
               half_extent_m=(ring.radial_width_m / 2.0,
                              ring.thickness_m / 2.0,
                              max(math.pi * ring.mean_radius_m
                                  / ring.segments, 0.01)),
               **ring.attributes)
        mean.append(mid)
        rim = f"{ring.identity}.rim.{i}"
        g.node(rim, ring._ring_point(i, ring.outer_radius_m),
               "structural-mount-port", material=ring.rim_material,
               wrench_point=True, surface_of=ring.identity,
               solver_condensed_into=mid, solver_condensed_mass=False,
               mass_in_total=False, mass_kg=0.4,
               part_role="annulus-rim", surface="outer", in_view=False,
               half_extent_m=(0.03, ring.thickness_m / 2.0, 0.03))
        outer.append(rim)
    for i in range(0, ring.segments, max(ring.segments // 4, 1)):
        g.edge(f"{ring.identity}.disc_tie.{i}", f"{ring.identity}.disc",
               mean[i], "rigid-distance", radius=0.008, rigid=True,
               beam_solvable=False, in_view=False, palette="rollbar-silver",
               structural_participation=False,
               load_path="the-drawn-disc-riding-with-its-own-beam-model")
    for i in range(ring.segments):
        j = (i + 1) % ring.segments
        # THE ARC ITSELF, as a beam. Closed on itself, so the ring can
        # develop the hoop action that makes it stiffer than the sum of
        # its segments -- an open ring is a spring.
        g.edge(f"{ring.identity}.arc.{i}", mean[i], mean[j],
               "rigid-distance", radius=sec["equivalent_radius_m"],
               wall_m=ring.thickness_m / 2.0, alloy=ring.alloy,
               palette="chassis-grey", beam_solvable=True,
               section="annular-arc", annulus=ring.identity,
               second_moment_out_m4=sec["second_moment_out_m4"],
               second_moment_in_m4=sec["second_moment_in_m4"],
               torsion_constant_m4=sec["torsion_constant_m4"],
               load_path="the-ring-carrying-its-own-arc-in-bending-and-torsion")
        # the rim is part of the same piece of plate, not a fitting
        g.edge(f"{ring.identity}.web.{i}", outer[i], mean[i],
               "rigid-distance", radius=0.006, rigid=True,
               beam_solvable=False, palette="rollbar-silver",
               load_path="annulus-rim-is-the-same-plate-as-its-station")
    return {"mean": mean, "outer": outer, "section": sec}


@dataclass
class Plate:
    """A flat surface carried by an orthogonal grid of its own strips."""
    identity: str
    corner: tuple            # one corner, in world
    span_u: tuple            # edge vector
    span_v: tuple            # the other edge vector
    thickness_m: float
    nu: int = 4
    nv: int = 3
    material: str = "steel-plate"
    alloy: str = "a36"
    attributes: dict = field(default_factory=dict)

    def point(self, i: int, j: int) -> tuple:
        c = np.asarray(self.corner, float)
        u = np.asarray(self.span_u, float) * (i / self.nu)
        v = np.asarray(self.span_v, float) * (j / self.nv)
        return tuple(float(x) for x in (c + u + v))

    @property
    def mass_kg(self) -> float:
        area = float(np.linalg.norm(np.cross(self.span_u, self.span_v)))
        return area * self.thickness_m * 7850.0


def emit_plate(g, plate: Plate, *, motion_group: str | None = None,
               assembly: str | None = None,
               seam: dict | None = None) -> dict:
    """A plate, and the seam that welds it to what it lands on.

    `seam` maps an edge name ("u0", "u1", "v0", "v1") to the node each
    station on that edge is joined to, and the constraint to join it
    with. THAT is what makes this a declaration rather than a picture: a
    plate welded along one edge and free along the other three is a
    different structure from one welded all round, and both were
    previously drawn identically because neither was drawn at all."""
    if motion_group is not None:
        g.motion_group = motion_group
    if assembly is not None:
        g.assembly = assembly
    du = float(np.linalg.norm(plate.span_u)) / plate.nu
    dv = float(np.linalg.norm(plate.span_v)) / plate.nv
    su = rectangular_section(dv, plate.thickness_m)
    sv = rectangular_section(du, plate.thickness_m)
    plate_normal = np.cross(np.asarray(plate.span_u, float),
                            np.asarray(plate.span_v, float))
    plate_normal /= max(float(np.linalg.norm(plate_normal)), 1.0e-12)
    u_hat = np.asarray(plate.span_u, float)
    u_hat /= max(float(np.linalg.norm(u_hat)), 1.0e-12)
    v_hat = np.asarray(plate.span_v, float)
    v_hat /= max(float(np.linalg.norm(v_hat)), 1.0e-12)
    section_up = tuple(float(v) for v in plate_normal)
    per = plate.mass_kg / ((plate.nu + 1) * (plate.nv + 1))
    ids = {}
    for i in range(plate.nu + 1):
        for j in range(plate.nv + 1):
            ident = f"{plate.identity}.node.{i}.{j}"
            # A grid station is a structural point, not the centre of a
            # full-size visual tile.  Giving every station du/2,dv/2 extents
            # made the boundary stations overhang the authored plate by half
            # a cell -- commonly several decimetres.  Boundary stations own
            # only the inward half-cell, shifted inward; interior stations
            # own the centered cell between their neighbours.  Their union
            # is therefore exactly [corner, corner + span_u + span_v].
            hu = du / 4.0 if i in (0, plate.nu) else du / 2.0
            hv = dv / 4.0 if j in (0, plate.nv) else dv / 2.0
            ou = du / 4.0 if i == 0 else (-du / 4.0
                                          if i == plate.nu else 0.0)
            ov = dv / 4.0 if j == 0 else (-dv / 4.0
                                          if j == plate.nv else 0.0)
            render_offset = u_hat * ou + v_hat * ov
            g.node(ident, plate.point(i, j), "load-bearing-structure",
                   material=plate.material, mass_in_total=False,
                   mass_kg=round(per, 2), part_role="plate-station",
                   plate=plate.identity, thickness_m=plate.thickness_m,
                   shape="plate-cell",
                   plate_cell_axes=(tuple(float(x) for x in u_hat),
                                    tuple(float(x) for x in v_hat),
                                    tuple(float(x) for x in plate_normal)),
                   plate_cell_half_extent_m=(hu, hv,
                                             plate.thickness_m / 2.0),
                   render_center_offset_m=tuple(float(x)
                                                for x in render_offset),
                   half_extent_m=(hu, plate.thickness_m / 2.0, hv),
                   **plate.attributes)
            ids[(i, j)] = ident
    for i in range(plate.nu + 1):
        for j in range(plate.nv + 1):
            if i < plate.nu:
                g.edge(f"{plate.identity}.u.{i}.{j}", ids[(i, j)],
                       ids[(i + 1, j)], "rigid-distance",
                       radius=su["equivalent_radius_m"],
                       wall_m=plate.thickness_m / 2.0, alloy=plate.alloy,
                       palette="chassis-grey", beam_solvable=True,
                       section="plate-strip", plate=plate.identity,
                       section_up=section_up,
                       section_properties={
                           "section_area_m2": su["area_m2"],
                           "second_moment_m4": su["second_moment_out_m4"],
                           "second_moment_y_m4": su["second_moment_out_m4"],
                           "second_moment_z_m4": su["second_moment_in_m4"],
                           "torsion_constant_m4": su["torsion_constant_m4"],
                           "section_outer_y_m": su["width_m"] / 2.0,
                           "section_outer_z_m": su["thickness_m"] / 2.0},
                       load_path="plate-strip-carrying-across-its-width")
            if j < plate.nv:
                g.edge(f"{plate.identity}.v.{i}.{j}", ids[(i, j)],
                       ids[(i, j + 1)], "rigid-distance",
                       radius=sv["equivalent_radius_m"],
                       wall_m=plate.thickness_m / 2.0, alloy=plate.alloy,
                       palette="chassis-grey", beam_solvable=True,
                       section="plate-strip", plate=plate.identity,
                       section_up=section_up,
                       section_properties={
                           "section_area_m2": sv["area_m2"],
                           "second_moment_m4": sv["second_moment_out_m4"],
                           "second_moment_y_m4": sv["second_moment_out_m4"],
                           "second_moment_z_m4": sv["second_moment_in_m4"],
                           "torsion_constant_m4": sv["torsion_constant_m4"],
                           "section_outer_y_m": sv["width_m"] / 2.0,
                           "section_outer_z_m": sv["thickness_m"] / 2.0},
                       load_path="plate-strip-carrying-along-its-length")
    # ---- THE SEAM ----
    edges = {"u0": [(0, j) for j in range(plate.nv + 1)],
             "u1": [(plate.nu, j) for j in range(plate.nv + 1)],
             "v0": [(i, 0) for i in range(plate.nu + 1)],
             "v1": [(i, plate.nv) for i in range(plate.nu + 1)]}
    welded = {}
    for name, spec in (seam or {}).items():
        target = spec["to"] if isinstance(spec, dict) else spec
        constraint = (spec.get("constraint", "rigid-distance")
                      if isinstance(spec, dict) else "rigid-distance")
        for k, key in enumerate(edges[name]):
            g.edge(f"{plate.identity}.seam.{name}.{k}", ids[key],
                   target(key) if callable(target) else target,
                   constraint, radius=plate.thickness_m,
                   alloy=plate.alloy, palette="rollbar-silver",
                   part_role="plate-seam", seam=name,
                   load_path="plate-welded-along-this-seam")
        welded[name] = len(edges[name])
    return {"nodes": ids, "edges": edges, "welded": welded}


@dataclass
class BoxedRing:
    """Two annuli and the cylinder between them: one closed cell.

    WHY BOXING IS NOT DECORATION. A flat ring resisting an out-of-plane
    load does it by bending and TWISTING -- an open strip has almost no
    torsional stiffness, so the twist is most of the deflection and
    stiffening the ring in bending barely helps. Close the section into
    a cell and the twist is carried by shear flow all the way round
    instead, and the torsion constant goes from the thin-rectangle
    value to Bredt's:

        open cell    J ~ sum(b t^3 / 3)      one strip at a time
        closed cell  J = 4 A_enclosed^2 / (perimeter / t)

    For a ring 300 mm deep and 40 mm thick that is three orders of
    magnitude. It is the single largest structural return available
    anywhere in this machine and it costs a cylinder wall.

    AND THE BOX IS WHERE EVERYTHING ALREADY WANTED TO LIVE:

        the TOP annulus      the truss lands on it, welded, anywhere
        the top OUTER rim    captive bearing, pressed
        the MIDDLE band      the gear the pinions run in
        the bottom OUTER rim the other captive bearing
        the BOTTOM annulus   bolted to the platform

    which is a real slew bearing rather than a ring with fittings hung
    off it. The two raceways take the couple as a push/pull pair at full
    depth, so the overhang is reacted across the box's height instead of
    across a single face that can only push.
    """
    identity: str
    centre: tuple
    inner_radius_m: float
    outer_radius_m: float
    height_m: float
    plate_thickness_m: float = 0.040
    wall_thickness_m: float = 0.025
    segments: int = 16
    axis: tuple = (0.0, 1.0, 0.0)
    #: THE BODY IS CASTABLE; THE RIM IS A DIFFERENT PROBLEM.
    #:
    #: A site that does its own reloading can run a foundry, and
    #: aluminium is what an outpost can actually pour: 6061 casts,
    #: welds and machines with equipment it plausibly has, where a
    #: steel plate three metres across is a rolling mill's problem.
    #: So the drum body -- both annuli, both walls, the drive
    #: machining -- is aluminium, and can be REPLACED IN THE FIELD
    #: if it ever cracks.
    #:
    #: The rim cannot be and does not want to be. The outer lip is
    #: where the races press and its whole duty is to be comfortable
    #: being smashed: Hertzian contact under preloaded rollers,
    #: forever, at the largest radius in the machine. 6061 yields at
    #: 276 MPa and that contact runs near 700; 4340 quenched and
    #: tempered yields at 1080. So the last band of radius is a
    #: STEEL ring let into the aluminium, carrying the raceway.
    #:
    #: Which is also why this is robust rather than merely strong:
    #: the part that wears is small, standard and replaceable, and
    #: the part that is big and awkward never touches a roller.
    material: str = "6061t6"
    alloy: str = "6061t6"
    #: the raceway band, steel whatever the body is
    rim_material: str = "4340qt"
    #: THE PASSAGE PROTRUDES INTO THE PLATES. The drive cavity is not
    #: confined to the gap between top and bottom: it is pocketed UP
    #: into the top plate and DOWN into the bottom one, so the usable
    #: height is the gap plus both pockets rather than a fraction of the
    #: gap. That is most of the volume, and it costs nothing structural
    #: because it is taken from the middle of the plate where the
    #: bending stress is lowest.
    drive_pocket_m: float = 0.025
    #: ...BUT THE OUTER MATING FLOORS DO NOT MOVE. The last
    #: `mating_band_m` of radius stays full-thickness plate, top and
    #: bottom, because that band IS the sliding surface -- greased,
    #: running on the static ring, and the thing the races are pressed
    #: against. Pocket into it and the bearing face becomes a membrane.
    #: The two floors do not need to stand apart for this: they keep
    #: their plane and keep sliding.
    mating_band_m: float = 0.180
    #: clearance left to whatever is machined in
    drive_fill: float = 0.90
    #: THE AXIAL STACK-UP INSIDE THE CAVITY. Everything -- donut,
    #: carrier plate, thrust race, service hub -- has to fit between
    #: the two pocket floors, and it did not: the donut was made the
    #: full cavity height so it pressed into both floors, and the
    #: carrier and hub were placed above it, straight through the top
    #: plate and out the other side. Declared here and subtracted in
    #: order, so the stack cannot silently overrun again.
    axial_clearance_m: float = 0.004
    carrier_thickness_m: float = 0.030
    thrust_race_m: float = 0.014
    #: ---- THE TRIPLE-SOURCE ROTOR, OUTSIDE IN ----
    #: The casing is already there as structure and as the traction
    #: surface, so making it the magnetic gap as well costs only the
    #: wall thickness it wanted anyway. Inboard of it the space is
    #: spent on two pressure rotors, and the order is set by what each
    #: one is worth per millimetre of radius: hydraulic first because
    #: 210 bar makes twenty times what 10 bar makes in the same swept
    #: volume, pneumatic in the core where the radius is cheap.
    #:
    #: A BOUNDARY-LAYER DISC STACK WAS CONSIDERED AND REJECTED. Air's
    #: viscosity is 1.8e-5, so sixty discs at a 0.4 mm gap make 0.76
    #: N.m against the vane rotor's 1610 in the same space -- four
    #: orders of magnitude. Filled with oil it gains a factor of 1600
    #: and is still an eighth of the hydraulic vane. It is not a motor.
    #: It would be an excellent viscous coupling, which is a different
    #: job and belongs on the clutch, not in the rotor.
    magnetic_band_m: float = 0.045
    hydraulic_band_m: float = 0.090
    airgap_shear_pa: float = 25_000.0
    vane_fill: float = 0.15
    #: the lock-up clamps harder than the metering spring, and stacks
    #: plates to reach direct drive without a huge single face
    lockup_clamp_ratio: float = 1.6
    lockup_plates: int = 4
    #: what the service hubs are rated to pass
    oil_pressure_pa: float = 2.1e7
    air_pressure_pa: float = 1.0e6
    unit_electrical_w: float = 45_000.0
    #: what is actually fitted into that envelope, in kilogrammes. The
    #: envelope itself is a void: it has a volume, not a mass.
    installed_drive_kg: float = 0.0
    #: HOW MANY CIRCULAR MOTORS SIT IN THE INNER SPACE, arranged around
    #: it the way rolling elements are arranged in an ordinary bearing:
    #: a ring formation of complete circles, not concentric rings and
    #: not a change to the bearing races, which stay bearings.
    drive_units: int = 6
    #: one entry per unit, cycled: what medium each is
    drive_mix: tuple = ("electric", "hydraulic", "pneumatic")
    #: RADIAL CLEARANCE EACH SIDE AT REST. The donuts do not touch
    #: either track until a clutch pushes them onto one -- this is the
    #: gap they float in, and twice it is the travel from full
    #: engagement on the inner track to full engagement on the outer.
    track_clearance_m: float = 0.006
    #: WHAT A FULLY APPLIED CLUTCH PRESSES WITH. The spring rate is
    #: derived from this and the travel rather than typed in, so
    #: asking for more bite makes a stiffer spring instead of silently
    #: bottoming the one that is there.
    clutch_engagement_n: float = 180_000.0
    #: the inner track is a separate hard ring, NOT the access tube
    inner_track_width_m: float = 0.070
    inner_track_material: str = "4340qt"

    def _face(self, which: str) -> Annulus:
        sign = -0.5 if which == "lower" else 0.5
        c = np.asarray(self.centre, float) + \
            np.asarray(self.axis, float) * (sign * self.height_m)
        return Annulus(identity=f"{self.identity}.{which}",
                       centre=tuple(float(x) for x in c),
                       inner_radius_m=self.inner_radius_m,
                       outer_radius_m=self.outer_radius_m,
                       thickness_m=self.plate_thickness_m,
                       segments=self.segments, axis=self.axis,
                       material=self.material, alloy=self.alloy,
                       rim_material=self.rim_material)

    @property
    def closed_torsion_m4(self) -> float:
        """Bredt: 4 A^2 / integral(ds/t), for the box's own cell."""
        b = self.outer_radius_m - self.inner_radius_m
        h = self.height_m
        area = b * h
        s_over_t = (2.0 * b / self.plate_thickness_m
                    + 2.0 * h / self.wall_thickness_m)
        return 4.0 * area ** 2 / max(s_over_t, 1e-9)

    @property
    def open_torsion_m4(self) -> float:
        b = self.outer_radius_m - self.inner_radius_m
        return 2.0 * (b * self.plate_thickness_m ** 3 / 3.0) \
            + 2.0 * (self.height_m * self.wall_thickness_m ** 3 / 3.0)


def emit_boxed_ring(g, box: BoxedRing, *, motion_group: str | None = None,
                    assembly: str | None = None) -> dict:
    """The box, and every surface anything else wants to attach to."""
    # THE POCKET COMES OUT OF THE PLATES. They are emitted as full
    # annuli and then the machining is deducted, because the metal is
    # removed from them -- counting the plates whole AND the cavity
    # empty would be right in neither direction.
    lower = emit_annulus(g, box._face("lower"), motion_group=motion_group,
                         assembly=assembly)
    upper = emit_annulus(g, box._face("upper"), motion_group=motion_group,
                         assembly=assembly)
    sec = rectangular_section(box.height_m, box.wall_thickness_m)
    # the pocket stops short of the mating band, so the drive annulus
    # is the inboard part of the middle and the rim stays solid
    _drive_outer = box.outer_radius_m - box.mating_band_m
    _drive_area = math.pi * (_drive_outer ** 2 - box.inner_radius_m ** 2)
    _usable_h = (box.height_m + 2.0 * box.drive_pocket_m) * box.drive_fill
    # ---- THE INNER SPACE IS CIRCULAR MOTORS ----
    # Six of them, set around the drive annulus the way rolling elements
    # are set around an ordinary bearing: whole circles at intervals,
    # each one a motor in its own right. Mixed media on purpose --
    # electric holds a position and resolves fine, hydraulic makes the
    # torque, pneumatic dumps a fast slug of it -- and they act
    # together, which is what whipping a nine-metre gun onto a moving
    # mark actually needs.
    #
    # AT THIS RADIUS TORQUE IS CHEAP. Every newton a unit makes is
    # multiplied by more than a metre and a half before it reaches the
    # gun, so six modest machines are a very large moment.
    # ---- TWO TRACKS, AND THE DONUTS RUN BETWEEN THEM ----
    # The outer track is a RIM HANGING DOWN from the top annulus, just
    # inside the bearing clearance. The inner track is a hard iron ring
    # standing on the bottom annulus -- a separate part from the access
    # trunk the crew climbs, which carries nobody's traction.
    #
    # THE DONUTS LIE FLAT BETWEEN THEM, axes parallel to the drum's own,
    # each one touching the outer rim on its outboard face and the inner
    # ring on its inboard face. That is a traction planetary: inner
    # track sun, donuts planets, outer rim the ring gear. Their axes do
    # NOT point radially -- a radial axis would roll a donut along the
    # gap instead of around it, and it would drive nothing.
    #
    # AND BOTH CONTACTS CAN BE CLUTCHED. Two engageable surfaces per
    # unit, six units, three media: hold the outer and release the
    # inner and a donut freewheels; grip both and it reacts; grip some
    # and slip others and the ring negotiates whatever motion is asked
    # for -- which is what aiming a nine-metre gun at something moving
    # actually requires.
    _u, _v = ring_frame(box.axis)
    _c = np.asarray(box.centre, float)
    _ax = np.asarray(box.axis, float)
    _ax = _ax / max(np.linalg.norm(_ax), 1e-12)
    _outer_track_r = _drive_outer - box.track_clearance_m
    _inner_track_r = box.inner_radius_m + box.inner_track_width_m
    # THE DONUT IS SMALLER THAN THE GAP, by the clearance each side.
    # Sized to exactly span it, it would be in permanent contact with
    # both tracks and there would be nothing for a clutch to do.
    _unit_r = ((_outer_track_r - _inner_track_r) / 2.0
               - box.track_clearance_m)
    _unit_pitch_r = (_outer_track_r + _inner_track_r) / 2.0
    for tag, radius, width, mat, host in (
            ("outer", _outer_track_r, box.wall_thickness_m * 1.6,
             box.rim_material, upper),
            ("inner", _inner_track_r, box.inner_track_width_m,
             box.inner_track_material, lower)):
        for i in range(box.segments):
            a = 2.0 * math.pi * i / box.segments
            pos = _c + (_u * math.cos(a) + _v * math.sin(a)) * radius
            ident = f"{box.identity}.track.{tag}.{i}"
            g.node(ident, tuple(float(x) for x in pos),
                   "load-bearing-structure", material=mat,
                   mass_in_total=False,
                   mass_kg=round(2.0 * math.pi * radius * width * _usable_h
                                 * 7850.0 / box.segments, 1),
                   part_role=f"drive-track-{tag}", boxed_ring=box.identity,
                   track_radius_m=round(radius, 4),
                   hangs_from="top-annulus" if tag == "outer"
                   else "bottom-annulus",
                   clutch_surface=True,
                   half_extent_m=(width / 2.0, _usable_h / 2.0,
                                  max(math.pi * radius / box.segments, 0.01)))
            g.edge(f"{box.identity}.track_hang.{tag}.{i}", ident,
                   host["mean"][i], "rigid-distance", radius=0.026,
                   alloy=box.alloy, palette="rollbar-silver",
                   fastening="bolted",
                   load_path=f"{tag}-track-carried-by-its-own-annulus")
        for i in range(box.segments):
            g.edge(f"{box.identity}.track_hoop.{tag}.{i}",
                   f"{box.identity}.track.{tag}.{i}",
                   f"{box.identity}.track.{tag}.{(i + 1) % box.segments}",
                   "rigid-distance", radius=0.022, alloy=box.alloy,
                   palette="chassis-grey", beam_solvable=True,
                   fastening="bolted",
                   load_path=f"{tag}-track-closing-on-itself")
    # ---- THE STACK, TOP DOWN, INSIDE THE CAVITY ----
    #   pocket ceiling
    #   clearance | service hub | carrier plate | thrust race | clearance
    #   the donut
    #   clearance
    #   pocket floor
    _cav_half = _usable_h / 2.0
    _above = (box.axial_clearance_m + box.carrier_thickness_m
              + box.thrust_race_m)
    _donut_h = _usable_h - _above - 2.0 * box.axial_clearance_m
    _donut_dy = -(_above / 2.0)          # it sits low; the gear is above it
    _carrier_dy = _cav_half - box.axial_clearance_m - box.carrier_thickness_m / 2.0
    _thrust_dy = _carrier_dy - box.carrier_thickness_m / 2.0 - box.thrust_race_m / 2.0

    def _unit_pos(k, dy=0.0):
        a = 2.0 * math.pi * k / box.drive_units
        return (_c + (_u * math.cos(a) + _v * math.sin(a)) * _unit_pitch_r
                + _ax * dy)

    units = []
    for k in range(box.drive_units):
        a = 2.0 * math.pi * k / box.drive_units
        pos = _c + (_u * math.cos(a) + _v * math.sin(a)) * _unit_pitch_r
        kind = box.drive_mix[k % len(box.drive_mix)]
        ident = f"{box.identity}.motor.{k}"
        # ITS OWN MOTION GROUP. A donut turns about ITS axis while the
        # drum turns about the drum's -- two rotations at once, which
        # one group cannot express. Declared per unit so the articulated
        # mesh can pose each of them without recognising anything.
        _was_group = g.motion_group
        g.motion_group = f"drive-unit-{k}"
        g.node(ident, tuple(float(x) for x in _unit_pos(k, _donut_dy)),
               "fluid-motor" if kind != "electric" else "electric-machine",
               material="6061t6", mass_in_total=False,
               mass_kg=round(math.pi * _unit_r ** 2 * _usable_h * 2300.0, 1),
               part_role="drive-unit", boxed_ring=box.identity,
               medium=kind, circular=True, shape="drum",
               # THE INTERLOCK, and it is geometric rather than a rule
               # written on top of the mechanism. A donut floats clear
               # of both tracks. Engage ONE clutch and that track
               # simply pushes the donut across its springs until it
               # stops against nothing -- there is no second contact to
               # react through, so the unit spins and the top does not
               # move. Engage BOTH and the donut is trapped between two
               # surfaces; now it cannot be shoved aside, and torque
               # crosses from the inner track to the outer.
               #
               # WHICH IS WHY INERTIA IS FREE TO PLAY WITH. On one
               # clutch a unit can be spun up, held, reversed or
               # matched to a target speed with the turret completely
               # stationary -- six flywheels being wound at three
               # different media's convenience. The second clutch is
               # what spends it, and spending it all at once is the
               # pop.
               transmits_only_when_both_engaged=True,
               rotor_inertia_kg_m2=round(
                   0.5 * (math.pi * _unit_r ** 2 * _usable_h * 2300.0)
                   * _unit_r ** 2, 4),
               drum_axis=tuple(float(x) for x in _ax),
               drum_radius_m=round(_unit_r, 4),
               drum_length_m=round(_donut_h, 4),
               axis_is_parallel_to_drum=True,
               runs_between=("outer-track", "inner-track"),
               unit_radius_m=round(_unit_r, 4),
               pitch_radius_m=round(_unit_pitch_r, 4),
               usable_height_m=round(_usable_h, 4),
               acts_at_radius_m=box.outer_radius_m,
               half_extent_m=(_unit_r, _donut_h / 2.0, _unit_r))
        g.motion_group = _was_group
        units.append(ident)
    for k in range(box.drive_units):
        g.edge(f"{box.identity}.motor_ring.{k}", units[k],
               units[(k + 1) % box.drive_units], "rigid-distance",
               radius=0.030, alloy="6061t6", palette="chassis-grey",
               fastening="bolted",
               load_path="the-drive-units-carried-in-one-ring-formation")

    for i in range(box.segments):
        # THE CELL WALLS, inner and outer, which are what close it
        for tag, lo, up in (("outer", lower["outer"][i], upper["outer"][i]),
                            ("inner", lower["mean"][i], upper["mean"][i])):
            g.edge(f"{box.identity}.wall.{tag}.{i}", lo, up,
                   "rigid-distance", radius=sec["equivalent_radius_m"],
                   wall_m=box.wall_thickness_m, alloy=box.alloy,
                   palette="chassis-grey", beam_solvable=True,
                   section="box-ring-wall", boxed_ring=box.identity,
                   load_path="the-wall-that-closes-the-cell")
        # and the diagonal that makes each bay a panel rather than a
        # parallelogram -- a box with no shear path is four hinges
        j = (i + 1) % box.segments
        # CROSSED, NOT LEANING. One diagonal per bay all leaning the
        # same way makes the ring CHIRAL: it braces one direction of
        # twist through a strut in tension and the other through the
        # same strut in compression, so the drum is stiffer turning one
        # way than the other. On a turret that is a systematic aiming
        # bias with the sign of the slew. Two diagonals per bay, mirror
        # images, cost one member each and remove it.
        for d, (a_end, b_end) in enumerate(((lower["outer"][i], upper["outer"][j]),
                                            (upper["outer"][i], lower["outer"][j]))):
            g.edge(f"{box.identity}.shear.{i}.{d}", a_end, b_end,
                   "rigid-distance",
                   radius=sec["equivalent_radius_m"] * 0.45,
                   wall_m=box.wall_thickness_m, alloy=box.alloy,
                   palette="chassis-grey", beam_solvable=True,
                   fastening="bolted",
                   load_path="box-bay-shear-panel-crossed-so-it-is-not-chiral")
        # THE ARBITRARY VOLUME IS GONE. Sixteen nodes used to sit
        # here declaring an envelope -- radial depth, usable height,
        # litres of room. That was the right thing to have while the
        # middle was an unknown, and the wrong thing to keep once it
        # was furnished: the cavity's extent is a consequence of the
        # two plates and the pocket, and its contents are the tracks,
        # the donuts, the cradle and the hubs. A node whose only job
        # is to say "there is room here" is a note, not a part, and
        # this one was being counted, drawn, massed and solved.
    _pocket_kg = (math.pi * (_drive_outer ** 2 - box.inner_radius_m ** 2)
                  * box.drive_pocket_m * 7850.0)
    for face in (lower, upper):
        for ident in face["mean"]:
            node = next(x for x in g.nodes if x["identity"] == ident)
            node["mass_kg"] = round(max(node["mass_kg"]
                                        - _pocket_kg / box.segments, 0.1), 2)
            node["pocketed"] = True
    # ---- THE CAST CRADLE THAT HOLDS THEM ALL ----
    # A donut held only by its two traction contacts is held by two
    # clutches, and a clutch is free in rotation by definition -- so
    # the units could swing. Under a 50 kN corner load the drum plates
    # moved 0.078 mm and the motors moved 22.8, which is not a soft
    # structure, it is six heavy objects hanging on pin joints.
    #
    # So they are cradled: one cast aluminium carrier, a saddle at each
    # unit, the saddles tied into a ring and bolted to both plates. Each
    # motor is JOURNALLED in its saddle -- rigid in translation, free
    # about its own axis, which is the one freedom it is supposed to
    # have. That is a planet carrier, and casting it is the same
    # argument as casting the plates: a site that can pour one can pour
    # this, and an intricate shape is cheap in a mould and expensive in
    # a mill.
    _cradle = []
    for k, ident in enumerate(units):
        a = 2.0 * math.pi * k / box.drive_units
        # THE CARRIER SITS OVER THE DONUTS, not inside them. Placed
        # coaxially the journal came out zero-length -- a bearing
        # between a body and itself. A cast carrier is a plate across
        # the top of the units with a boss down into each one, so the
        # saddle is a plate-thickness above the unit's own centre and
        # the journal is the boss.
        pos = _unit_pos(k, _carrier_dy)
        saddle = f"{box.identity}.cradle.{k}"
        g.node(saddle, tuple(float(x) for x in pos),
               "load-bearing-structure", material="6061t6",
               mass_in_total=False,
               mass_kg=round(math.pi * (_unit_r * 1.12) ** 2
                             * box.plate_thickness_m * 2700.0, 1),
               part_role="drive-cradle-saddle", boxed_ring=box.identity,
               cast=True, fastening="bolted",
               half_extent_m=(_unit_r * 1.12,
                              box.carrier_thickness_m / 2.0,
                              _unit_r * 1.12))
        _cradle.append(saddle)
        # ---- THE THRUST RACE, WHICH IS WHAT HOLDS IT UP ----
        # The one freedom left unexplained: a donut on a vertical axis
        # has its own weight along that axis and nothing was carrying
        # it. It must not rest on either plate -- rubbing a pocket floor
        # is friction against the drum itself, which is the one thing
        # this whole arrangement exists to avoid -- so the CRADLE
        # carries it, on the crudest low-friction thing that will do the
        # job: a flat annular ball race between the donut's top face and
        # the carrier above it.
        #
        # It takes weight and axial shock, leaves the donut free to spin,
        # and leaves it free to move RADIALLY on its springs, which the
        # clutches still need.
        thrust = f"{box.identity}.thrust.{k}"
        g.node(thrust, tuple(float(x) for x in _unit_pos(k, _thrust_dy)),
               "chassis-load-node", material="4340qt",
               mass_in_total=False, mass_kg=3.2,
               part_role="drive-unit-thrust-race", boxed_ring=box.identity,
               carries="the-donut's-own-weight",
               race_diameter_m=round(_unit_r * 1.4, 4),
               half_extent_m=(_unit_r * 0.70, box.thrust_race_m / 2.0,
                              _unit_r * 0.70))
        g.edge(f"{box.identity}.thrust_seat.{k}", thrust, saddle,
               "rigid-distance", radius=0.016, rigid=True,
               beam_solvable=False, palette="rollbar-silver",
               load_path="thrust-race-seated-in-the-carrier")
        # THE ANTI-ROTATION DOWEL. A thrust washer left free takes up
        # the donut's rotation and grinds against the carrier instead of
        # the balls -- which is the whole failure mode of a flat thrust
        # race fitted without one. A pin into the boss stops the
        # stationary half being stationary only by luck.
        g.edge(f"{box.identity}.thrust_dowel.{k}", thrust, saddle,
               "rigid-distance", radius=0.010, alloy="4340qt", rigid=True,
               beam_solvable=False, palette="rollbar-silver",
               part_role="thrust-race-anti-rotation-dowel",
               load_path="the-pin-that-stops-the-static-washer-turning")
        g.edge(f"{box.identity}.thrust_run.{k}", ident, thrust,
               "pinion-carrier-bearing", radius=_unit_r * 0.70,
               alloy="4340qt", palette="chassis-grey",
               part_role="drive-unit-thrust-bearing",
               load_path="the-donut-running-on-its-thrust-race")
        # ---- THE JOURNAL FLOATS RADIALLY, ON A STIFF SPRING ----
        # Not rigid. The axle is carried in the cast saddle on springs
        # of the kind a clutch pack uses for torsional shock: hard
        # enough that they are not a suspension, soft enough that the
        # engagement is a ramp rather than an impact.
        #
        # AT REST THE DONUT TOUCHES NOTHING. It sits centred in the gap,
        # clear of both tracks, and the drive is genuinely disconnected
        # -- not slipping, not dragging, disconnected. Apply a clutch
        # and the donut is pushed across its travel against that spring
        # until it bites; the spring is what the clutch is working
        # against, and how far it has been pushed IS how hard the
        # contact is pressed.
        #
        # AND IT CAN GO FULLY TO EITHER EXTREME, which changes the
        # radius the ring of donuts runs at -- so the ratio is not
        # fixed, it is a consequence of which clutches are holding and
        # how hard. Popping one dumps the whole spring's stored force
        # into the track at once, which is the jerk this exists for.
        _travel = box.track_clearance_m * 2.0
        _rate = box.clutch_engagement_n / max(_travel, 1e-6)
        g.edge(f"{box.identity}.journal.{k}", ident, saddle,
               "pinion-carrier-bearing", radius=_unit_r * 0.22,
               alloy="4340qt", palette="rollbar-silver",
               part_role="drive-unit-journal",
               radial_travel_m=round(_travel, 5),
               rests_touching_nothing=True,
               spring_rate_n_per_m=round(_rate, 0),
               engagement_force_n=box.clutch_engagement_n,
               radius_swing_m=round(_travel, 5),
               load_path="the-motor-turning-in-its-own-sprung-cradle")
        # the spring itself, radial, one per unit
        g.edge(f"{box.identity}.journal_spring.{k}", ident, saddle,
               "spring-damper", radius=_unit_r * 0.16,
               alloy="4340qt", palette="actuator-yellow",
               part_role="drive-unit-centring-spring",
               spring_rate_n_per_m=round(_rate, 0),
               free_travel_m=round(_travel, 5),
               centred=True,
               load_path="what-the-clutch-has-to-push-against")
    for k in range(box.drive_units):
        g.edge(f"{box.identity}.cradle_ring.{k}", _cradle[k],
               _cradle[(k + 1) % box.drive_units], "rigid-distance",
               radius=0.075, wall_m=0.014, alloy="6061t6", palette="chassis-grey",
               beam_solvable=True, fastening="bolted",
               load_path="the-cradle-closing-on-itself")
        i = int(round(k * box.segments / box.drive_units)) % box.segments
        for face, tag in ((lower, "lower"), (upper, "upper")):
            g.edge(f"{box.identity}.cradle_seat.{tag}.{k}", _cradle[k],
                   face["mean"][i], "rigid-distance", radius=0.055,
                   alloy="6061t6", palette="rollbar-silver",
                   fastening="bolted",
                   load_path="cradle-bolted-to-the-plates-it-sits-between")

    # ---- EVERY ROTOR IN THE DONUT, AS ITS OWN PART ----
    _rotors = []
    for k, ident in enumerate(units):
        _r_mag_i = _unit_r - box.magnetic_band_m
        _r_hyd_i = _r_mag_i - box.hydraulic_band_m
        _r_pne_i = max(_r_hyd_i * 0.16, 0.03)
        _mag_area = 2.0 * math.pi * _unit_r * _donut_h
        _mag_t = box.airgap_shear_pa * _mag_area * _unit_r
        _hyd_v = math.pi * (_r_mag_i ** 2 - _r_hyd_i ** 2) * _donut_h \
            * box.vane_fill
        _hyd_t = box.oil_pressure_pa * _hyd_v / (2.0 * math.pi)
        _pne_v = math.pi * (_r_hyd_i ** 2 - _r_pne_i ** 2) * _donut_h \
            * box.vane_fill
        _pne_t = box.air_pressure_pa * _pne_v / (2.0 * math.pi)
        for tag, r_o, r_i, mat, rho, torque, extra in (
                ("mag", _unit_r, _r_mag_i, "4340qt", 7600.0, _mag_t,
                 dict(airgap_shear_pa=box.airgap_shear_pa,
                      airgap_area_m2=round(_mag_area, 4),
                      is_the_casing=True)),
                ("hyd", _r_mag_i, _r_hyd_i, "4340qt", 7850.0, _hyd_t,
                 dict(displacement_cc_rev=round(_hyd_v * 1e6, 0),
                      supply_pressure_pa=box.oil_pressure_pa)),
                ("pne", _r_hyd_i, _r_pne_i, "6061t6", 2700.0, _pne_t,
                 dict(displacement_cc_rev=round(_pne_v * 1e6, 0),
                      supply_pressure_pa=box.air_pressure_pa))):
            rid = f"{box.identity}.rotor.{tag}.{k}"
            mass = math.pi * (r_o ** 2 - r_i ** 2) * _donut_h * rho * 0.55
            g.node(rid, tuple(float(x) for x in _unit_pos(k, _donut_dy)),
                   "electric-machine" if tag == "mag" else "fluid-motor",
                   material=mat, mass_in_total=False, mass_kg=round(mass, 1),
                   structural_participation=False,
                   solver_condensed_into=ident, solver_condensed_mass=True,
                   part_role=f"donut-rotor-{tag}", boxed_ring=box.identity,
                   on_unit=ident, outer_radius_m=round(r_o, 4),
                   inner_radius_m=round(r_i, 4),
                   torque_nm=round(torque, 1),
                   # THERMODYNAMICS. Every one of these turns some of
                   # what it is given into heat, and the heat has one
                   # way out: the oil the hub already brings. Declared
                   # as a loss fraction and a path, so a rotor that
                   # cannot be cooled says so rather than running for
                   # ever at full output on paper.
                   loss_fraction=0.09 if tag == "mag" else 0.14,
                   cooled_by="hydraulic", coolant_path=f"{box.identity}.hub.{k}",
                   half_extent_m=(r_o, _donut_h / 2.0, r_o), **extra)
            g.edge(f"{box.identity}.rotor_shaft.{tag}.{k}", rid, ident,
                   "rigid-offset", radius=0.020, rigid=True,
                   beam_solvable=False, palette="rollbar-silver",
                   load_path="rotor-on-the-donut's-own-shaft")
            # PILOTED ON THE ONE OUTSIDE IT. Three rotors on one shaft
            # are not three free bodies: each is bored to register on
            # the next, which is how a stack this concentric is actually
            # assembled and what stops each one being located by its
            # shaft joint alone.
            if _rotors and _rotors[-1].endswith(f".{k}"):
                g.edge(f"{box.identity}.rotor_pilot.{tag}.{k}", rid,
                       _rotors[-1], "rigid-offset", radius=0.016,
                       rigid=True, beam_solvable=False,
                       palette="rollbar-silver",
                       load_path="rotor-registered-on-the-one-outboard-of-it")
            _rotors.append(rid)

    # ---- THE SERVICE HUB: POWER, AIR AND OIL ACROSS A TURNING JOINT ----
    # A donut is a motor and it needs feeding, but it is also spinning
    # inside a ring that is itself spinning. Routing that as plumbing
    # would mean a worm of hose chasing each unit round -- which is how
    # an automatic transmission gets oil to a turning clutch pack, and
    # it works because the oil goes through the SHAFT, not around it.
    #
    # THE SAME TRICK, AND THE FAMILIAR ONE: a wheel hub feeds a tyre
    # across a turning joint without a hose going round with the wheel.
    # The carrier boss is already on each unit's axis, so it becomes the
    # hub: a rotary union with concentric annular grooves for oil and
    # air, and a slip ring stack for power, all on the axis where the
    # relative speed is lowest and the sealing diameter smallest.
    #
    # THE INTERFACE IS FRICTIONLESS IN THE SENSE THAT MATTERS: the
    # seals run on a small diameter at the unit's own speed rather than
    # on the drum's three-metre rim, so the drag is a rounding error
    # against what the drive makes. Putting the union anywhere else
    # would have been the whole argument against doing it at all.
    _hub_ring = {"oil": [], "air": [], "power": []}
    for k, ident in enumerate(units):
        saddle = _cradle[k]
        hub = f"{box.identity}.hub.{k}"
        a = 2.0 * math.pi * k / box.drive_units
        # ---- WHAT THE HUB HAS TO PASS ----
        # The union is not a fitting, it is a set of passages, and the
        # passages are sized by flow, not by what looks proportionate.
        #
        #   HYDRAULIC. Power is pressure times flow, so a 45 kW motor at
        #   210 bar wants 129 L/min. A pressure line runs at about 6 m/s
        #   before the losses stop being worth it, and a return line at
        #   half that because it has no pressure to spend -- so the
        #   return passage is the BIGGER of the two, which is the thing
        #   people get backwards.
        #
        #   PNEUMATIC. Air is compressible and the motor expands it, so
        #   the flow that matters at the union is at LINE conditions,
        #   not free air. At 10 bar that is 2.7 m3/min, and air lines
        #   run at 20 m/s. It still comes out the largest passage of the
        #   three, which is why air is the one that decides the boss.
        #
        #   ELECTRICAL is a slip ring and a solved problem; it needs
        #   contact area, not bore.
        #
        # ANNULAR, NOT DRILLED. A 54 mm round hole will not fit three
        # times through a boss this size, but the union's grooves are
        # already annular -- area is mean circumference times radial
        # width, so the same flow goes through a few millimetres of slot
        # all the way round.
        _oil_q = box.unit_electrical_w / max(box.oil_pressure_pa, 1.0)
        _air_q = box.unit_electrical_w / max(box.air_pressure_pa, 1.0)
        _seal_d = _unit_r * 0.30
        def _slot(flow_m3_s, velocity):
            area = flow_m3_s / velocity
            return area, area / (math.pi * _seal_d)
        # THE FLUID DECIDES THE VELOCITY, not a rule of thumb typed in.
        # `fluids.of_circuit` already knows what runs in each circuit and
        # what it weighs, so the allowable line speed comes from the
        # fluid's own density -- the dynamic head that sets a line's
        # losses is rho*v^2/2, so a light fluid is allowed to move fast
        # and a heavy one is not, and both fall out of one number.
        from fluids import of_circuit
        _oil = of_circuit("hydraulic")
        _air = of_circuit("pneumatic")
        _HEAD_PA = 16_000.0           # the dynamic head a line is run at
        _v_oil = math.sqrt(2.0 * _HEAD_PA / max(_oil.density_kg_m3, 1.0))
        _v_air = math.sqrt(2.0 * _HEAD_PA
                           / max(_air.density_kg_m3
                                 * box.air_pressure_pa / 1.0e5, 1.0))
        _oil_a, _oil_w = _slot(_oil_q, _v_oil)
        _ret_a, _ret_w = _slot(_oil_q, _v_oil * 0.5)
        _air_a, _air_w = _slot(_air_q, _v_air)
        g.node(hub, tuple(float(x) for x in _unit_pos(k, _carrier_dy)),
               "manifold",
               structural_participation=False,
               solver_condensed_into=saddle, solver_condensed_mass=True,
               ports=[
                   {"name": "oil-supply", "circuit": "hydraulic",
                    "fluid": _oil.key, "slot_width_m": round(_oil_w, 5),
                    "area_m2": round(_oil_a, 6),
                    "velocity_m_s": round(_v_oil, 2),
                    "pressure_pa": box.oil_pressure_pa,
                    "flow_l_min": round(_oil_q * 60_000.0, 1)},
                   {"name": "oil-return", "circuit": "hydraulic-return",
                    "fluid": _oil.key, "slot_width_m": round(_ret_w, 5),
                    "area_m2": round(_ret_a, 6),
                    "velocity_m_s": round(_v_oil * 0.5, 2),
                    "pressure_pa": 3.0e5,
                    "flow_l_min": round(_oil_q * 60_000.0, 1)},
                   {"name": "air-supply", "circuit": "pneumatic",
                    "fluid": _air.key, "slot_width_m": round(_air_w, 5),
                    "area_m2": round(_air_a, 6),
                    "velocity_m_s": round(_v_air, 2),
                    "pressure_pa": box.air_pressure_pa,
                    "flow_l_min": round(_air_q * 60_000.0, 1)},
               ],
               coolant_capacity_w=round(
                   _oil_q * _oil.density_kg_m3
                   * _oil.specific_heat_j_per_kg_k * 30.0, 0),
               oil_flow_l_min=round(_oil_q * 60_000.0, 1),
               air_flow_l_min=round(_air_q * 60_000.0, 1),
               oil_supply_slot_m=round(_oil_w, 5),
               oil_return_slot_m=round(_ret_w, 5),
               air_slot_m=round(_air_w, 5),
               slot_stack_m=round(_oil_w + _ret_w + _air_w, 5),
               material="4340qt", mass_in_total=False, mass_kg=11.0,
               part_role="drive-unit-service-hub", boxed_ring=box.identity,
               rotary_union=True, slip_ring=True,
               on_axis_of=ident,
               oil_pressure_pa=box.oil_pressure_pa,
               air_pressure_pa=box.air_pressure_pa,
               electrical_w=box.unit_electrical_w,
               seal_diameter_m=round(_unit_r * 0.30, 4),
               note="grooves for oil and air, a slip-ring stack for "
                    "power, on the axis because that is where the seal "
                    "is smallest and slowest",
               half_extent_m=(_unit_r * 0.30,
                              box.carrier_thickness_m * 0.45,
                              _unit_r * 0.30))
        # INSIDE ONE CASTING, so rigid: the union is machined into
        # the boss, it is not bolted to it through a member.
        # ZERO LENGTH ON PURPOSE: the union is bored into the carrier
        # boss, so the two are the same metal at the same place. That is
        # what `rigid-offset` is declared for -- two surfaces of one
        # piece described twice.
        g.edge(f"{box.identity}.hub_mount.{k}", hub, saddle,
               "rigid-offset", radius=0.022, alloy="4340qt", rigid=True,
               beam_solvable=False, palette="rollbar-silver",
               load_path="the-union-carried-by-the-boss-it-seals-on")
        # FED PER ROTOR, NOT PER DONUT. The three rotors on one shaft
        # want three different things, and running one line to the shell
        # and branching it inside would put the branch in the rotating
        # part where nobody can reach it. The union has the grooves
        # already; each rotor takes its own.
        for service, constraint, circuit, rotor in (
                ("oil", "oil-line", "hydraulic", "hyd"),
                ("air", "air-line", "pneumatic", "pne"),
                ("power", "insulated-copper-wire", "drive-power", "mag")):
            g.edge(f"{box.identity}.feed.{service}.{k}", hub,
                   f"{box.identity}.rotor.{rotor}.{k}",
                   constraint, radius=0.014,
                   circuit_identity=circuit, palette="rollbar-silver",
                   crosses_rotating_joint=True, feeds_rotor=rotor,
                   load_path=f"{service}-through-the-hub-into-the-{rotor}-rotor")
            # and the heat comes back the same way it went in
            if service == "oil":
                for _r in ("mag", "hyd", "pne"):
                    g.edge(f"{box.identity}.scavenge.{_r}.{k}",
                           f"{box.identity}.rotor.{_r}.{k}", hub,
                           "oil-line", radius=0.018,
                           circuit_identity="hydraulic-return",
                           palette="rollbar-silver",
                           carries="rotor-losses-as-heat",
                           load_path=f"{_r}-rotor-losses-back-out-through-the-hub")
            _hub_ring[service].append(hub)
        # ---- THE UNION'S GROOVES ARE INTERNAL CHANNELS ----
        # Not three pipes wrapped round a boss: three annular passages
        # cored into it, each opening onto the rotor it feeds and onto
        # the stationary side. A groove all the way round is what lets
        # the joint TURN without the passage caring where the port has
        # got to -- which is the entire trick of a rotary union, and it
        # is a shape rather than a fitting.
        from channels import InternalChannel, emit_channel
        for _svc, _circ, _rotor, _bore, _width, _press in (
                ("oil", "hydraulic", "hyd", _oil_a / max(_oil_w, 1e-9),
                 _oil_w, box.oil_pressure_pa),
                ("return", "hydraulic-return", "hyd",
                 _ret_a / max(_ret_w, 1e-9), _ret_w, 3.0e5),
                ("air", "pneumatic", "pne", _air_a / max(_air_w, 1e-9),
                 _air_w, box.air_pressure_pa)):
            emit_channel(g, InternalChannel(
                identity=f"{box.identity}.groove.{_svc}.{k}",
                body=hub, ports=(hub, f"{box.identity}.rotor.{_rotor}.{k}"),
                bore_m=_bore, groove_width_m=_width, circuit=_circ,
                route="annular", pressure_pa=_press,
                note=f"{_svc} groove: a ring, so the union can turn"),
                material="4340qt")
    # AND ONE RING MAIN PER SERVICE, so a unit can be isolated without
    # taking the other five with it.
    for service, constraint, circuit in (
            ("oil", "oil-line", "hydraulic"),
            ("air", "air-line", "pneumatic"),
            ("power", "insulated-copper-wire", "drive-power")):
        for k in range(box.drive_units):
            g.edge(f"{box.identity}.main.{service}.{k}",
                   _hub_ring[service][k],
                   _hub_ring[service][(k + 1) % box.drive_units],
                   constraint, radius=0.016, circuit_identity=circuit,
                   palette="rollbar-silver", isolatable=True,
                   load_path=f"{service}-ring-main-round-the-carrier")

    # EACH DONUT TOUCHES BOTH TRACKS, and each contact is a clutch.
    for k, ident in enumerate(units):
        i = int(round(k * box.segments / box.drive_units)) % box.segments
        # OUTER IS DRY, INNER IS TRACTION FLUID, and the split is the
        # point. The outer face carries the torque -- mu 0.35 passes
        # 84% of what the rotors make against the spring that is
        # already there, where traction fluid would want 933 kN of
        # squeeze to do the same -- and it takes the wear somewhere it
        # can be relined. The inner face stays wet and smooth and does
        # the fine trimming, which is what a continuously variable bite
        # actually needs. min(outer, inner) still governs, so the
        # interlock keeps its meaning.
        for tag, mu, film, lining in (("outer", 0.35, "dry", "sintered-bronze"),
                                      ("inner", 0.08, "traction-fluid",
                                       "hardened-steel")):
            g.edge(f"{box.identity}.contact.{tag}.{k}", ident,
                   f"{box.identity}.track.{tag}.{i}",
                   "actuated-damped-clutch-gimbal-base",
                   friction_coefficient=mu, film=film, lining=lining,
                   passes_tangential_n=round(box.clutch_engagement_n * mu, 0),
                   radius=0.030, alloy="4340qt", palette="chassis-grey",
                   part_role="traction-contact", contact=tag,
                   clutch=True, engaged_fraction=0.0,
                   rest_gap_m=box.track_clearance_m,
                   presses_with_n=box.clutch_engagement_n,
                   traction_radius_m=round(_unit_r, 4),
                   load_path=f"donut-running-on-the-{tag}-track")
        # ---- DIRECT-DRIVE LOCK-UP, ON THE INNER TRACK ONLY ----
        # The inner face was the throttle: traction fluid at mu 0.08
        # passes 14.4 kN where the outer's dry 0.35 passes 63, and two
        # faces in series mean the drum only ever saw the smaller. A
        # lock-up ends that -- clamped hard, it is direct drive -- while
        # the OUTER stays metering, which is what keeps the modulation.
        #
        # NOT TEETH. A dog cannot slip, so with one home a shock coming
        # from the DRUM side goes through it into the rotors and the
        # journal with nothing to give. A clamped face holds everything
        # it is asked for in service and lets go above its rated torque,
        # which is the overload path the teeth would have removed.
        #
        # And it needs no sync window: a face that can slip matches its
        # own speeds on the way in, so it can be applied whenever it is
        # wanted rather than only when the speeds already agree.
        _lock_n = box.clutch_engagement_n * box.lockup_clamp_ratio
        g.edge(f"{box.identity}.lockup.{k}", ident,
               f"{box.identity}.track.inner.{i}", "direct-drive-lockup",
               radius=0.048, alloy="4340qt", palette="actuator-yellow",
               part_role="inner-lockup", on_contact="inner",
               engaged=False, clamp_force_n=round(_lock_n, 0),
               friction_coefficient=0.35, plates=box.lockup_plates,
               slip_torque_nm=round(_lock_n * 0.35 * box.lockup_plates
                                    * _unit_r, 0),
               replaces_when_engaged="the-inner-traction-face",
               load_path="clamped-to-direct-drive-on-the-inner-track")
    return {"lower": lower, "upper": upper, "motors": units,
            "cradle": _cradle,
            "closed_j_m4": box.closed_torsion_m4,
            "open_j_m4": box.open_torsion_m4}


# =====================================================================
#  A RACE IS A PATH WITH BEARINGS CAPTIVE IN IT
# =====================================================================
@dataclass
class BearingRace:
    """One captive bearing race: straight, or an arc of any sweep.

    THE SHAPE IS A PARAMETER, NOT A PART. A slew ring, a recoil journal
    and a linear rail are the same object -- a path, a section, a row of
    rolling elements held captive against a counterface -- differing
    only in whether the path curves and by how much. Written as three
    types they drift apart: the turret already had a flat annular track
    with pads AND a square journal with carriages AND a rail with
    slides, three spellings of one idea, each with its own bugs and its
    own idea of what a preload was.

    `radius_m` None makes it straight between `start` and `end`.
    Otherwise it is an arc of `sweep_deg` about `centre` in the plane
    normal to `axis` -- a full 360 for a slew ring, 90 for a quadrant
    elevation arc, 20 for a limited-travel trunnion race.

    CAPTIVE MEANS TWO-SIDED. A race that only pushes is a track, and a
    track under an overturning load lifts off over the half of itself
    that goes into tension. `preload_n` is what holds the element
    against both faces, so the race carries tension AND compression and
    the structure stops being a hinge. That is the whole reason the word
    captive is in the name."""
    identity: str
    stations: int = 16
    #: straight
    start: tuple | None = None
    end: tuple | None = None
    #: or curved
    centre: tuple | None = None
    radius_m: float | None = None
    sweep_deg: float = 360.0
    start_deg: float = 0.0
    axis: tuple = (0.0, 1.0, 0.0)
    #: the rolling element
    element: str = "slew-thrust-roller"
    element_diameter_m: float = 0.090
    element_length_m: float = 0.110
    preload_n: float = 0.0
    material: str = "hardened-steel"

    @property
    def closed(self) -> bool:
        return self.radius_m is not None and abs(self.sweep_deg) >= 359.9

    @property
    def path_length_m(self) -> float:
        if self.radius_m is None:
            return float(np.linalg.norm(np.asarray(self.end, float)
                                        - np.asarray(self.start, float)))
        return abs(math.radians(self.sweep_deg)) * self.radius_m

    def station_point(self, i: int) -> tuple:
        """Where station i sits. One expression for both shapes, so a
        straight race and an arc cannot disagree about their own ends."""
        n = max(self.stations - (0 if self.closed else 1), 1)
        f = i / n
        if self.radius_m is None:
            a = np.asarray(self.start, float)
            b = np.asarray(self.end, float)
            p = a + (b - a) * f
        else:
            ang = math.radians(self.start_deg + self.sweep_deg * f)
            c = np.asarray(self.centre, float)
            u, v = ring_frame(self.axis)
            p = c + (u * math.cos(ang) + v * math.sin(ang)) * self.radius_m
        return tuple(float(x) for x in p)

    @property
    def pitch_m(self) -> float:
        return self.path_length_m / max(self.stations, 1)


def emit_race(g, race: BearingRace, counterface, *,
              motion_group: str | None = None,
              assembly: str | None = None) -> dict:
    """Emit the race and make every element captive against a counterface.

    `counterface` is either one node identity or a callable taking the
    station index -- because a slew race runs against a whole ring of
    nodes and a linear rail runs against one block, and both are
    ordinary."""
    if motion_group is not None:
        g.motion_group = motion_group
    if assembly is not None:
        g.assembly = assembly
    stations = []
    for i in range(race.stations):
        ident = f"{race.identity}.element.{i}"
        g.node(ident, race.station_point(i), "chassis-load-node",
               material=race.material, mass_in_total=False,
               mass_kg=round(math.pi * (race.element_diameter_m / 2.0) ** 2
                             * race.element_length_m * 7850.0, 2),
               part_role="captive-bearing-element", race=race.identity,
               element_diameter_m=race.element_diameter_m,
               element_length_m=race.element_length_m,
               preload_n=race.preload_n,
               race_shape="arc" if race.radius_m is not None else "straight",
               race_radius_m=race.radius_m, race_sweep_deg=race.sweep_deg,
               half_extent_m=(race.element_diameter_m / 2.0,
                              race.element_diameter_m / 2.0,
                              race.element_length_m / 2.0))
        stations.append(ident)
        other = counterface(i) if callable(counterface) else counterface
        g.edge(f"{race.identity}.captive.{i}", ident, other, race.element,
               radius=race.element_diameter_m / 2.0, palette="chassis-grey",
               part_role="captive-bearing", race=race.identity,
               preload_n=race.preload_n,
               load_path="element-held-against-both-faces-of-its-race")
    # THE CAGE. Rolling elements are held at pitch by one, and without
    # it they bunch at the loaded arc and leave the rest of the race
    # empty -- which is a real failure mode, not a modelling detail. It
    # is also why an element has three connections and not two: two
    # races and its neighbours.
    if race.stations > 1:
        last = race.stations if race.closed else race.stations - 1
        for i in range(last):
            g.edge(f"{race.identity}.cage.{i}", stations[i],
                   stations[(i + 1) % race.stations], "rigid-distance",
                   radius=0.010, palette="rollbar-silver",
                   part_role="bearing-cage", race=race.identity,
                   load_path="cage-holding-the-elements-at-pitch")
    return {"elements": stations, "pitch_m": race.pitch_m,
            "length_m": race.path_length_m}


def traction_state(engaged: dict, *, inner_locked: bool = False) -> dict:
    """What a drive unit is doing, given which of its contacts are in.

    `engaged` maps "outer"/"inner" to an engagement fraction 0..1. The
    result says whether the top can move at all, which is the one
    question the interlock exists to answer.

    THE AND IS NOT A SAFETY RULE. It is the geometry: a donut with one
    free side has nothing to push against, so a single clutch
    accelerates the unit's own rotor and nothing else. Writing it as a
    rule would let someone "fix" it later; writing it as the shape
    means it cannot be fixed away."""
    outer = float(engaged.get("outer", 0.0))
    # A LOCKED FACE IS FULL ENGAGEMENT. Clamped to direct drive the
    # inner side stops metering anything, and the OUTER becomes the
    # only thing deciding how much torque crosses -- which is what
    # makes the pair useful rather than redundant.
    inner = 1.0 if inner_locked else float(engaged.get("inner", 0.0))
    both = min(outer, inner)
    return {
        "outer": outer, "inner": inner,
        "transmits": both > 0.0,
        "transmitted_fraction": both,
        "top_can_turn": both > 0.0,
        "rotor_free": both <= 0.0 and max(outer, inner) > 0.0,
        "disconnected": max(outer, inner) <= 0.0,
        "inner_locked": bool(inner_locked),
        "metered_by": "the outer friction face" if inner_locked else
                      "whichever face is weaker",
        "note": ("both in: torque crosses" if both > 0.0 else
                 "one in: the donut is shoved across its springs and "
                 "spins; the top does not move" if max(outer, inner) > 0.0
                 else "floating clear of both tracks"),
    }
