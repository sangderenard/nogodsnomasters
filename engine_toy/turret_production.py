"""The hydraulic pop-up turret, authored in the PRODUCTION graph's own
vocabulary so the production physics can run it.

The rule this module exists to obey: the game's physics owns the
motion. Nothing here prescribes where a part goes. Every part is a node
that carries a wrench, every connection is an authored member with a
rest length, a section, a damage law and where appropriate a bushing,
and the turret rises because a hydraulic member's COMMANDED REST LENGTH
grows and the structure follows it. That is the only arrangement in
which the pressure and flow this project already computes actually
communicate power into the world.

WHAT THE PRODUCTION GRAPH ALREADY HAS, and is reused here rather than
reinvented (abstract_ui_vehicles._vehicle_mechanical_graph):

  gimbal-yaw-bearing            traverse
  gimbal-pitch-bearing          elevation
  point-impulse-wrench-coupling recoil, as an r-cross-impulse load path
  linear-hydraulic-actuator     a member whose rest length is commanded,
                                with series relief so it is backdrivable
                                rather than a rigid position source

WHAT WE HAVE TO SUPPLY, because the production turret is a small
body-pin mount with no hoist and no hydraulics:

  the WELL, as a real triangulated frame rather than a box -- eight
  corner nodes, perimeter rails and both diagonals per face, because a
  four-bar rectangle is a mechanism and only a triangulated one is a
  structure;

  the CARRIAGE that rises, likewise triangulated, and the GUIDE STRUTS
  that stop it swinging on the end of the ram -- a single central ram
  with nothing else holding it is a pendulum, and the guides are what
  make the assembly rise straight;

  the HOIST itself as hydraulic members;

  and the RING, which carries the whole rotating mass down into the
  carriage.

Every member here is authored with the same fields the production
helper authors: rest length derived from the real node positions, a
section from the declared radius, an elastic-plastic-with-shear-fracture
damage law scaled off that section, and six-axis Kelvin-Voigt bushings
on the joints that really have them.
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field

from joints import MEMBER_CONSTRAINTS, member_constraint

# Structural steel, and the same derivation the production helper uses:
# yield at 250 MPa with a utilisation factor on each mode.
STEEL_YIELD_PA = 250_000_000.0
AXIAL_UTILISATION = 0.72
SHEAR_UTILISATION = 0.42
# a routed line carries fluid, not load: no damage law, no bushing
#: WHAT A MEMBER TRANSMITS lives in joints.py now, as an object per
#: constraint in the same RIGID/FREE/COMPLIANT vocabulary the junction
#: types already used. These five sets were five separate string
#: identities for one thing, and nothing could check any of them: a
#: misspelling produced a member with the wrong steel, no bushing and no
#: damage law, in silence. They are kept only as views onto the registry
#: so that a stale reader still gets the truth rather than a stale copy.
ZERO_LENGTH_CONSTRAINTS = {k for k, m in MEMBER_CONSTRAINTS.items()
                           if m.zero_length}
ROUTED_LINE_CONSTRAINTS = {k for k, m in MEMBER_CONSTRAINTS.items()
                           if m.routed}
BUSHED_CONSTRAINTS = {k for k, m in MEMBER_CONSTRAINTS.items() if m.bushed}


# What a node of each kind is, physically, so it has something to draw,
# pick and stop a bullet with.
_DEFAULT_BODY_M = {
    "structural-body-pin-frame-foot": (0.12, 0.10, 0.12),
    "chassis-load-node": (0.09, 0.09, 0.09),
    "load-bearing-structure": (0.16, 0.14, 0.16),
    "slew-ring-race-pad": (0.11, 0.05, 0.11),
    "yaw-clutch-bearing": (0.26, 0.07, 0.26),
    "pitch-bearing": (0.10, 0.09, 0.16),
    "recoiling-weapon-mass": (0.06, 0.06, 0.62),
    "manifold": (0.07, 0.07, 0.10),
    "high-pressure-canister": (0.13, 0.40, 0.13),
}
# How much of a part's envelope is actually metal. A plate structure
# is mostly air; a shaft is solid. Declared per kind rather than
# guessed, because it is the difference between a turret that weighs
# two tonnes and one that weighs twenty.
_HOLLOW_FRACTION = {
    "structure": 0.93, "armour-shell": 0.91, "chassis-load-node": 0.75,
    "structural-body-pin-frame-foot": 0.55, "load-bearing-structure": 0.62,
    "slew-ring-race-pad": 0.30, "yaw-clutch-bearing": 0.55,
    "pitch-bearing": 0.45, "recoiling-weapon-mass": 0.18,
    "high-pressure-canister": 0.88, "fluid-reservoir": 0.90,
    "manifold": 0.35, "spool-valve": 0.40, "electric-motor": 0.25,
    "hydraulic-cylinder": 0.55, "gun-barrel": 0.60,
}


def _derived_mass_kg(record: dict) -> float:
    """Volume from whatever shape the part declared, times the density
    of what it says it is made of, times the solid fraction."""
    from ballistics import mass_of
    shape = record.get("shape")
    if shape == "ring":
        outer = float(record.get("ring_outer_radius_m", 0.5))
        inner = float(record.get("ring_inner_radius_m", 0.3))
        volume = math.pi * (outer ** 2 - inner ** 2) * float(record.get("ring_thickness_m", 0.06))
    elif shape == "drum":
        radius = float(record.get("drum_radius_m", 0.05))
        volume = math.pi * radius ** 2 * float(record.get("drum_length_m", 0.10))
    else:
        hx, hy, hz = (float(v) for v in record["body_half_extent_m"])
        volume = 8.0 * hx * hy * hz
    hollow = _HOLLOW_FRACTION.get(str(record.get("kind", "")), 0.70)
    return mass_of(record.get("material", "steel-plate"), volume, hollow_fraction=hollow)


def _bushing(frame_mount: bool) -> dict:
    """A six-axis Kelvin-Voigt junction, derived from its own geometry
    exactly as the production helper derives it -- polyurethane, a real
    annulus, and damping from the critical damping of that stiffness
    against the junction's own effective mass."""
    outer = 0.024 if frame_mount else 0.018
    inner = 0.010 if frame_mount else 0.008
    length = 0.038 if frame_mount else 0.032
    youngs = 65.0e6 if frame_mount else 48.0e6
    shear = 22.0e6 if frame_mount else 16.5e6
    area = math.pi * (outer ** 2 - inner ** 2)
    polar = math.pi * (outer ** 4 - inner ** 4) / 2
    k_lin = youngs * area / length
    k_ang = shear * polar / length
    zeta = 0.22 if frame_mount else 0.18
    eff_mass = 18.0 if frame_mount else 7.5
    eff_inertia = 0.18 if frame_mount else 0.055
    preload = 0.00035 if frame_mount else 0.00020
    return {
        "model": "six-axis-kelvin-voigt-junction",
        "parameter_pack": "performance-polyurethane-calculated-static-v2",
        "compile_policy": "parameterized-source-resolved-to-static-kernel-constants",
        "frame_mount": frame_mount,
        "material": {"shore_hardness_a": 92, "youngs_modulus_pa": youngs,
                     "shear_modulus_pa": shear, "loss_factor": 2 * zeta},
        "geometry_m": {"outer_radius": outer, "inner_radius": inner, "length": length},
        "linear_stiffness_n_per_m": k_lin,
        "linear_damping_n_s_per_m": 2 * zeta * math.sqrt(k_lin * eff_mass),
        "angular_stiffness_nm_per_rad": k_ang,
        "angular_damping_nm_s_per_rad": 2 * zeta * math.sqrt(k_ang * eff_inertia),
        "preload_compression_m": preload,
        "preload_force_n": k_lin * preload,
        "yield_displacement_m": 0.0045 if frame_mount else 0.0060,
        "yield_force_n": k_lin * (0.0045 if frame_mount else 0.0060),
        "fracture_displacement_m": 0.014 if frame_mount else 0.018,
        "fracture_force_n": k_lin * (0.014 if frame_mount else 0.018),
        "static_friction_torque_nm": 1.8,
        "dissipation": "sum-c-linear-v-relative-squared-and-c-angular-omega-relative-squared",
    }


@dataclass
class ProductionGraph:
    """A node/edge document in the production vocabulary."""
    identity: str
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)

    #: the mechanism whose motion the next bodies authored will ride.
    #: "frame" is ground; every other value names a stage of the mount.
    motion_group: str = "frame"
    #: Volumes that must stay empty: an ejection column, a retraction
    #: column. Declared by whichever part owns the requirement.
    clear_volumes: list = field(default_factory=list)
    # WHICH SUBASSEMBLY THIS IS PART OF. `motion_group` says what MOVES
    # together; `assembly` says what is BOLTED together, and they are
    # not the same question -- the breech, the barrel and the recoil
    # gear all move as one and are three different assemblies to build,
    # buy, colour and argue about. Declared the same way, around a
    # block, so no part of this file ever has to be identified by the
    # shape of its name.
    assembly: str = "unassigned"

    def node(self, identity: str, position, kind: str, *,
             half_extent_m=None, **attributes) -> None:
        """A node carries a WRENCH. That is the whole of its contract:
        the solver reduces forces and moments through the edges onto
        these, and nothing here says where it will end up.

        It also needs a BODY. A node with no declared extent draws
        nothing, so it cannot be seen, picked or shot -- the first
        version of this module left them all bodiless and the whole
        turret came to sixteen triangles, which is a structure you
        cannot test because there is nothing there to hit."""
        body = half_extent_m
        if body is None:
            body = _DEFAULT_BODY_M.get(kind, (0.06, 0.06, 0.06))
        record = {
            "identity": identity, "kind": kind,
            "reference_position": [float(v) for v in position],
            "body_half_extent_m": [float(v) for v in body],
            "wrench": {"force": [0.0, 0.0, 0.0], "moment": [0.0, 0.0, 0.0]},
            # WHICH MECHANISM CARRIES THIS BODY. Declared, never guessed
            # -- a mesh that articulates has to know what moves when the
            # mount traverses and what stays behind, and working that
            # out by looking at identities is exactly the kind of
            # name-sniffing this file exists to avoid. Builders set
            # `g.motion_group` around a block and every body authored in
            # it says so for itself.
            "motion_group": self.motion_group,
            "assembly": self.assembly,
            # Structural solvers resolve ``auto`` from incident physical
            # members.  A baked engine/machine may instead mark internal
            # render/state parts false or name their one gestalt body with
            # ``solver_condensed_into``.
            "structural_participation": attributes.pop(
                "structural_participation", "auto"),
            # declared visible: the engine view's keyword allowlist was
            # written for engine parts and knows nothing about a turret
            "in_view": True,
            **attributes,
        }
        # MASS FROM GEOMETRY AND MATERIAL, not from a number typed in.
        # A part that declares what it is made of and how big it is has
        # already said what it weighs; asking for the mass separately is
        # asking the same question twice and getting two answers.
        if "mass_kg" not in record:
            record["mass_kg"] = round(_derived_mass_kg(record), 2)
            record["mass_is_derived"] = True
        # ONE IDENTITY, ONE BODY. A repeated identity is not a merge: it
        # is two bodies at the same place with the mass counted twice,
        # and every lookup downstream silently picks whichever came
        # first. This turret shipped with a doubled yaw ring exactly
        # that way, so the duplicate is an error, not a warning.
        if any(n["identity"] == identity for n in self.nodes):
            raise ValueError(
                f"{self.identity}: node {identity!r} authored twice -- "
                f"a node identity names one body, not a place two "
                f"builders may each describe")
        self.nodes.append(record)

    def edge(self, identity: str, a: str, b: str, constraint: str, *,
             radius: float = 0.012, palette: str = "rollbar-silver",
             frame_mount: bool = False, **attributes) -> None:
        pa = next(n["reference_position"] for n in self.nodes if n["identity"] == a)
        pb = next(n["reference_position"] for n in self.nodes if n["identity"] == b)
        rest_length = math.sqrt(sum((pa[i] - pb[i]) ** 2 for i in range(3)))
        # WHAT THIS MEMBER IS, asked once, of the one place that knows.
        # An undeclared spelling raises here -- at the line that authored
        # it -- instead of silently becoming a welded 4130 bar eight
        # modules downstream.
        mc = member_constraint(constraint)
        routed = mc.routed
        spring_like = mc.spring_like
        # A MEMBER IS A TUBE OF A DECLARED ALLOY, not a solid bar of one
        # hardcoded steel. `radius` is its OUTSIDE radius; the wall and
        # the material come from `milspec`, and every property the
        # game's solver reads is declared rather than left to that
        # solver's own car defaults -- which is what every member
        # authored before this was silently being analysed as.
        from milspec import TubeSection, damage_record, MATERIAL_BY_KEY
        section_properties = attributes.pop("section_properties", None)
        alloy = attributes.pop("alloy", None) or mc.alloy
        wall = attributes.pop("wall_m", None) or max(
            0.0025, min(radius * 0.42, 0.0045 + radius * 0.16))
        # A MEMBER MAY SAY WHAT SECTION IT IS. A machine frame is square
        # hollow section or angle, not round tube, and a panel is sheet;
        # a declared section is used as given and the round tube below
        # is only what a member that declared nothing gets. The drawn
        # radius follows the section so the picture is the paper.
        declared = attributes.pop("section", None)
        seam = attributes.pop("seam", None)
        if declared is not None:
            section = declared
            radius = float(getattr(declared, "outer_diameter_m", radius * 2.0)) / 2.0
        else:
          section = TubeSection(outer_diameter_m=radius * 2.0, wall_m=wall,
                              material=alloy,
                              designation=f"{radius * 2000:.0f} mm OD x "
                                          f"{wall * 1000:.1f} mm {alloy}")
        area = section.area_m2
        # a slide, a ram or a guide has its length COMMANDED; the rest
        # of the structure does not, and the difference is the whole
        # reason 320 mm of recoil used to report as 178% strain
        travels = mc.travels
        damage = None if routed else damage_record(
            section, rest_length, spring_like=spring_like, travels=travels)
        if damage is not None and section_properties:
            damage.update(section_properties)
        bushing = _bushing(frame_mount) if mc.bushed else None
        # A LINE IS STRUCTURAL WHEN IT WANTS TO BE. A hose carries no
        # load, and `routed` says so; but a rigid drain, a conduit, a
        # steel steam line CAN carry what hangs on its end -- a trap, a
        # valve, a junction box -- and often does, with a small bracket
        # to help. An author who declares `load_bearing=True` on a routed
        # member gets its real cantilever stiffness, 3EI/L^3 from the
        # section it already carries, plus whatever `support_stiffness_
        # n_per_m` the bracket contributes in parallel; and wrench_paths
        # then walks through it as the compliant link it is.
        # A SUSPENSION IS A DECLARED STIFFNESS, not a bushing pack. A
        # washer's tub hangs on springs of a few kN/m with dampers; that
        # is nothing like an engine isolator and is authored as its own
        # numbers on the edge, which wrench_paths reads as the compliant
        # link it is.
        suspension = attributes.pop("suspension", None)
        if suspension is not None:
            attributes["suspension"] = {
                "model": "spring-and-damper",
                "linear_stiffness_n_per_m": float(suspension["linear_stiffness_n_per_m"]),
                "damping_ratio": float(suspension.get("damping_ratio", 0.2)),
                "travel_m": float(suspension.get("travel_m", 0.03)),
            }
        load_bearing = attributes.pop("load_bearing", False)
        helped = float(attributes.pop("support_stiffness_n_per_m", 0.0) or 0.0)
        if routed and load_bearing:
            m = section.mat
            k_pipe = (3.0 * m.youngs_pa * section.second_moment_m4
                      / max(rest_length, 1e-3) ** 3)
            attributes["line_support"] = {
                "model": "cantilever-pipe-plus-bracket",
                "pipe_stiffness_n_per_m": k_pipe,
                "bracket_stiffness_n_per_m": helped,
                "linear_stiffness_n_per_m": k_pipe + helped,
                "damping_ratio": 0.02,
                "section_designation": section.designation,
            }
        if seam is not None:
            # a seam's pack is sized by the seam's own length: the count
            # of fasteners is the length over the pitch
            attributes["seam"] = (seam.pack(rest_length) if hasattr(seam, "pack")
                                  else dict(seam))
        self.edges.append({
            "identity": identity, "a": a, "b": b, "constraint": constraint,
            "assembly": self.assembly,
            # DECLARED ON THE EDGE, so a solver reads what this member
            # transmits instead of matching its spelling against a set
            # of its own.
            # THE TOKEN IS THE IDENTITY; the spelling above stays for
            # people and for documents already written.
            "constraint_token": mc.token,
            "freedoms_released": list(mc.freedoms()),
            "release_ends": mc.release_ends,
            "routed": mc.routed,
            "rest_length": rest_length, "radius": radius, "palette_role": palette,
            "material": mc.material,
            "in_view": True,
            **({"damage": damage} if damage else {}),
            **({"joint_bushings": {"a": dict(bushing), "b": dict(bushing)}} if bushing else {}),
            **attributes,
        })

    def as_document(self) -> dict:
        return {"schema": "engine-toy-drivetrain-graph-v1", "identity": self.identity,
                "machine": True, "production_vocabulary": True,
                "nodes": self.nodes, "edges": self.edges}

    # -- checks that a graph has to pass before anyone tries to solve it --
    def check(self) -> list[str]:
        problems = []
        # ---- DECLARED CLEAR VOLUMES ARE ENFORCED ----
        # A part can declare a volume that must stay empty -- the column
        # a spent case falls down, the column a mount retracts into.
        # Nothing checked these before, so the ejection path was blocked
        # by six of the rig's own braces and nothing said a word. A
        # clearance that is not checked is a comment.
        for vol in self.clear_volumes:
            lo = np.array(vol["min"], float); hi = np.array(vol["max"], float)
            drawing = {n["identity"] for n in self.nodes
                       if n.get("is_drawing_body")}
            for e in self.edges:
                if e.get("clears") == vol["identity"]:
                    continue
                # A TIE TO A DRAWING BODY IS NOT A PART. The disc that
                # gives an annulus its round shape sits at the ring's
                # centre -- on the axis -- and the ties that make it
                # ride with its own beam model radiate out from there.
                # They are bookkeeping: rigid, not beam-solvable, not in
                # view. Treating them as members means every annulus in
                # the machine reports itself as obstructing whatever is
                # down its own bore, which is a fact about the model and
                # not about the metal.
                if e["a"] in drawing or e["b"] in drawing:
                    continue
                pa = np.array(next(n["reference_position"] for n in self.nodes
                                   if n["identity"] == e["a"]), float)
                pb = np.array(next(n["reference_position"] for n in self.nodes
                                   if n["identity"] == e["b"]), float)
                t = np.linspace(0.0, 1.0, 61)[:, None]
                q = pa + (pb - pa) * t
                hit = int(np.all((q > lo) & (q < hi), axis=1).sum())
                if hit:
                    problems.append(
                        f"{e['identity']}: crosses the clear volume "
                        f"{vol['identity']!r} ({hit}/61 samples) -- "
                        f"{vol.get('note', '')}")
        ids = {n["identity"] for n in self.nodes}
        for e in self.edges:
            for end in ("a", "b"):
                if e[end] not in ids:
                    problems.append(f"{e['identity']}: endpoint {e[end]} does not exist")
            # A MATED PORT PAIR HAS ZERO LENGTH ON PURPOSE. Both ends
            # are on the two surfaces that touch, which is exactly why
            # a ported joint cannot reach into the middle of a body.
            if (e.get("rest_length", 0.0) <= 0.0
                    and not member_constraint(e["constraint"]).zero_length):
                problems.append(f"{e['identity']}: zero rest length between coincident nodes")
        # a structure needs more members than a mechanism: every free
        # node wants at least three non-coplanar connections or it is a
        # hinge the author did not mean to draw
        parent_of = {n["identity"]: n.get("solver_condensed_into")
                     or n["identity"] for n in self.nodes}
        degree = {i: 0 for i in ids}
        for e in self.edges:
            a, b = parent_of.get(e["a"], e["a"]), parent_of.get(e["b"], e["b"])
            if a != b:
                degree[a] = degree.get(a, 0) + 1
                degree[b] = degree.get(b, 0) + 1
        for ident, d in sorted(degree.items()):
            node = next(n for n in self.nodes if n["identity"] == ident)
            if node.get("fixed_to") or node.get("kind") == "recoiling-weapon-mass":
                continue
            # A VOID IS NOT UNDERCONSTRAINED. An internal channel is
            # metal that is not there -- a cored passage with openings,
            # no section, no stiffness and nothing to hold in place. The
            # degree test asks whether a BODY is located, and a hole in
            # a body is located by the body. A two-opening passage is a
            # passage; demanding a third connection of it would mean
            # inventing plumbing to satisfy a structural check.
            if node.get("internal_channel") or node.get("solver_condensed_into"):
                continue
            # A welded/bent fabrication path legitimately has two incident
            # members: its fixed-angle joint transmits moment.  Degree three
            # is a useful space-frame heuristic, not a law for a continuous
            # strap or weld seam.
            if node.get("fabrication_path_node"):
                continue
            if d < 3:
                problems.append(f"{ident}: only {d} connection(s) -- underconstrained")
        return problems


# =====================================================================
#  THE TURRET
# =====================================================================

# =====================================================================
#  POSING THE DOCUMENT: WHAT MOVES WHEN THE MOUNT AIMS
# =====================================================================
#: Each stage names the one it rides on. This is the whole kinematic
#: chain of the station, written once: a body's declared `motion_group`
#: plus this table is everything needed to put it where the mount
#: currently has it, and nothing anywhere has to recognise a name.
MOTION_PARENTS = {
    "frame": None,
    "carriage": "frame",
    "traverse": "carriage",
    "elevation": "traverse",
    "fine": "elevation",
    "mg-yaw": "traverse",
    "mg-pitch": "mg-yaw",
}


def _translation(v):
    import numpy as np
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _rotation_about(axis, degrees: float, pivot):
    """A rotation of `degrees` about `axis` through `pivot`, in the
    document's own rest coordinates."""
    import numpy as np
    a = np.asarray(axis, float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    t = math.radians(degrees)
    c, s_, C = math.cos(t), math.sin(t), 1.0 - math.cos(t)
    x, y, z = a
    r = np.array([
        [c + x * x * C, x * y * C - z * s_, x * z * C + y * s_],
        [y * x * C + z * s_, c + y * y * C, y * z * C - x * s_],
        [z * x * C - y * s_, z * y * C + x * s_, c + z * z * C]])
    m = np.eye(4)
    m[:3, :3] = r
    p = np.asarray(pivot, float)
    m[:3, 3] = p - r @ p
    return m


def pose_frames(doc: dict, *, hoist_m: float = 0.0, traverse_deg: float = 0.0,
                elevation_deg: float = 0.0, fine_yaw_deg: float = 0.0,
                fine_pitch_deg: float = 0.0, recoil_m: float = 0.0,
                mg_yaw_deg: float = 0.0, mg_pitch_deg: float = 0.0,
                whip_rad: float = 0.0) -> dict:
    """The transform each motion group is under, for this aim.

    `whip_rad` is the tube's own bending, live. It is not part of the
    aim: nothing commanded it and no actuator holds it. The barrel is a
    beam with mass and stiffness, every shot rings it, and at any real
    rate of fire it never stops -- so the muzzle is pointing somewhere
    slightly different at the instant each round leaves. Rotating the
    recoiling group by it is what makes the drawn tube, the shot tube
    and the targeting tube agree that it is bent.

    Split out from `pose_document` because two things want it: moving
    the document (for ray casts, friendly-fire masks and anything that
    reasons about where the parts ARE) and moving a mesh's vertices (for
    drawing it). Computing the chain twice in two places is how the
    drawn turret and the shot turret come to disagree, so they share
    this."""
    import numpy as np

    rest = {n["identity"]: np.asarray(n["reference_position"], float)
            for n in doc["nodes"]}

    def at(identity, fallback):
        return rest.get(identity, np.asarray(fallback, float))

    trunnion = at("turret.pitch", (0.0, 0.0, 0.0))
    mg_post = at("mg.yaw", (0.0, 0.0, 0.0))
    mg_trun = at("mg.pitch", mg_post)
    fine_hub = at("fine.hub", trunnion)

    # THE MOUNT TURNS ABOUT ITS OWN AXIS, not about the origin. Those
    # are the same point only while the document sits where it was
    # authored; place the station anywhere else -- on a pole four metres
    # from the engine, say -- and a yaw taken about the origin swings
    # the whole turret round the scene instead of turning it on its
    # ring. The carriage hub IS the axis, so it is read from the
    # document rather than assumed.
    axis_point = at("carriage.hub", (0.0, 0.0, 0.0))
    yaw_axis = (float(axis_point[0]), 0.0, float(axis_point[2]))

    frames = {"frame": np.eye(4)}
    frames["carriage"] = _translation((0.0, hoist_m, 0.0))
    frames["traverse"] = frames["carriage"] @ _rotation_about(
        (0.0, 1.0, 0.0), traverse_deg, yaw_axis)
    # nose-up is positive, about the trunnion's own pin (the X axis)
    frames["elevation"] = frames["traverse"] @ _rotation_about(
        (1.0, 0.0, 0.0), -elevation_deg, trunnion)
    frames["fine"] = (frames["elevation"]
                      @ _rotation_about((0.0, 1.0, 0.0), fine_yaw_deg, fine_hub)
                      @ _rotation_about((1.0, 0.0, 0.0), -fine_pitch_deg, fine_hub))
    frames["mg-yaw"] = frames["traverse"] @ _rotation_about(
        (0.0, 1.0, 0.0), mg_yaw_deg, mg_post)
    frames["mg-pitch"] = frames["mg-yaw"] @ _rotation_about(
        (1.0, 0.0, 0.0), -mg_pitch_deg, mg_trun)
    # THE TUBE SLIDES. Recoil is a translation along the bore, taken in
    # the gun's own frame before that frame is aimed -- which is why it
    # is composed here and not added to the world position afterwards.
    # THE TUBE BENDS ABOUT THE TRUNNION, which is where it is held.
    # Applied after the fine stage and before recoil, because the whip
    # is the tube's own shape and the recoil slides along that shape.
    frames["recoil"] = (frames["fine"]
                        @ _rotation_about((1.0, 0.0, 0.0),
                                          math.degrees(whip_rad), trunnion)
                        @ _translation((0.0, 0.0, -abs(recoil_m))))
    return frames


def pose_document(doc: dict, *, hoist_m: float = 0.0, traverse_deg: float = 0.0,
                  elevation_deg: float = 0.0, fine_yaw_deg: float = 0.0,
                  fine_pitch_deg: float = 0.0, recoil_m: float = 0.0,
                  mg_yaw_deg: float = 0.0, mg_pitch_deg: float = 0.0,
                  whip_rad: float = 0.0) -> dict:
    """Put the mount where it is currently pointing.

    Returns a NEW document with every body moved; the rest positions in
    the original are never touched, so this is idempotent and calling it
    twice for one state cannot drift. That matters more than it sounds:
    the previous kinematic stand-in accumulated its transforms into the
    graph it was given, and the only reason it did not visibly wander
    was that it recomputed from a saved base pose it had to remember to
    keep.

    The chain is exactly the mount's own: the hoist lifts the carriage
    out of the well, the ring traverses everything above it, the
    trunnion elevates the cradle, the six-leg platform twitches the gun
    about whatever pose the coarse stages left, and the tube recoils
    along its own bore. The machine gun branches off the traverse and
    aims by itself from there.

    Every pivot is read from the document's REST positions, which is
    what makes composing the stages correct: each stage's local
    rotation is written in rest coordinates and then carried by its
    parent's transform, so a trunnion that has been hoisted and
    traversed still elevates about its own pin rather than about where
    its pin used to be.
    """
    import numpy as np

    frames = pose_frames(doc, hoist_m=hoist_m, traverse_deg=traverse_deg,
                         elevation_deg=elevation_deg, fine_yaw_deg=fine_yaw_deg,
                         fine_pitch_deg=fine_pitch_deg, recoil_m=recoil_m,
                         mg_yaw_deg=mg_yaw_deg, mg_pitch_deg=mg_pitch_deg, whip_rad=whip_rad)
    rest = {n["identity"]: np.asarray(n["reference_position"], float)
            for n in doc["nodes"]}
    recoiling = frames["recoil"]

    nodes = []
    for n in doc["nodes"]:
        group = n.get("motion_group", "frame")
        m = frames.get(group, frames["frame"])
        if n.get("recoils"):
            m = recoiling
        p = rest[n["identity"]]
        world = m @ np.array([p[0], p[1], p[2], 1.0])
        moved = dict(n)
        moved["reference_position"] = [float(world[0]), float(world[1]), float(world[2])]
        # A BODY'S ORIENTATION IS PART OF ITS POSE. The mesh builds a
        # tube along `drum_axis` and a ring about `ring_axis` and reads
        # them straight off the node -- move the barrel's centre without
        # turning its axis and the gun elevates while still pointing
        # where it was, which is worse than not moving at all.
        for key in ("drum_axis", "ring_axis", "slide_axis"):
            axis = n.get(key)
            if axis is None:
                continue
            a = m[:3, :3] @ np.asarray(axis, float)
            moved[key] = [float(a[0]), float(a[1]), float(a[2])]
        nodes.append(moved)
    return {**doc, "nodes": nodes}


def author_captive_pinion_ring(g: "ProductionGraph", *, prefix: str,
                               lower_ring_node: str, frame_node: str,
                               ring_radius_m: float, ring_y: float,
                               pinions,
                               upper_ring: bool = False,
                               upper_ring_reference: str = "same-body",
                               plate_gap_m: float = 0.075,
                               module_m: float = 0.008,
                               face_width_m: float = 0.060,
                               upper_ring_tooth_delta: int = 0) -> dict:
    """Author a captive pinion drive: real pinion bodies on a ring, with
    an optional second ring laid over them.

    Parametric on purpose -- this arrangement is useful far beyond a
    turret (a crane house, a dish, a rotating bed, a winch) and nothing
    in it knows what it is turning. Give it a ring to drive, something
    to react against, and a list of pinions, and it emits the hardware.

    `upper_ring=False` is the ordinary single-ring drive: each pinion
    hangs off its carrier and the carrier eats the whole separating
    force. `upper_ring=True` lays a second ring over the first with
    `plate_gap_m` between them and runs the pinions in the middle,
    where they are held in mesh from both sides.

    `upper_ring_reference` then decides what that second ring IS:

      "same-body"  it turns with the ring being driven. Two meshes per
                   pinion sharing the load, opposed separating forces,
                   a straddled shaft. A stiffer, quieter, longer-lived
                   drive of the same ratio.

      "frame"      it is held to the structure instead, and the pinion
                   becomes a planet between a moving ring and a fixed
                   one. With `upper_ring_tooth_delta` teeth of
                   difference the ratio stops being ring/pinion and
                   becomes N_l/(N_l - N_u) -- a very large reduction in
                   a very flat package, at a real cost in efficiency.

    The mechanics (separating force, carrier load, shaft bending,
    ratio, efficiency) live in `slew_drive.CaptivePinionRing`; this
    authors the bodies and members that that law describes. Each edge
    carries its own declared parameters, so nothing downstream has to
    guess the arrangement from a name.
    """
    from slew_drive import RingGear, CaptivePinionRing

    lower = RingGear(pitch_diameter_m=ring_radius_m * 2.0, module_m=module_m,
                     face_width_m=face_width_m, plate_gap_m=plate_gap_m)
    upper = None
    if upper_ring:
        upper = RingGear(pitch_diameter_m=ring_radius_m * 2.0
                         + upper_ring_tooth_delta * module_m,
                         module_m=module_m, face_width_m=face_width_m,
                         plate_gap_m=plate_gap_m)
        upper_y = ring_y + plate_gap_m
        # the upper ring moves with whatever it is referenced to:
        # with the driven ring when it is a second driving mesh,
        # with the frame when it is the stationary member of a
        # differential planetary.
        g.motion_group = ("traverse" if upper_ring_reference == "same-body"
                          else "carriage")
        g.assembly = "traverse-ring-drive"
        # A RING GEAR IS A RIM. Authored as one node on the axis, every
        # web and every mesh became a spoke to the centre -- and the
        # centre of this machine is the shaft a spent case falls down.
        # Emitted as rim nodes it is both the right shape and out of
        # the way, and the mesh loads land where the teeth are.
        UPPER_N = 6

        def _upper_at(i):
            a = 2.0 * math.pi * i / UPPER_N
            return (upper.radius_m * 0.92 * math.cos(a), upper_y,
                    upper.radius_m * 0.92 * math.sin(a))

        def _upper_near(x, z):
            return min(range(UPPER_N), key=lambda j: (
                (_upper_at(j)[0] - x) ** 2 + (_upper_at(j)[2] - z) ** 2))

        for i in range(UPPER_N):
            g.node(f"{prefix}.upper_ring.{i}", list(_upper_at(i)), "ring-gear",
                   material="hardened-steel",
                   half_extent_m=(0.060, face_width_m / 2.0, 0.060),
                   ring_teeth=upper.teeth, module_m=module_m,
                   ring_radius_m=upper.radius_m,
                   ring_reference=upper_ring_reference)
        for i in range(UPPER_N):
            g.edge(f"{prefix}.upper_rim.{i}", f"{prefix}.upper_ring.{i}",
                   f"{prefix}.upper_ring.{(i + 1) % UPPER_N}", "rigid-distance",
                   radius=0.024, palette="rollbar-silver",
                   load_path="the-upper-ring-gear-itself")
        # WHAT THE UPPER RING IS BOLTED TO IS THE WHOLE DISTINCTION.
        # Tied to the ring below it, it is a second driving mesh. Tied
        # to the frame, it is the stationary member of a differential
        # planetary and the drive's ratio changes completely.
        # WHICH node: these may be given as a function of position, so a
        # member reaches the rim station beside it rather than a point
        # on the axis. A ring is a rim; asking for "the ring node" was
        # the question that produced spokes through the middle.
        tied_to = lower_ring_node if upper_ring_reference == "same-body" else frame_node
        _frame_is_callable = callable(frame_node)
        for k, ang in enumerate((0.0, 2.0943951, 4.1887902)):
            g.node(f"{prefix}.upper_stud.{k}",
                   [upper.radius_m * 0.84 * math.cos(ang), upper_y,
                    upper.radius_m * 0.84 * math.sin(ang)],
                   "chassis-load-node", material="steel-plate",
                   half_extent_m=(0.022, 0.022, 0.022))
            _u = _upper_near(upper.radius_m * 0.84 * math.cos(ang),
                             upper.radius_m * 0.84 * math.sin(ang))
            for _d in (0, 1):
                g.edge(f"{prefix}.upper_stud_web.{k}.{_d}",
                       f"{prefix}.upper_stud.{k}",
                       f"{prefix}.upper_ring.{(_u + _d) % UPPER_N}",
                       "rigid-distance", radius=0.018,
                       palette="rollbar-silver",
                       load_path="stud-into-the-rim-beside-it")
            g.edge(f"{prefix}.upper_ground.{k}", f"{prefix}.upper_stud.{k}",
                   tied_to(upper.radius_m * 0.84 * math.cos(ang),
                           upper.radius_m * 0.84 * math.sin(ang))
                   if callable(tied_to) else tied_to,
                   "rigid-distance", radius=0.024, palette="rollbar-silver",
                   frame_mount=(tied_to is frame_node),
                   load_path=f"upper-ring-referenced-to-the-{upper_ring_reference}")
        # the three studs tie to each other: a spider, not three loose
        # posts. Without this ring of ties each stud has only its web
        # and its ground tie, which is two load paths and a hinge.
        for k in range(3):
            g.edge(f"{prefix}.upper_spider.{k}", f"{prefix}.upper_stud.{k}",
                   f"{prefix}.upper_stud.{(k + 1) % 3}", "rigid-distance",
                   radius=0.016, palette="rollbar-silver",
                   load_path="upper-ring-spider-closing-the-stud-triangle")

    pinion_y = ring_y + (plate_gap_m / 2.0 if upper_ring else 0.0)
    emitted = []
    for k, spec in enumerate(pinions):
        name = spec["identity"]
        teeth = int(spec.get("teeth", 18))
        tech = spec.get("technology", "electric")
        torque = float(spec.get("drive_torque_nm", 3800.0))
        ang = 2.0 * math.pi * k / max(len(pinions), 1)
        pr = teeth * module_m / 2.0            # pinion pitch radius
        cx = (ring_radius_m + pr) * math.cos(ang)
        cz = (ring_radius_m + pr) * math.sin(ang)
        law = CaptivePinionRing(
            identity=f"{prefix}.{name}", lower_ring=lower, upper_ring=upper,
            upper_ring_reference=upper_ring_reference, pinion_teeth=teeth,
            plate_gap_m=plate_gap_m)
        tangential_n = torque / max(ring_radius_m, 1e-6)

        # the ring turns past the pinion; the pinion stays on the carriage
        g.motion_group = "carriage"
        g.assembly = "traverse-ring-drive"
        g.node(f"{prefix}.pinion.{name}", [cx, pinion_y, cz], "drive-pinion",
               material="hardened-steel", shape="drum", drum_radius_m=pr,
               drum_length_m=face_width_m,
               half_extent_m=(pr, face_width_m / 2.0, pr),
               pinion_teeth=teeth, pinion_technology=tech)
        # the carrier sits outboard of the mesh and is what the drive
        # unit bolts to; it is the member that eats whatever separating
        # force the second ring did not cancel
        g.node(f"{prefix}.carrier.{name}",
               [cx * 1.14, pinion_y, cz * 1.14], "pinion-carrier",
               material="cast-iron", half_extent_m=(0.070, 0.075, 0.070))
        emitted.append(law)

        g.edge(f"{prefix}.mesh_lower.{name}", f"{prefix}.pinion.{name}",
               lower_ring_node(cx, cz) if callable(lower_ring_node)
               else lower_ring_node,
               "captive-pinion-mesh", radius=0.026, palette="rollbar-silver",
               pinion_teeth=teeth, ring_teeth=lower.teeth, module_m=module_m,
               pressure_angle_deg=law.pressure_angle_deg,
               pinion_technology=tech, drive_torque_nm=torque,
               driven_by=f"slew_drive.pinion.{name}",
               tangential_force_n=round(tangential_n, 1),
               separating_force_n=round(law.separating_force_n(
                   law.tooth_load_per_mesh_n(tangential_n)), 1),
               reduction=round(law.reduction(), 3),
               load_path=f"{tech}-pinion-tooth-load-into-the-driven-ring")
        if upper is not None:
            g.edge(f"{prefix}.mesh_upper.{name}", f"{prefix}.pinion.{name}",
                   f"{prefix}.upper_ring.{_upper_near(cx, cz)}",
                   "captive-pinion-mesh", radius=0.026,
                   palette="rollbar-silver", pinion_teeth=teeth,
                   ring_teeth=upper.teeth, module_m=module_m,
                   pressure_angle_deg=law.pressure_angle_deg,
                   upper_ring_reference=upper_ring_reference,
                   drives=(upper_ring_reference == "same-body"),
                   load_path=("second-driving-mesh-sharing-the-load"
                              if upper_ring_reference == "same-body"
                              else "reaction-mesh-against-the-stationary-ring"))
        g.edge(f"{prefix}.carrier_bearing.{name}", f"{prefix}.pinion.{name}",
               f"{prefix}.carrier.{name}", "pinion-carrier-bearing", radius=0.030,
               palette="rollbar-silver", straddled=law.straddled,
               carrier_radial_load_n=round(law.carrier_radial_load_n(tangential_n), 1),
               shaft_bending_nm=round(law.shaft_bending_moment_nm(tangential_n), 1),
               shaft_diameter_m=law.shaft_diameter_m,
               load_path=("shaft-supported-either-side-of-the-mesh" if law.straddled
                          else "cantilevered-shaft-carrying-the-whole-separating-force"))
        # the retainer that keeps the pinion on its shaft: real, and it
        # is also the third load path that stops the pinion body being
        # a two-connection free node
        g.edge(f"{prefix}.thrust_retainer.{name}", f"{prefix}.pinion.{name}",
               f"{prefix}.carrier.{name}", "preloaded-captive-body-retainer-spring",
               radius=0.012, palette="actuator-yellow",
               load_path="axial-retention-of-the-pinion-on-its-shaft")
        for j, lean in enumerate((0.86, 1.0)):
            g.edge(f"{prefix}.carrier_brace.{name}.{j}", f"{prefix}.carrier.{name}",
                   frame_node(cx * 1.14, cz * 1.14) if callable(frame_node)
                   else frame_node,
                   "rigid-distance", radius=0.026 * lean,
                   palette="rollbar-silver", frame_mount=True,
                   load_path="pinion-drive-reaction-into-the-frame")
    return {"lower_ring": lower, "upper_ring": upper, "laws": emitted}


def _build_well_carriage_and_race(g: ProductionGraph, *, well_half: float = 0.80,
                                  well_depth: float = 1.05, hoist_stroke_m: float = 0.850,
                                  ring_radius: float = 0.78) -> dict:
    """The structure every turret on this well shares: the fixed box,
    the carriage the hoist raises, and the distributed slew race.

    Factored out so two different guns on this well cannot quietly
    drift apart the way this file and machines.py already had --
    one well, described once, and every builder that wants a turret
    calls this and then authors its own head on top of it."""
    h = well_half
    floor_y = -well_depth
    rim_y = 0.0

    # ---- THE WELL: a triangulated box, fixed to the world ----
    # Four feet and four rim corners. The rim is what the hoist pushes
    # against and what the guides run in, so it has to be a real frame:
    # perimeter rails, both face diagonals, and the verticals.
    corners = (("nn", -1, -1), ("np", -1, 1), ("pn", 1, -1), ("pp", 1, 1))
    for name, sx, sz in corners:
        # the well is bolted to the world: this is ground
        g.motion_group = "frame"
        g.assembly = "well-and-foundation"
        g.node(f"well.foot.{name}", [sx * h, floor_y, sz * h],
               "structural-body-pin-frame-foot", fixed_to="world",
               material="steel-plate")
        g.node(f"well.rim.{name}", [sx * h, rim_y, sz * h],
               "chassis-load-node", fixed_to="world", material="steel-plate",
               structural_deformable=True)
    ring_pairs = (("nn", "np"), ("np", "pp"), ("pp", "pn"), ("pn", "nn"))
    for a, b in ring_pairs:
        g.edge(f"well.floor_rail.{a}_{b}", f"well.foot.{a}", f"well.foot.{b}",
               "rigid-distance", radius=0.030, palette="chassis-grey")
        g.edge(f"well.rim_rail.{a}_{b}", f"well.rim.{a}", f"well.rim.{b}",
               "rigid-distance", radius=0.030, palette="chassis-grey")
    for name, _sx, _sz in corners:
        g.edge(f"well.column.{name}", f"well.foot.{name}", f"well.rim.{name}",
               "rigid-distance", radius=0.034, palette="chassis-grey")
    # the diagonals: without these the box is a mechanism, not a frame
    g.edge("well.floor_diagonal.a", "well.foot.nn", "well.foot.pp", "rigid-distance",
           radius=0.022, palette="chassis-grey")
    g.edge("well.floor_diagonal.b", "well.foot.np", "well.foot.pn", "rigid-distance",
           radius=0.022, palette="chassis-grey")
    g.edge("well.rim_diagonal.a", "well.rim.nn", "well.rim.pp", "rigid-distance",
           radius=0.022, palette="chassis-grey")
    g.edge("well.rim_diagonal.b", "well.rim.np", "well.rim.pn", "rigid-distance",
           radius=0.022, palette="chassis-grey")
    for (a, b) in ring_pairs:
        g.edge(f"well.wall_brace.{a}_{b}", f"well.foot.{a}", f"well.rim.{b}",
               "rigid-distance", radius=0.020, palette="chassis-grey")

    # ---- THE CARRIAGE that rises ----
    carriage_y = floor_y + 0.42
    for name, sx, sz in corners:
        # everything from here rides up out of the well on the hoist
        g.motion_group = "carriage"
        g.assembly = "carriage"
        g.node(f"carriage.corner.{name}", [sx * ring_radius, carriage_y, sz * ring_radius],
               "chassis-load-node", material="steel-plate", structural_deformable=True)
    g.node("carriage.hub", [0.0, carriage_y, 0.0], "load-bearing-structure",
           material="steel-plate")
    for a, b in ring_pairs:
        g.edge(f"carriage.rail.{a}_{b}", f"carriage.corner.{a}", f"carriage.corner.{b}",
               "rigid-distance", radius=0.026, palette="rollbar-silver")
    for name, _sx, _sz in corners:
        g.edge(f"carriage.spoke.{name}", "carriage.hub", f"carriage.corner.{name}",
               "rigid-distance", radius=0.024, palette="rollbar-silver")

    # ---- THE HOIST: hydraulic members, commanded by rest length ----
    # One central ram carries the load. Three guide struts triangulate
    # the carriage against the well so the assembly rises straight
    # instead of swinging on the end of the ram.
    # the ram's own foot stays on the floor
    g.motion_group = "frame"
    g.assembly = "hoist"
    g.node("hoist.base", [0.0, floor_y, 0.0], "load-bearing-structure",
           fixed_to="world", material="steel-plate",
           half_extent_m=(0.16, 0.09, 0.16))
    g.edge("hoist.ram", "hoist.base", "carriage.hub", "linear-hydraulic-actuator",
           radius=0.0625, palette="actuator-yellow", frame_mount=True,
           family="linear-hydraulic-actuator",
           kind="commanded-rest-length-hoist-ram",
           state=["commanded_rest_length_m", "relief_stroke_m", "relief_velocity_m_s",
                  "axial_force_n", "dissipated_energy_j", "temperature_k", "health"],
           # THE COMMAND. engine_toy computes the fluid side -- what
           # pressure the pack can hold and how much volume it has put
           # in -- and hands over a rest length. The solver decides
           # where the structure actually ends up.
           commanded_rest_length_m=0.0,
           minimum_rest_length_m=0.0,
           maximum_rest_length_m=float(hoist_stroke_m),
           bore_m=0.125, rod_m=0.070,
           # series compliance, so it is backdrivable rather than a
           # rigid position source that would drive infinite force into
           # anything that got in its way
           linear_stiffness_n_per_m=2.4e7,
           linear_damping_n_s_per_m=9.0e4,
           holding_force_n=180_000.0, relief_force_n=260_000.0,
           maximum_relief_stroke_m=0.020, maximum_relief_rate_m_per_s=0.35,
           energy_law=("commanded-length-plus-series-relief-stroke; relief work becomes heat; "
                       "extension draws metered volume from the turret hydraulic manifold"),
           failure_mode="hold-current-total-length-until-repaired-or-line-opens")
    for name, sx, sz in (("nn", -1, -1), ("np", -1, 1), ("pn", 1, -1)):
        g.edge(f"hoist.guide.{name}", f"well.rim.{name}", f"carriage.corner.{name}",
               "prismatic-guide-strut", radius=0.028, palette="chassis-grey",
               load_path="carriage-lateral-and-moment-reaction-into-well-rim",
               free_axis="world-vertical",
               lateral_stiffness_n_per_m=5.5e7, lateral_damping_n_s_per_m=4.2e4)

    # ---- THE RING, TRAVERSE, ELEVATION, WEAPON ----
    # THE SLEW RING IS A DISTRIBUTED RACE, not a point. A turret of this
    # mass hangs its whole rotating weight and its recoil moment on the
    # bearing, and a single central connection cannot carry a moment at
    # all -- it is a ball joint. Real slew rings take the load around
    # their circumference, so the rolling contact is authored as one
    # bearing member per race pad, with the rotating pads tied into a
    # rigid ring of their own.
    #
    # This is not a detail. The structural check in this module flagged
    # the single-point version as underconstrained, which it was: the
    # turret would have swung freely on its own mount.
    race_y = carriage_y + 0.10
    for name, sx, sz in corners:
        # the race and the yaw ring turn on top of the carriage
        g.motion_group = "traverse"
        g.assembly = "slew-ring"
        g.node(f"turret.race.{name}", [sx * ring_radius * 0.82, race_y, sz * ring_radius * 0.82],
               "slew-ring-race-pad", material="hardened-steel")
        g.edge(f"turret.slew_bearing.{name}", f"carriage.corner.{name}", f"turret.race.{name}",
               "gimbal-yaw-bearing", radius=0.040, frame_mount=True,
               load_path="rotating-mass-and-recoil-moment-through-race-into-carriage",
               rolling_contact=True)
    for a, b in ring_pairs:
        g.edge(f"turret.race_rail.{a}_{b}", f"turret.race.{a}", f"turret.race.{b}",
               "rigid-distance", radius=0.028, palette="rollbar-silver")
    g.edge("turret.race_diagonal.a", "turret.race.nn", "turret.race.pp", "rigid-distance",
           radius=0.020, palette="rollbar-silver")
    g.edge("turret.race_diagonal.b", "turret.race.np", "turret.race.pn", "rigid-distance",
           radius=0.020, palette="rollbar-silver")
    g.assembly = "slew-ring"
    g.node("turret.yaw", [0.0, race_y, 0.0], "yaw-clutch-bearing",
           material="hardened-steel", shape="ring", ring_axis=(0.0, 1.0, 0.0),
           ring_outer_radius_m=ring_radius * 1.02, ring_inner_radius_m=ring_radius * 0.62,
           ring_thickness_m=0.075,
           half_extent_m=(ring_radius * 1.02, 0.038, ring_radius * 1.02))
    for name, _sx, _sz in corners:
        g.edge(f"turret.yaw_spoke.{name}", "turret.yaw", f"turret.race.{name}",
               "rigid-distance", radius=0.026, palette="rollbar-silver")
    return {"corners": corners, "ring_pairs": ring_pairs, "carriage_y": carriage_y,
            "race_y": race_y, "floor_y": floor_y, "ring_radius": ring_radius}


def _build_outrigger_base_and_race(g: ProductionGraph, *,
                                   base_half_x: float = 1.35,
                                   base_half_z: float = 1.15,
                                   bay_depth: float = 1.15,
                                   lift_stroke_m: float = 1.30,
                                   design_lift_mass_kg: float = 6_200.0,
                                   ring_radius: float = 0.78) -> dict:
    """The four-post outrigger base: what this station actually stands on.

    NOT A HOIST. The previous base was a well with a single central ram
    that raised the whole turret out of a pit, and that one ram ran
    straight up the middle of the machine -- through the column a spent
    case has to fall down, which is why the ejection check found it
    obstructing 60 of 61 samples. A gun that cannot get rid of its
    brass is a gun that fires once.

    What replaces it is the thing the job actually wants: a rectangular
    base with the turret at its top centre, standing on four square
    steel posts, each with its own hydraulic outrigger jack. The jacks
    do not aim anything and do not move in action -- they LIFT THE WHOLE
    MACHINE, high enough to back a trailer or a chassis underneath and
    set it down on it. That is how this gets moved, and it is why the
    stroke is a declared clearance rather than a number.

    The volume inside the base is not packaging. It is the magazine
    (ammunition in, over the top of the gun), the case bin (brass out,
    straight down the column), general stores, and a wall of hookup
    ports that any heavy machine -- a C18 first of all -- plugs into.
    Plus just enough of its own plant to run alone, badly, when nothing
    is plugged in.
    """
    deck_y = 0.0
    #: HOW HIGH THE GUN SITS ABOVE THE RING IT TURNS ON.
    #:
    #: The rings stay on the deck. What rises is the gun assembly -- the
    #: trunnion, the cradle, the tube and the recoil tank slung under
    #: it -- carried up there by the standards at the end of the
    #: turntable floor.
    #:
    #: It was built with the gun essentially sitting on its own
    #: turntable, and once the trunnion moved to the balance point and
    #: the tank was hung under the barrel, the bottom of that stack came
    #: out at y = -0.094 against a ring face at y = +0.100. The whole
    #: lower half of the gun was inside the base.
    #:
    #: RAISING THE RING WAS THE WRONG FIX and I tried it first: it lifts
    #: the bearing, the drive and the turntable along with the gun, which
    #: adds a metre of lever between the deck and everything the ring
    #: carries and solves a clearance problem by moving the machine's
    #: whole mass further from its own foundation. The gun goes up. The
    #: rings stay down.
    GUN_RISE_M = 1.00
    floor_y = -bay_depth
    corners = (("nn", -1, -1), ("np", -1, 1), ("pn", 1, -1), ("pp", 1, 1))
    ring_pairs = (("nn", "np"), ("np", "pp"), ("pp", "pn"), ("pn", "nn"))
    # ---- WHERE THE LEGS GO ----
    # Four corners is fine for a small base and wrong for this one:
    # 4.88 m of deck between two legs is a long span to hang a gun off,
    # and when a shot lands the whole overturning moment goes into two
    # of the four. Mid-edge legs halve every span, put a foot under the
    # middle of the widest side, and -- because the righting moment is
    # what actually stops a gun platform going over -- put support
    # directly under the line the recoil pushes about.
    #
    # Declared as a list because that is the honest shape of it: the
    # leg set is a property of the base, and the rest of this function
    # reads it rather than assuming the number four.
    legs = (("nn", -1.0, -1.0), ("np", -1.0, 1.0),
            ("pn", 1.0, -1.0), ("pp", 1.0, 1.0),
            ("fn", 0.0, -1.0), ("fp", 0.0, 1.0),
            ("mn", -1.0, 0.0), ("mp", 1.0, 0.0))

    # ---- THE DECK: a rectangular frame with an OPEN CENTRE ----
    # The centre carries no structure at all. Everything the turret
    # needs is taken around a ring beam under the slew race, and the
    # middle is left as a clear shaft from the breech flat to the bin.
    g.motion_group = "frame"
    g.assembly = "base-frame"
    for name, sx, sz in corners:
        g.node(f"deck.corner.{name}", [sx * base_half_x, deck_y, sz * base_half_z],
               "chassis-load-node", material="steel-plate",
               structural_deformable=True, half_extent_m=(0.11, 0.11, 0.11))
    RING_N = 8

    def _ring_index(x, z):
        return min(range(RING_N), key=lambda j: (
            (ring_radius * math.cos(2.0 * math.pi * j / RING_N) - x) ** 2
            + (ring_radius * math.sin(2.0 * math.pi * j / RING_N) - z) ** 2))

    for i in range(RING_N):
        a = 2.0 * math.pi * i / RING_N
        g.node(f"deck.ring.{i}",
               [ring_radius * math.cos(a), deck_y, ring_radius * math.sin(a)],
               "chassis-load-node", material="steel-plate",
               structural_deformable=True, half_extent_m=(0.08, 0.055, 0.08))
    for a, b in ring_pairs:
        g.edge(f"deck.rail.{a}_{b}", f"deck.corner.{a}", f"deck.corner.{b}",
               "rigid-distance", radius=0.038, palette="chassis-grey",
               load_path="deck-perimeter-beam")
    for i in range(RING_N):
        g.edge(f"deck.ring_beam.{i}", f"deck.ring.{i}",
               f"deck.ring.{(i + 1) % RING_N}", "rigid-distance",
               radius=0.032, palette="rollbar-silver",
               load_path="ring-beam-carrying-the-slew-race")
    # corners tie into the ring beam, not into a hub in the middle
    for i, (name, sx, sz) in enumerate(corners):
        near = min(range(RING_N), key=lambda j: (
            (ring_radius * math.cos(2 * math.pi * j / RING_N) - sx * base_half_x) ** 2
            + (ring_radius * math.sin(2 * math.pi * j / RING_N) - sz * base_half_z) ** 2))
        for d in (0, 1):
            g.edge(f"deck.brace.{name}.{d}", f"deck.corner.{name}",
                   f"deck.ring.{(near + d) % RING_N}", "rigid-distance",
                   radius=0.026, palette="chassis-grey",
                   load_path="deck-corner-into-the-ring-beam")

    # ---- THE FOUR POSTS AND THEIR OUTRIGGER JACKS ----
    # Square steel section, corner to corner with the deck. Each carries
    # a quarter of the machine plus the overturning moment a shot puts
    # into it, and each ends in a jack that can pick the whole thing up.
    # over however many legs there actually are
    per_leg_n = design_lift_mass_kg * 9.81 / len(legs) * 1.5
    # ---- A JACK IS A CYLINDER, AND IT IS VERTICAL ----
    # What was here was one edge from the corner of the frame out to a
    # pad offset sideways and down, so it leaned in two directions at
    # once. Nothing leans: an outrigger jack hangs straight down the
    # post, and it has to, because the load it carries is the weight of
    # the machine and weight only points one way. A leaning jack puts a
    # side load into its own rod seals and bends the post.
    #
    # And it is real hardware, not a line:
    #
    #   BARREL   a steel tube bolted to the post foot. It does NOT
    #            move. Its bore is what the pressure acts on.
    #   ROD      a solid shaft inside the barrel, sliding straight down
    #            out of it. This is the part that extends.
    #   PAD      a plate on the bottom of the rod, on the ground.
    #
    # Extending the jack slides the ROD out of the BARREL. The pad is
    # already on the ground, so what actually moves is everything
    # above: the machine rises by exactly the rod travel, and nothing
    # anywhere needs its position edited to make that true.
    from prism_bodies import Prism, emit_prism
    from ballistics import mass_of as _mass_of
    DOWN = (0.0, -1.0, 0.0)
    # ---- THE JACK IS TELESCOPIC, AND IT HAS TO STOW ----
    # A single-stage cylinder that lifts 1.3 m is over 1.6 m long
    # collapsed, and this machine is only 1.05 m from its deck to its
    # bay floor. So a single stage does not stow: retracted, its barrel
    # hung a metre and a half below the machine and the thing could
    # never have sat down, let alone been carried anywhere.
    #
    # Two stages is the ordinary answer and it is what is on every
    # truck crane: the cylinder lives INSIDE the hollow post box, and
    # it pushes out a sleeve which pushes out a rod. Collapsed it is a
    # little over half the travel, so the whole leg tucks up inside the
    # post and the pad sits just below the floor.
    JACK_STAGES = 2
    jack_bore_m = 0.100
    stage_stroke = lift_stroke_m / JACK_STAGES
    barrel_len_m = stage_stroke * 1.16 + 0.10
    barrel_od_m = jack_bore_m + 2.0 * 0.014
    stage_d_m = 0.076
    rod_d_m = 0.060
    g.assembly = "outriggers"
    for name, sx, sz in legs:
        g.motion_group = "frame"
        g.node(f"post.foot.{name}", [sx * base_half_x, floor_y, sz * base_half_z],
               "chassis-load-node", material="steel-plate",
               section="square-hollow", section_across_m=0.22, wall_m=0.012,
               half_extent_m=(0.11, 0.11, 0.11))
        if name not in [c[0] for c in corners]:
            # a mid-edge leg needs a deck node of its own to stand
            # under; it is part of the deck's perimeter beam, not a
            # spur hung off a corner
            g.node(f"deck.corner.{name}",
                   [sx * base_half_x, deck_y, sz * base_half_z],
                   "chassis-load-node", material="steel-plate",
                   structural_deformable=True, half_extent_m=(0.10, 0.10, 0.10))
            _near = _ring_index(sx * base_half_x, sz * base_half_z)
            for _d in (0, 1):
                g.edge(f"deck.brace.{name}.{_d}", f"deck.corner.{name}",
                       f"deck.ring.{(_near + _d) % RING_N}", "rigid-distance",
                       radius=0.026, palette="chassis-grey",
                       load_path="mid-edge-leg-head-into-the-ring-beam")
            for _c, _cs, _cz in corners:
                if (_cs == sx or sx == 0.0) and (_cz == sz or sz == 0.0):
                    g.edge(f"deck.edge.{name}.{_c}", f"deck.corner.{name}",
                           f"deck.corner.{_c}", "rigid-distance",
                           radius=0.034, palette="chassis-grey",
                           load_path="deck-perimeter-beam-through-the-mid-leg")
        g.edge(f"post.column.{name}", f"deck.corner.{name}", f"post.foot.{name}",
               "rigid-distance", radius=0.110, palette="chassis-grey",
               alloy="a36", wall_m=0.012,
               load_path="square-steel-post-carrying-its-share-of-the-machine")

        bx, bz = sx * base_half_x, sz * base_half_z
        # the barrel is UP INSIDE THE POST BOX, not hanging under it
        barrel_top_y = deck_y - 0.12
        barrel = Prism(
            identity=f"outrigger.barrel.{name}", kind="load-bearing-structure",
            centre=(bx, barrel_top_y - barrel_len_m / 2.0, bz), shape="tube",
            axis=DOWN, radius=barrel_od_m / 2.0,
            inner_radius=jack_bore_m / 2.0, length=barrel_len_m,
            material="hardened-steel",
            attributes=dict(part_role="outrigger-cylinder-barrel",
                            bore_m=jack_bore_m, wall_m=0.014,
                            stages=JACK_STAGES, axis_is_vertical=True,
                            housed_in=f"post.column.{name}",
                            mass_in_total=False,
                            mass_kg=round(_mass_of(
                                "hardened-steel",
                                math.pi * ((barrel_od_m / 2) ** 2
                                           - (jack_bore_m / 2) ** 2)
                                * barrel_len_m), 1)))
        b_ports = {"head": barrel.end_point("rear"),
                   "gland": barrel.end_point("front")}
        for q, ang in enumerate((0.0, 1.5708, 3.14159, 4.71239)):
            b_ports[f"flange{q}"] = barrel.side_point(ang, along=-0.90)
        b_nodes = emit_prism(g, barrel, b_ports, motion_group="frame",
                             assembly="outriggers")
        for q in range(4):
            g.edge(f"outrigger.barrel_flange.{name}.{q}",
                   b_nodes[f"flange{q}"], f"post.foot.{name}"
                   if q % 2 else f"deck.corner.{name}",
                   "rigid-distance", radius=0.020, palette="chassis-grey",
                   load_path="cylinder-head-carried-inside-the-post-box")

        # stage 1: a sleeve that comes out of the barrel
        stage_len_m = barrel_len_m - 0.06
        prev_nodes, prev_bottom = b_nodes, barrel_top_y - barrel_len_m
        stage_group = "frame"
        for st in range(JACK_STAGES):
            last = st == JACK_STAGES - 1
            od = rod_d_m if last else stage_d_m
            top = barrel_top_y - 0.03 - st * 0.03
            body = Prism(
                identity=f"outrigger.stage{st}.{name}",
                kind="load-bearing-structure",
                centre=(bx, top - stage_len_m / 2.0, bz),
                shape="cylinder" if last else "tube",
                axis=DOWN, radius=od / 2.0,
                inner_radius=0.0 if last else od / 2.0 - 0.010,
                length=stage_len_m, material="hardened-steel",
                attributes=dict(
                    part_role="outrigger-telescopic-rod" if last
                    else "outrigger-telescopic-sleeve",
                    stage=st, travel_m=stage_stroke,
                    slides_in=(f"outrigger.barrel.{name}" if st == 0
                               else f"outrigger.stage{st - 1}.{name}"),
                    mass_in_total=False,
                    mass_kg=round(_mass_of("hardened-steel",
                                           math.pi * (od / 2) ** 2
                                           * stage_len_m * 0.8), 1)))
            s_ports = {"piston": body.end_point("rear"),
                       "foot": body.end_point("front")}
            for q, ang in enumerate((0.0, 3.14159)):
                s_ports[f"bearing{q}"] = body.side_point(ang, along=-0.60)
            s_nodes = emit_prism(g, body, s_ports,
                                 motion_group=f"outrigger-stage{st}",
                                 assembly="outriggers")
            # THE PRESSURE FACE of this stage, pushing off the one above
            g.edge(f"outrigger.jack.{name}.{st}", prev_nodes["head"]
                   if st == 0 else prev_nodes["piston"], s_nodes["piston"],
                   "linear-hydraulic-actuator",
                   radius=(jack_bore_m if st == 0 else stage_d_m) / 2.0,
                   palette="actuator-yellow", frame_mount=(st == 0),
                   family="linear-hydraulic-actuator",
                   kind="outrigger-lift-jack", stage=st, stages=JACK_STAGES,
                   carries_firing_load=True,
                   axis=DOWN, axis_is_vertical=True,
                   commanded_rest_length_m=0.0,
                   minimum_rest_length_m=0.0,
                   maximum_rest_length_m=float(stage_stroke),
                   bore_m=jack_bore_m if st == 0 else stage_d_m,
                   rod_m=stage_d_m if st == 0 else rod_d_m,
                   piston_area_m2=round(math.pi * ((jack_bore_m if st == 0
                                                    else stage_d_m) / 2) ** 2, 6),
                   linear_stiffness_n_per_m=3.1e7,
                   linear_damping_n_s_per_m=1.1e5,
                   holding_force_n=round(per_leg_n, 1),
                   relief_force_n=round(per_leg_n * 1.4, 1),
                   maximum_relief_stroke_m=0.015,
                   maximum_relief_rate_m_per_s=0.20,
                   load_path="oil-on-this-stage's-piston-pushing-it-out-of-"
                             "the-one-above",
                   failure_mode="lock-at-length-on-a-pilot-operated-check-valve")
            for q in range(2):
                g.edge(f"outrigger.gland.{name}.{st}.{q}",
                       prev_nodes["gland"] if st == 0 else prev_nodes["foot"],
                       s_nodes[f"bearing{q}"], "single-axis-slider",
                       radius=od * 0.6, palette="chassis-grey",
                       free_axis="world-vertical",
                       lateral_stiffness_n_per_m=6.5e7,
                       lateral_damping_n_s_per_m=5.0e4,
                       load_path="stage-guided-by-the-bush-above-it")
            prev_nodes, prev_bottom = s_nodes, top - stage_len_m
            last_nodes = s_nodes

        # the pad, on the ground, directly under the last stage
        g.motion_group = f"outrigger-stage{JACK_STAGES - 1}"
        g.node(f"outrigger.pad.{name}", [bx, prev_bottom - 0.030, bz],
               "structural-body-pin-frame-foot", fixed_to="world",
               material="steel-plate",
               lifts_machine=True,
               lift_stroke_m=lift_stroke_m,
               stages=JACK_STAGES,
               clears_under_m=lift_stroke_m,
               enough_to_insert="trailer-or-chassis",
               pad_bearing_area_m2=0.36 * 0.36,
               capacity_n=round(per_leg_n, 1),
               directly_under=f"outrigger.stage{JACK_STAGES - 1}.{name}",
               half_extent_m=(0.18, 0.030, 0.18))
        for q, tgt in enumerate(("foot", "bearing0", "bearing1")):
            g.edge(f"outrigger.pad_pin.{name}.{q}", f"outrigger.pad.{name}",
                   last_nodes[tgt], "rigid-distance", radius=0.024,
                   palette="chassis-grey",
                   load_path="pad-pinned-to-the-bottom-of-the-last-stage")
        # PUT THE CONTEXT BACK. `g.motion_group` is a builder-wide
        # setting, so leaving it on the last telescopic stage handed
        # that group to the bay floor, the stores and the ammunition
        # racks authored after this loop -- fifty-seven bodies that
        # then "stayed on the ground" when the machine stood up,
        # because they had been quietly declared part of a leg.
        g.motion_group = "frame"

    # the sill runs round the whole perimeter, through the mid-edge
    # feet, so it is one beam and not four with gaps in it
    _perim = ("nn", "fn", "pn", "mp", "pp", "fp", "np", "mn")
    for _i, _a in enumerate(_perim):
        _b = _perim[(_i + 1) % len(_perim)]
        g.edge(f"post.sill.{_a}_{_b}", f"post.foot.{_a}", f"post.foot.{_b}",
               "rigid-distance", radius=0.034, palette="chassis-grey",
               load_path="sill-beam-tying-the-feet-all-the-way-round")
        g.edge(f"post.knee.{_a}_{_b}", f"post.foot.{_a}", f"deck.corner.{_b}",
               "rigid-distance", radius=0.024, palette="chassis-grey",
               load_path="knee-brace-so-the-posts-are-a-frame-not-a-table")

    g.assembly = "base-frame"
    # ---- THE BAY HAS A FLOOR ----
    # Without one the stores were hanging off the sill beams and there
    # was nothing between them and the ground -- so "lifting the
    # machine" would have lifted a frame and left its contents sitting
    # in the dirt. The floor is what makes the interior part of the
    # machine: the jacks pick up the posts, the posts pick up the
    # floor, and everything standing on the floor goes up with it. That
    # is also what turns the outrigger stroke into real CLEARANCE
    # UNDER EVERYTHING, which is the whole point of the stroke.
    #
    # It has a hole in it, in the same place as every other hole in
    # this machine: the case column.
    g.assembly = "base-frame"
    _fx, _fz = base_half_x * 0.86, base_half_z * 0.86
    _floor_pts = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1),
                  (-1, 1), (-1, 0)]
    for _i, (_sx, _sz) in enumerate(_floor_pts):
        g.node(f"bay.floor.{_i}", [_sx * _fx, floor_y, _sz * _fz],
               "chassis-load-node", material="steel-plate",
               structural_deformable=True, half_extent_m=(0.14, 0.020, 0.14),
               part_role="bay-floor-plate",
               lifts_with="the-posts-so-the-interior-goes-up-too")
    for _i in range(len(_floor_pts)):
        g.edge(f"bay.floor_edge.{_i}", f"bay.floor.{_i}",
               f"bay.floor.{(_i + 1) % len(_floor_pts)}", "rigid-distance",
               radius=0.026, palette="chassis-grey",
               load_path="bay-floor-perimeter")
    # the floor hangs on the four post feet
    for _i, (_name, _sx, _sz) in enumerate(corners):
        _near = min(range(len(_floor_pts)),
                    key=lambda j: (_floor_pts[j][0] - _sx) ** 2
                    + (_floor_pts[j][1] - _sz) ** 2)
        for _d in (0, 1):
            g.edge(f"bay.floor_hanger.{_name}.{_d}",
                   f"bay.floor.{(_near + _d) % len(_floor_pts)}",
                   f"post.foot.{_name}", "rigid-distance", radius=0.024,
                   palette="chassis-grey",
                   load_path="the-floor-is-carried-by-the-posts-that-are-lifted")
    # cross ties, routed clear of the case column
    # CHORDS, NOT DIAGONALS. A tie across the floor stiffens it and
    # crosses the case shaft; a chord from corner to corner past the
    # edge midpoint stiffens it just as well and stays out of the hole.
    for _a, _b in ((0, 2), (2, 4), (4, 6), (6, 0)):
        g.edge(f"bay.floor_tie.{_a}_{_b}", f"bay.floor.{_a}",
               f"bay.floor.{_b}", "rigid-distance", radius=0.020,
               palette="chassis-grey",
               load_path="floor-chord-clear-of-the-case-shaft")

    # ---- INSIDE THE BASE ----
    # Everything sits OFF the centreline. The shaft down the middle is
    # the case column and the feed column, and it stays empty.
    g.assembly = "base-stores"
    bay_y = floor_y + 0.34
    stores = (
        ("bay.magazine", (-base_half_x * 0.60, bay_y, base_half_z * 0.42),
         dict(part_role="ammunition-magazine", feeds="turret.breech",
              feed_route="up-the-off-side-and-over-the-top-of-the-gun",
              rounds_stowed=320, mass_kg=980.0)),
        # THE BIN STANDS ON THE FLOOR. Slung underneath it was the
        # lowest thing on the machine, so it ate the whole outrigger
        # stroke: the jacks lifted 1300 mm and the bin still hung
        # 122 mm below the pads. Cases fall through the gun's mounting
        # plate and down the shaft into a bin standing in the bay,
        # which is where a bin goes.
        ("bay.case_bin", (0.0, floor_y + 0.30, 0.0),
         dict(part_role="spent-case-bin", catches="turret.case_ejection_column",
              under_the_column=True, cases_stowed=400, mass_kg=120.0)),
        ("bay.stores", (base_half_x * 0.60, bay_y, -base_half_z * 0.42),
         dict(part_role="general-stores", mass_kg=260.0)),
        ("bay.hookup_wall", (base_half_x * 0.82, bay_y + 0.18, base_half_z * 0.30),
         dict(part_role="service-hookup-wall", mass_kg=85.0)),
        ("bay.aux_plant", (-base_half_x * 0.62, bay_y, -base_half_z * 0.46),
         dict(part_role="standalone-auxiliary-plant",
              runs_alone=True,
              capability_when_alone="traverse-and-fine-aim-only; no sustained "
                                    "fire, no coolant loop, no charging",
              mass_kg=340.0)),
    )
    for ident, pos, attrs in stores:
        g.node(ident, list(pos), "load-bearing-structure",
               material="steel-plate", mass_in_total=False,
               half_extent_m=(0.26, 0.22, 0.26), **attrs)
    # each store is carried by the two nearest post feet and the sill,
    # never by anything crossing the middle
    _carry = {"bay.magazine": ("np", "nn"), "bay.stores": ("pn", "pp"),
              "bay.hookup_wall": ("pp", "pn"), "bay.aux_plant": ("nn", "np"),
              "bay.case_bin": ("nn", "pp")}
    for ident, (a, b) in _carry.items():
        for c in (a, b):
            g.edge(f"{ident.replace('.', '_')}.mount.{c}", ident,
                   f"post.foot.{c}", "rigid-distance", radius=0.020,
                   palette="chassis-grey",
                   load_path="stowage-carried-on-the-sill-frame")
        g.edge(f"{ident.replace('.', '_')}.steady", ident,
               f"deck.corner.{a}", "rigid-distance", radius=0.016,
               palette="chassis-grey", load_path="steady-brace-up-to-the-deck")

    # ---- AMMUNITION, IN STANDARD CONTAINERS, IN RACKS BUILT FOR THEM ----
    # `rounds_stowed=320` was a number with nothing behind it. What is
    # actually carried is CONTAINERS: 120 mm goes one round to a sealed
    # metre-long tube, and .50 BMG goes as linked ribbon coiled in an
    # M2A1 can, which is the box everything in the world is built to
    # accept. The rack is sized around the container, the capacity
    # falls out of the space available, and the loaded mass is real --
    # so the outrigger jacks are sized against a machine with its
    # ammunition in it rather than an empty one.
    from ammunition_stowage import CONTAINERS, Rack, emit_rack
    _main_rack = Rack(
        identity="bay.rack.main",
        container=CONTAINERS["120mm-single-round-tube"],
        centre=(-base_half_x * 0.42, floor_y + 0.44, base_half_z * 0.34),
        span_m=(base_half_x * 0.86, 0.80, base_half_z * 1.10),
        axis="z")
    _mg_rack = Rack(
        identity="bay.rack.mg",
        container=CONTAINERS["m2a1-50cal-can"],
        centre=(base_half_x * 0.52, floor_y + 0.40, -base_half_z * 0.46),
        span_m=(base_half_x * 0.62, 0.70, base_half_z * 0.70),
        axis="x")
    for _rack, _feet in ((_main_rack, ("nn", "np")), (_mg_rack, ("pn", "pp"))):
        _floor_near = sorted(
            range(len(_floor_pts)),
            key=lambda j: (_floor_pts[j][0] * _fx - _rack.centre[0]) ** 2
            + (_floor_pts[j][1] * _fz - _rack.centre[2]) ** 2)[:3]
        emit_rack(g, _rack,
                  carried_by=[f"bay.floor.{j}" for j in _floor_near],
                  motion_group="frame", assembly="ammunition")

    # ---- THE TWO ENGINES, AT THEIR REAL SIZE ----
    # Both come from the catalogue and both are measured, not guessed:
    # `EnginePackage.build` walks each engine's own drivetrain graph and
    # returns the prism its built-in parts actually occupy. Nothing here
    # types in a size.
    #
    # WHY TWO. The turbine will burn anything and makes 1200 kW, and it
    # is ruinous at idle -- a gas turbine's compressor runs whether you
    # are taking power off it or not. So it is the thing you light when
    # you need to move, charge hard, or run the coolant plant flat out.
    # The multifuel is the thing that is actually running most of the
    # time: 122 kW, a fifth of the fuel burn, and enough to hold the
    # bank up, run the hydraulics and keep the barrel cool.
    #
    # AND THE ENVELOPE IS AN UNDERSTATEMENT, which is worth saying
    # rather than discovering later. The turbine's drivetrain graph is
    # built from the placeholder displacement its catalogue entry
    # carries (the real cycle lives in `turbine`), so the prism it
    # returns is a piston engine's package of that size -- not an
    # AGT1500 with its recuperator, which is nearer 1.8 m3. The number
    # below is what the model says; the real engine is bigger.
    from engine_package import EnginePackage as _Pkg
    from turbine_parts import modules_for as _tmods, envelope as _tenv, emit_modules
    import engines as _eng
    _bay_engines = (
        ("bay.engine.turbine", "agt1500-abrams-turbine",
         (-base_half_x * 0.40, floor_y + 0.42, -base_half_z * 0.44),
         dict(part_role="main-power-unit", runs_when="moving-or-charging-hard",
              burns="anything", note="ruinous at idle")),
        ("bay.engine.multifuel", "ldt465-multifuel-deuce",
         (base_half_x * 0.34, floor_y + 0.40, base_half_z * 0.48),
         dict(part_role="auxiliary-power-unit", runs_when="most-of-the-time",
              burns="anything", note="the one that is actually on")),
    )
    g.assembly = "powerplant"
    g.motion_group = "frame"
    _engine_volume = 0.0
    for ident, key, pos, attrs in _bay_engines:
        _e = next(x for x in _eng.CATALOGUE if x.identity == key)
        # A TURBINE IS NOT A BLOCK WITH HOLES IN IT. `EnginePackage`
        # builds its prism from the drivetrain graph, and a turbine's
        # drivetrain graph is a piston engine's block sized from a
        # PLACEHOLDER displacement -- so it returned 0.89 x 0.66 x
        # 0.79 m for an engine that is really 1.63 m long and weighs
        # 1134 kg. Wrong by a factor of three in volume, and "does it
        # fit in the bay" was about to be answered against it.
        _mods = _tmods(_e)
        if _mods:
            _env = _tenv(_mods)
            _sx, _sy, _sz = _env["size_m"]
        else:
            _pk = _Pkg.build(_e)
            _sx, _sy, _sz = _pk.prism.size_m
        _engine_volume += _sx * _sy * _sz
        g.node(ident, list(pos), "load-bearing-structure",
               material="cast-iron", mass_in_total=False,
               mass_kg=float(_e.mass_kg),
               engine_identity=key, engine_label=_e.label,
               package_size_m=(round(_sx, 3), round(_sy, 3), round(_sz, 3)),
               package_volume_m3=round(_sx * _sy * _sz, 3),
               fuels=sorted(_e.fuel_compatibility),
               half_extent_m=(round(_sx / 2, 4), round(_sy / 2, 4),
                              round(_sz / 2, 4)),
               **attrs)
        _near = sorted(range(len(_floor_pts)),
                       key=lambda j: (_floor_pts[j][0] * _fx - pos[0]) ** 2
                       + (_floor_pts[j][1] * _fz - pos[2]) ** 2)[:3]
        for j in _near:
            g.edge(f"{ident.replace('.', '_')}.mount.{j}", ident,
                   f"bay.floor.{j}", "rigid-distance", radius=0.024,
                   palette="chassis-grey",
                   load_path="engine-on-its-bearers-on-the-bay-floor")
    g.engine_volume_m3 = round(_engine_volume, 3)

    # ---- THE HOOKUP WALL ----
    # Every service this site needs, as REAL PORTS on the existing port
    # system rather than a note saying "hookups here". They are line
    # ports, so they read as OPEN until something is plumbed to them,
    # and the mate report says exactly which are connected and which
    # are still hanging -- which is the whole question when a C18
    # rolls up and you need to know what it can actually feed.
    from assembly_ports import PartPort, mate_ports, emit_ports_graph
    _wall_at = np.array([base_half_x * 0.82, bay_y + 0.18, base_half_z * 0.30])
    _services = (
        ("power_3ph", "electrical", 0.030, "electrical", "480 V three phase"),
        ("power_dc", "electrical", 0.018, "electrical", "24 V control and keep-alive"),
        ("hydraulic_supply", "hydraulic", 0.020, "hydraulic", "pressure in"),
        ("hydraulic_return", "hydraulic", 0.025, "hydraulic", "return"),
        ("coolant_supply", "coolant", 0.032, "coolant", "barrel water in"),
        ("coolant_return", "coolant", 0.032, "coolant", "barrel water out"),
        ("air", "air", 0.016, "compressed-air", "purge, tools, recuperator charge"),
        ("fuel", "fuel", 0.020, "fuel", "for the auxiliary plant"),
        ("data", "data", 0.012, "signal", "fire control, sensors, remote"),
    )
    _wall_ports = []
    for i, (nm, kind, r, fluid, note) in enumerate(_services):
        row, col = divmod(i, 3)
        _wall_ports.append(PartPort(
            f"bay.hookup.{nm}", "bay.hookup_wall", kind,
            _wall_at + np.array([0.02, 0.16 - row * 0.16, -0.22 + col * 0.22]),
            np.array([1.0, 0.0, 0.0]), r, False, fluid=fluid,
            joint="bolted-flange"))
    _wall_res = mate_ports(_wall_ports)
    emit_ports_graph(
        _wall_ports, _wall_res,
        lambda ident, pos, kind, **kw: g.node(
            ident, pos, "engine-block-port", material="steel-plate",
            mass_in_total=False, half_extent_m=(0.026, 0.026, 0.026),
            serves="any-heavy-machine-a-c18-first-of-all", **kw),
        lambda ident, a_, b_, constraint, **kw: g.edge(
            ident, a_, b_, constraint, palette="rollbar-silver"),
        prefix="base")
    for i, (nm, *_rest) in enumerate(_services):
        for tgt, rad in (("bay.hookup_wall", 0.014),
                         (f"post.foot.{'pp' if i % 2 else 'pn'}", 0.012),
                         (f"deck.corner.{'pp' if i % 2 else 'pn'}", 0.010)):
            g.edge(f"bay.hookup_stay.{nm}.{tgt.split('.')[-1]}",
                   f"base.bay.hookup.{nm}", tgt, "rigid-distance",
                   radius=rad, palette="chassis-grey",
                   load_path="hookup-boss-carried-by-the-wall-and-the-frame")

    # ---- THE RING, on the deck ----
    # THE ROTATING RACE SITS ON THE RING, both on the deck.
    race_y = deck_y + 0.10
    for name, sx, sz in corners:
        g.motion_group = "traverse"
        g.assembly = "slew-ring"
        g.node(f"turret.race.{name}",
               [sx * ring_radius * 0.72, race_y, sz * ring_radius * 0.72],
               "slew-ring-race-pad", material="hardened-steel")
    for i, (name, sx, sz) in enumerate(corners):
        # THE RACE ROLLS ON THE RING BEAM, at the two ring nodes nearest
        # it -- not on a corner of a carriage that no longer exists.
        near = min(range(RING_N), key=lambda j: (
            (math.cos(2 * math.pi * j / RING_N) - sx * 0.7071) ** 2
            + (math.sin(2 * math.pi * j / RING_N) - sz * 0.7071) ** 2))
        g.edge(f"turret.slew_bearing.{name}", f"deck.ring.{near}",
               f"turret.race.{name}", "gimbal-yaw-bearing", radius=0.040,
               frame_mount=True, rolling_contact=True,
               load_path="rotating-mass-and-recoil-moment-into-the-ring-beam")
        g.edge(f"turret.slew_bearing_b.{name}", f"deck.ring.{(near + 1) % RING_N}",
               f"turret.race.{name}", "gimbal-yaw-bearing", radius=0.034,
               frame_mount=True, rolling_contact=True,
               load_path="second-contact-so-the-pad-cannot-pivot-on-one-point")
    for a, b in ring_pairs:
        g.edge(f"turret.race_rail.{a}_{b}", f"turret.race.{a}", f"turret.race.{b}",
               "rigid-distance", radius=0.028, palette="rollbar-silver",
               load_path="the-rotating-race-ring")
    # THE YAW RING IS A RING. Authored as a point in the middle it was
    # both a ball joint that could carry no moment and a plug in the
    # ejection shaft; it is emitted as eight rim nodes instead, which is
    # what a slew ring physically is.
    g.assembly = "slew-ring"
    for i in range(RING_N):
        a = 2.0 * math.pi * i / RING_N
        g.node(f"turret.yaw.{i}",
               [ring_radius * 0.86 * math.cos(a), race_y,
                ring_radius * 0.86 * math.sin(a)],
               "yaw-clutch-bearing", material="hardened-steel",
               half_extent_m=(0.075, 0.038, 0.075))
    for i in range(RING_N):
        g.edge(f"turret.yaw_rim.{i}", f"turret.yaw.{i}",
               f"turret.yaw.{(i + 1) % RING_N}", "rigid-distance",
               radius=0.030, palette="rollbar-silver",
               load_path="the-yaw-ring-itself")
    for i, (name, sx, sz) in enumerate(corners):
        near = min(range(RING_N), key=lambda j: (
            (math.cos(2 * math.pi * j / RING_N) - sx * 0.7071) ** 2
            + (math.sin(2 * math.pi * j / RING_N) - sz * 0.7071) ** 2))
        g.edge(f"turret.yaw_spoke.{name}", f"turret.yaw.{near}",
               f"turret.race.{name}", "rigid-distance", radius=0.026,
               palette="rollbar-silver",
               load_path="race-pad-tied-into-the-yaw-ring")
    # THE GUN RIDES A METRE ABOVE THE RING IT TURNS ON. Everything the
    # turret builder places from `carriage_y` -- trunnion, cradle, tube,
    # tank -- comes up with it; the race, the floor and the drive are
    # placed from `race_y` and do not.
    return {"corners": corners, "ring_pairs": ring_pairs,
            "carriage_y": deck_y + GUN_RISE_M,
            "race_y": race_y, "floor_y": floor_y, "ring_radius": ring_radius,
            "ring_nodes": RING_N, "legs": legs, "base_half_x": base_half_x,
            "base_half_z": base_half_z, "lift_stroke_m": lift_stroke_m,
            "frame_ring": [f"deck.ring.{i}" for i in range(RING_N)],
            "gun_rise_m": GUN_RISE_M}


def document_owner(document: dict, node: dict) -> str | None:
    """Which motion group a wrench point belongs to: its own, or that of
    the body it is a surface of."""
    if node.get("motion_group"):
        return node["motion_group"]
    parent = node.get("surface_of")
    if parent:
        for n in document["nodes"]:
            if n["identity"] == parent:
                return n.get("motion_group")
    return None


def raise_on_hydraulics(graph, *, dt: float = 0.05,
                        ground_bearing_pa: float = 250_000.0,
                        hpu=None) -> dict:
    """Stand the machine up using the project's OWN hydraulics.

    `outriggers.OutriggerSet` runs four `actuators.LinearActuator`
    cylinders off a `bench.HydraulicPowerUnit`: the pump makes the flow
    its displacement and speed allow, the motor limits that flow at
    pressure, the cylinders move at flow divided by area, and they stop
    at the end of their stroke. The lift that comes out is what the
    hardware achieved -- it takes time, it can be short, and it can
    stall outright if the pack cannot make the pressure.

    Nothing here decides how far the legs go. The hydraulics do."""
    from outriggers import OutriggerSet
    doc = graph.as_document()
    jacks = [e for e in doc["edges"] if e.get("kind") == "outrigger-lift-jack"]
    if not jacks:
        raise ValueError(f"{graph.identity}: no outrigger jacks to raise")
    # A LEG'S TRAVEL IS THE SUM OF ITS STAGES, and they are per leg, so
    # the stages of ONE leg are summed rather than every jack edge in
    # the machine. Taking jacks[0] read a single stage and reported
    # half the lift, which the clearance check then failed on -- the
    # right answer to the wrong question.
    per_leg = {}
    for e in jacks:
        leg = e["identity"].rsplit(".", 2)[1] if e["identity"].count(".") > 2             else e["identity"].rsplit(".", 1)[-1]
        per_leg.setdefault(leg, []).append(e)
    one = max(per_leg.values(), key=len)
    total_stroke = sum(float(e["maximum_rest_length_m"]) for e in one)
    n_stages = len(one)
    j = one[0]
    mass = sum(float(n.get("mass_kg", 0.0)) for n in doc["nodes"])
    pads = [n for n in doc["nodes"] if n.get("lifts_machine")]
    legs = OutriggerSet(
        machine_mass_kg=mass,
        bore_m=float(j["bore_m"]), rod_m=float(j["rod_m"]),
        stroke_m=total_stroke, stages=n_stages,
        leg_names=tuple(sorted(per_leg)),
        pad_area_m2=float(pads[0].get("pad_bearing_area_m2", 0.13)) if pads else 0.13,
        ground_bearing_pa=ground_bearing_pa, hpu=hpu)
    log = legs.raise_fully(dt=dt)
    return {"set": legs, "log": log, "lift_m": log["extension_m"],
            "machine_mass_kg": mass}


def deployed_document(graph, *, extension: float = None,
                      lift_m: float = None) -> dict:
    """The machine standing on its outriggers, jacks out.

    Everything that is not pinned to the ground rises by the jack
    stroke; the pads stay where they are, because they are what it is
    standing on. `extension` is 0 stowed to 1 fully out.

    This is what the stroke is FOR and it is the only view in which
    that can be checked: the clearance under the machine is the lowest
    point of anything it carries, measured against the ground the pads
    are on, and if that number is smaller than a trailer then the legs
    are too short no matter what they were declared to be.
    """
    document = graph.as_document()
    # THE LIFT IS MEASURED, NOT CHOSEN. Pass `lift_m` from
    # `raise_on_hydraulics` -- what the cylinders actually achieved.
    # `extension` is kept only for drawing a mid-stroke pose of a
    # travel that the hydraulics already proved is reachable.
    # THE SUM OF THE STAGES, not the biggest one. Clamping the lift to
    # a single stage's stroke silently halved every deployed pose while
    # the hydraulics were reporting the right answer next to it.
    _by_leg: dict = {}
    for e in document["edges"]:
        if e.get("kind") != "outrigger-lift-jack":
            continue
        leg = e["identity"].rsplit(".", 2)[1] if e["identity"].count(".") > 2             else e["identity"].rsplit(".", 1)[-1]
        _by_leg[leg] = _by_leg.get(leg, 0.0) + float(
            e.get("maximum_rest_length_m", 0.0))
    stroke = max(_by_leg.values(), default=0.0)
    if lift_m is None:
        if extension is None:
            raise ValueError(
                "say how far it is up: `lift_m` from raise_on_hydraulics, "
                "or `extension` as a fraction of a stroke the hydraulics "
                "have been shown to reach")
        lift_m = stroke * float(extension)
    lift = min(float(lift_m), stroke)
    # WHAT STAYS DOWN is the rod and the pad. The rod slides out of the
    # barrel; the pad is on the ground under it. Everything else -- the
    # barrel, the posts, the floor, the stores, the gun -- is what
    # goes up, and it goes up by exactly the rod travel. Lifting every
    # node including the rod would have been a picture of the machine
    # hovering with its legs still folded, which is a moved mesh and
    # not an extended cylinder.
    ground = {n["identity"] for n in document["nodes"]
              if n.get("lifts_machine")
              or str(n.get("motion_group", "")).startswith("outrigger-stage")
              or str(document_owner(document, n) or "").startswith(
                  "outrigger-stage")}
    out = {**document, "nodes": []}
    for n in document["nodes"]:
        if n["identity"] in ground:
            out["nodes"].append(n)
            continue
        p = list(n["reference_position"])
        p[1] += lift
        out["nodes"].append({**n, "reference_position": p})
    out["edges"] = [
        {**e, "rest_length": e.get("rest_length", 0.0) + lift}
        if e.get("kind") == "outrigger-lift-jack" else e
        for e in document["edges"]]
    out["deployed"] = {"extension": lift / stroke if stroke else 0.0,
                       "lift_m": lift,
                       "stayed_down": sorted(ground)}
    return out


def deployed_clearance(graph, *, extension: float = None,
                       lift_m: float = None) -> dict:
    """How much room there actually is under the machine when it is up.

    Measured from the ground the pads stand on to the LOWEST point of
    any body the machine carries, its own half-extent included -- not
    to a node, because a node is the middle of a box and the box hangs
    below it."""
    doc = deployed_document(graph, extension=extension, lift_m=lift_m)
    pads = [n for n in doc["nodes"] if n.get("lifts_machine")]
    ground = min(float(n["reference_position"][1])
                 - float(n.get("body_half_extent_m", (0, 0, 0))[1])
                 for n in pads) if pads else 0.0
    # THE LEG ITSELF IS NOT AN OBSTRUCTION. The rod and the pad are
    # what the machine is standing on; measuring clearance against them
    # would be measuring the machine against its own feet. What has to
    # clear is everything the jacks LIFTED.
    down = set(doc["deployed"]["stayed_down"])
    lowest, who = None, None
    for n in doc["nodes"]:
        if n["identity"] in down:
            continue
        y = (float(n["reference_position"][1])
             - float(n.get("body_half_extent_m", (0.0, 0.0, 0.0))[1]))
        if lowest is None or y < lowest:
            lowest, who = y, n["identity"]
    return {"ground_y": ground, "lowest_y": lowest,
            "lowest_part": who,
            "clearance_m": (lowest - ground) if lowest is not None else 0.0,
            "lift_m": doc["deployed"]["lift_m"]}

def build_pop_up_turret(*, well_half: float = 0.80, well_depth: float = 1.05,
                        hoist_stroke_m: float = 0.850,
                        ring_radius: float = 0.78) -> ProductionGraph:
    """The original single-gun turret head, on the shared well."""
    g = ProductionGraph(identity="pop-up-turret/production")
    base = _build_well_carriage_and_race(g, well_half=well_half, well_depth=well_depth,
                                         hoist_stroke_m=hoist_stroke_m, ring_radius=ring_radius)
    corners, ring_pairs = base["corners"], base["ring_pairs"]
    carriage_y, race_y = base["carriage_y"], base["race_y"]
    floor_y = base["floor_y"]
    # z = 0.62 is the barrel's own station -- putting the trunnion
    # there, not at z = 0, is the whole point: a pivot AT the recoiling
    # mass's centre of gravity carries (almost) no weight moment at
    # all, which is how real tank and naval mounts are laid out and
    # why their receivers carry a counterweight bustle when the tube
    # itself cannot be centred on the trunnion.
    TRUNNION_Z = 0.62
    # the trunnion turns with the mount but does not elevate
    g.motion_group = "traverse"
    g.assembly = "gun-gimbal"
    g.node("turret.pitch", [0.0, carriage_y + 0.42, TRUNNION_Z], "pitch-bearing",
           material="hardened-steel")
    # the tube and its cradle are what actually elevate
    g.motion_group = "elevation"
    g.assembly = "gun-tube"
    g.node("turret.weapon", [0.0, carriage_y + 0.44, 0.62], "recoiling-weapon-mass",
           mass_in_total=False, material="gun-steel", shape="drum",
           # L/70 means seventy calibres: 40 mm x 70 = 2.8 m of tube.
           # At 1.3 m it was not a Bofors barrel, it was a stub, and it
           # weighed a third of what one does.
           drum_axis=(0.0, 0.0, 1.0), drum_radius_m=0.055, drum_length_m=2.80,
           half_extent_m=(0.055, 0.055, 1.40))
    g.edge("turret.mount", "carriage.hub", "turret.yaw", "actuated-damped-clutch-gimbal-base",
           radius=0.060, frame_mount=True,
           load_path="turret-wrench-through-carriage-to-hoist-and-guides",
           clutch_engagement=1.0, angular_stiffness_nm_per_rad=92_000.0,
           angular_damping_nm_s_per_rad=7_400.0, holding_torque_nm=24_000.0,
           release_behavior="bounded-free-gimbal-after-commanded-clutch-release")
    # the traverse drive itself: torque between the carriage (stationary
    # side) and the race ring (rotating side), which is where a real
    # rack-and-pinion or slew drive actually pushes
    g.edge("turret.traverse_drive", "carriage.hub", "turret.yaw",
           "gimbal-yaw-bearing", radius=0.052, driven_by="turret.traverse_actuator",
           drive_torque_nm=15_890.0, frame_mount=True,
           load_path="slew-drive-pinion-reaction-into-carriage")
    # THE TRUNNION STANDARD. The pitch bearing has to be carried by the
    # rotating body, and carried in a way that resists the firing moment
    # -- so a pedestal up the axis plus two braces out to the race ring,
    # which is the triangle a real trunnion standard makes. Without
    # these the pitch node hangs off one member and the structural check
    # says so.
    g.edge("turret.pedestal", "turret.yaw", "turret.pitch", "rigid-distance",
           radius=0.048, palette="rollbar-silver",
           load_path="firing-moment-and-elevation-reaction-into-the-race-ring")
    for name in ("nn", "pp"):
        g.edge(f"turret.trunnion_brace.{name}", f"turret.race.{name}", "turret.pitch",
               "rigid-distance", radius=0.024, palette="rollbar-silver")
    # ---- THE ELEVATION LINKAGE ----
    # The pitch bearing is a hinge; a hinge does not move anything. What
    # elevates the gun is a ram pushing on a LUG that is offset from the
    # trunnion axis, because a force through the pivot makes no moment
    # at all. The offset IS the mechanism.
    #
    # Two more real details that a bare cylinder would miss:
    #
    #   THE MOMENT ARM CHANGES THROUGH THE TRAVEL. The ram is a straight
    #   line and the lug swings on an arc, so the perpendicular distance
    #   between them varies -- strongest near the middle of the arc and
    #   weakest at the ends. That is why elevation limits are set where
    #   they are, and why a gun is slowest near its stops.
    #
    #   THE EQUILIBRATOR carries the barrel's weight moment so the ram
    #   does not have to. A 171 kg tube on a 0.6 m arm is about a
    #   kilonewton-metre that never goes away, and sizing the ram to
    #   fight it continuously rather than balancing it first is how you
    #   get a mount that cannot hold elevation when the pump stops.
    cradle_y = carriage_y + 0.44
    g.assembly = "cradle"
    g.node("turret.cradle", [0.0, cradle_y, TRUNNION_Z + 0.18], "load-bearing-structure",
           material="steel-plate", half_extent_m=(0.13, 0.10, 0.30))
    g.node("turret.elevation_lug", [0.0, cradle_y - 0.26, TRUNNION_Z - 0.14], "chassis-load-node",
           material="steel-plate", half_extent_m=(0.05, 0.05, 0.05))
    # WHERE THE ANCHOR GOES DECIDES THE MOMENT ARM, and the first
    # placement got it wrong in the classic way: behind and below the
    # lug, so the ram pointed nearly AT the trunnion and its line passed
    # within 90 mm of the pivot. A force through a pivot makes no
    # moment, and a force nearly through it makes very little -- it
    # needed 11.6 kN to hold a barrel that weighs 1.7 kN.
    #
    # The arm is the perpendicular distance from the trunnion to the
    # ram's LINE, so it is largest when the ram runs perpendicular to
    # the lug's radius. Putting the anchor below and FORWARD does that,
    # and the same barrel then needs about a third of the force.
    # the ram's lower anchor is on the mount, not on the gun
    g.motion_group = "traverse"
    g.assembly = "cradle"
    g.node("turret.elevation_anchor", [0.0, cradle_y - 0.47, TRUNNION_Z + 0.22], "chassis-load-node",
           material="steel-plate", half_extent_m=(0.06, 0.06, 0.06))

    # THE CRADLE IS A SLEEVE, NOT A WELD. The barrel does not ride
    # rigidly on the cradle -- it slides through it along the bore
    # axis, on the same real hardware every recoil-operated mount uses:
    # machined ways in the cradle, the tube (or slide) riding in them,
    # a recoil BRAKE metering the way back and a RECUPERATOR pulling it
    # back to battery. That is exactly what GasOverOilStrut already is;
    # the fix is wiring it as the joint the barrel actually moves
    # through, instead of a lumped point coupling that let none of the
    # recoil impulse show up as real travel. See "turret.recoil_slide"
    # below, and `oleo_recoil_slide` in segment_types.py for the type
    # this uses when authored through the editor's own palette.
    # the lug hangs off the cradle, below and behind the trunnion
    g.edge("turret.lug_to_cradle", "turret.elevation_lug", "turret.cradle", "rigid-distance",
           radius=0.026, palette="rollbar-silver")
    g.edge("turret.lug_brace", "turret.elevation_lug", "turret.pitch", "rigid-distance",
           radius=0.022, palette="rollbar-silver")
    # the anchor is part of the rotating body
    g.edge("turret.anchor_to_yaw", "turret.elevation_anchor", "turret.yaw", "rigid-distance",
           radius=0.030, palette="rollbar-silver")
    g.edge("turret.anchor_brace", "turret.elevation_anchor", "turret.race.pn", "rigid-distance",
           radius=0.022, palette="rollbar-silver")

    # AND THE RAM ITSELF
    g.edge("turret.elevation_ram", "turret.elevation_anchor", "turret.elevation_lug",
           "linear-hydraulic-actuator", radius=0.032, palette="actuator-yellow",
           frame_mount=True, family="linear-hydraulic-actuator",
           kind="commanded-rest-length-elevation-ram",
           state=["commanded_rest_length_m", "relief_stroke_m", "axial_force_n",
                  "dissipated_energy_j", "temperature_k", "health"],
           commanded_rest_length_m=0.0, minimum_rest_length_m=0.0,
           maximum_rest_length_m=0.320, bore_m=0.063, rod_m=0.036,
           linear_stiffness_n_per_m=9.0e6, linear_damping_n_s_per_m=3.2e4,
           holding_force_n=48_000.0, relief_force_n=70_000.0,
           maximum_relief_stroke_m=0.012,
           elevation_range_deg=[-8.0, 42.0],
           energy_law="commanded-length-plus-series-relief; the equilibrator carries "
                      "the barrel moment so this carries only the difference")
    # the equilibrator: a gas spring balancing the barrel's own weight
    g.edge("turret.equilibrator", "turret.elevation_anchor", "turret.cradle",
           "spring-damper", radius=0.026, palette="actuator-yellow",
           stiffness_n_per_m=46_000.0, preload_force_n=3_200.0,
           compression_damping_n_s_per_m=1_100.0, rebound_damping_n_s_per_m=1_500.0,
           load_path="balances-the-barrel-weight-moment-so-the-ram-carries-the-difference")

    g.edge("turret.gimbal.pitch", "turret.pitch", "turret.cradle", "gimbal-pitch-bearing",
           radius=0.045, driven_by="turret.elevation_ram",
           elevation_range_deg=[-8.0, 42.0],
           load_path="trunnion-pin-carries-the-cradle-not-the-bare-tube")
    # THE RECOIL SLIDE. A real prismatic joint along the bore axis: the
    # gas spring returns the tube to battery, the orifice brake meters
    # its return, and the STROKE here is generous on purpose -- a long
    # travel is what buys margin to absorb a round that lands before
    # the previous one has fully recovered, which is the real behaviour
    # of any gun run near its cyclic rate. Runtime state
    # (compression/velocity) is carried by `actuators.GasOverOilStrut`
    # and stepped every tick by `machines.step_recoil_slide` -- this
    # edge only authors the hardware.
    g.edge("turret.recoil_slide", "turret.weapon", "turret.cradle", "oleo-recoil-slide",
           radius=0.036, palette="actuator-yellow",
           slide_axis=(0.0, 0.0, 1.0), recoiling_mass_kg=171.3, recoil_stroke_m=0.420,
           gas_charge_pressure_pa=5.0e5, gas_volume_m3=0.0085,
           orifice_area_m2=math.pi * 0.0215 ** 2 / 4 * 0.7,
           load_path="tube-rides-in-the-cradle-ways-resisted-by-brake-and-recuperator")
    # the discrete SHOT EVENT still gets the production graph's own
    # r-cross-impulse load path, in parallel with the slide that
    # actually absorbs it -- this is where the impulse is applied, the
    # slide above is what happens to it
    g.edge("turret.recoil", "turret.weapon", "turret.cradle", "point-impulse-wrench-coupling",
           radius=0.020, load_path="individual-shot-recoil-r-cross-impulse-into-the-slide",
           impulse_n_s=1348.3)

    # ---- and the fluid that drives it, as routed lines ----
    # the plant rides up with the carriage but does not traverse
    g.motion_group = "carriage"
    g.assembly = "hydraulic-plant"
    g.node("turret.manifold", [0.42, floor_y + 0.18, 0.30], "manifold",
           fixed_to="world", material="steel-plate",
           half_extent_m=(0.07, 0.07, 0.10))
    g.node("turret.accumulator", [0.58, floor_y + 0.45, 0.52], "high-pressure-canister",
           fixed_to="world", capacity_kg=17.4, material="pressed-steel",
           shape="drum", drum_axis=(0.0, 1.0, 0.0), drum_radius_m=0.125,
           drum_length_m=0.80, half_extent_m=(0.125, 0.40, 0.125))
    g.edge("turret.line.manifold_to_ram", "turret.manifold", "hoist.base",
           "pressure-rated-hydraulic-line", radius=0.014, palette="hydraulic-red")
    g.edge("turret.line.accumulator_to_manifold", "turret.accumulator", "turret.manifold",
           "pressure-rated-hydraulic-line", radius=0.014, palette="hydraulic-red")
    g.edge("turret.line.manifold_to_yaw", "turret.manifold", "carriage.hub",
           "flexible-hydraulic-hose", radius=0.008, palette="hydraulic-red")
    return g


# =====================================================================
#  THE THREE-MECHANISM CANNON, PLUS ITS OWN MACHINE GUN
# =====================================================================
def elevating_centre_of_gravity(document: dict) -> tuple:
    """The mass that swings on the trunnion, and where it balances.

    WRENCH POINTS ARE NOT BODIES. A prism's ports carry a nominal mass
    so they draw and can be hit, and counting them puts a hundred and
    nineteen kilogrammes of proxy into the sum. They declare themselves
    with `wrench_point`, so the exclusion is a declaration and not a
    name test.

    `mass_in_total` is NOT the test either, whatever it looks like. It
    means "already counted somewhere else in a vehicle total" -- the
    outer barrel carries it, and the outer barrel is 647 kg of real
    steel. Filtering on it reported this machine's recoiling mass as
    27.6 kg."""
    import numpy as _np
    mass = 0.0
    moment = _np.zeros(3)
    for n in document["nodes"]:
        if n.get("wrench_point"):
            continue
        if str(n.get("motion_group")) not in ("fine", "elevation", "carriage"):
            continue
        w = float(n.get("mass_kg", 0.0))
        mass += w
        moment += w * _np.asarray(n["reference_position"], dtype=float)
    return mass, (moment / mass if mass else moment)


def balanced_station(*, tolerance_m: float = 0.002, passes: int = 8,
                     **kwargs) -> ProductionGraph:
    """The station with its trunnion on the real balance point.

    Builds, weighs what it built, and builds again with the answer --
    and then AGAIN, because two passes is not enough and assuming it was
    left 0.19 m of arm on the table.

    The reason is that the trunnion is not a passenger in the sum. The
    collar sits on it, the cradle is placed from it, the elevation lug
    and the equilibrator anchor off it: move the pin and a few hundred
    kilogrammes of the thing being balanced moves with it, so the centre
    of gravity chases the pin. It is a fixed-point problem, not a
    correction, and it is solved by iterating it until the arm stops
    moving rather than by asserting that one step was sufficient."""
    kwargs.pop("trunnion_z", None)
    graph = build_gimbal_cannon_station(**kwargs)
    for _ in range(passes):
        _mass, centre = elevating_centre_of_gravity(graph.as_document())
        target = float(centre[2])
        here = next(n["reference_position"][2] for n in graph.nodes
                    if n["identity"] == "turret.pitch")
        if abs(target - here) <= tolerance_m:
            break
        graph = build_gimbal_cannon_station(trunnion_z=target, **kwargs)
    return graph


def build_gimbal_cannon_station(*, bore_mm: float = 40.0, well_half: float = 0.80,
                                well_depth: float = 1.05, hoist_stroke_m: float = 0.850,
                                ring_radius: float = 1.62,
                                traverse_upper_ring: bool = True,
                                traverse_upper_ring_reference: str = "same-body",
                                traverse_upper_ring_tooth_delta: int = 0,
                                recoil_peak_force_n: float = 320_000.0,
                                standard_breech_bore_mm: float = 76.0,
                                firing_lock: bool = False,
                                preload_margin: float = 1.4,
                                firing_lock_friction: float = 0.15,
                                firing_lock_pressure_pa: float = 40.0e6,
                                firing_lock_margin: float = 1.5,
                                trunnion_z: float | None = None
                                ) -> ProductionGraph:
    """One cannon aimed by three real mechanisms in series, plus an
    independently-sighted machine gun on the same carriage.

    COARSE, COARSE, FINE -- and each stage does the job the one before
    it cannot:

      THE RING carries several pinions on one gear (slew_drive.py's own
      hardware), so yaw comes from whichever power sources are actually
      running -- electric alone for a quiet watch, everything engaged
      for a fast engagement, and it keeps turning if one pinion fails.
      This is the coarse, continuous axis: it can turn all the way
      around and it is not fast.

      THE TRUNNION is the pitch axis, and it sits at the barrel's own
      centre of gravity for whatever bore is chosen (`calibres.cannon`
      derives the tube's mass and length from the bore itself, so this
      re-balances automatically rather than needing hand-tuning per
      calibre). Coarse, and also not fast -- a barrel this heavy is not
      meant to be flicked around on its trunnion pins.

      THE SIX-ACTUATOR PLATFORM at the breech is the FINE stage, and it
      is deliberately the smallest, fastest, shortest-travel mechanism
      of the three. Six short hydraulic legs in the same Stewart
      geometry `armature.py` already proves converges in single digits
      of iterations: this is what gives "barrel twitch" -- correcting
      the last fraction of a degree while the ring and trunnion are
      still catching up, or riding through a hit that jars the coarse
      stages without losing the sight picture entirely. The recoil
      slide is carried ON this platform, not on the cradle directly, so
      recoil travel happens relative to the fine stage exactly the way
      it would on the real gun.

    THE MACHINE GUN gets its OWN two-axis gimbal on the same carriage,
    structurally independent of the cannon's ring -- its own hemisphere
    of aim, so it can watch a completely different arc while the main
    gun is committed to a target, and losing the cannon's drive does
    not touch it at all.
    """
    from calibres import cannon as _cannon_calibre
    from ballistics import mass_of as _mass_of
    from actuators import size_recoil_slide as _size_recoil_slide

    g = ProductionGraph(identity="gimbal-cannon-station/production")
    # THIS STATION DOES NOT POP UP. It stands on four square steel posts
    # with an outrigger jack at each corner, and those jacks exist to
    # lift the whole machine high enough to run a trailer or a chassis
    # underneath it. The hoist it used to have put a 125 mm ram straight
    # up the middle -- through the case ejection column, which is the
    # one volume in this machine that has to stay empty.
    base = _build_outrigger_base_and_race(
        # THE BASE IS SIZED BY WHAT GOES IN IT, and what goes in it is
        # 4.8 m3 of engines, ammunition and stores in a bay that was
        # 4.90 m3 gross -- 98% packed, with no room to reach anything
        # and not a litre of water. Machinery wants about 40% free
        # volume for access, ducting and services.
        #
        # THE FOOTPRINT IS TWO STANDARD MEASURES, so it loads onto
        # ordinary equipment and sits in ordinary spaces:
        #
        #   WIDTH   two ISO container widths, 2 x 2.438 = 4.876 m.
        #           Deliberately wide. A gun platform's enemy is the
        #           overturning moment a shot puts into it, and track
        #           width is the only cheap way to resist it -- the
        #           righting moment is weight times half the track, so
        #           doubling the width doubles it for nothing.
        #   LENGTH  a flatbed's between-axles deck section, 3.66 m.
        #           Not a whole container: the length is where a base
        #           gets heavy and awkward without buying stability.
        #
        # It travels as an oversize load on a low-bed, which is what
        # the outrigger jacks are for -- they pick it up and the
        # trailer reverses underneath.
        g, base_half_x=2.438, base_half_z=1.830,
        # AND THE STROKE GROWS WITH THE BAY. A deeper base hangs lower,
        # so the same 1300 mm of leg no longer buys 1050 mm of room
        # underneath -- it bought 881 mm and the clearance check said
        # so. The stroke is tied to the depth it has to lift clear of,
        # rather than being a constant that silently stops being
        # enough the moment the base changes.
        bay_depth=well_depth * 1.28,
        lift_stroke_m=well_depth * 1.28 + 0.34,
        ring_radius=ring_radius)
    RING_N = base["ring_nodes"]

    def _ring_near(x, z, offset=0):
        """The ring index nearest a point, so a member reaches the
        structure beside it instead of a hub in the middle."""
        best = min(range(RING_N), key=lambda j: (
            (ring_radius * math.cos(2.0 * math.pi * j / RING_N) - x) ** 2
            + (ring_radius * math.sin(2.0 * math.pi * j / RING_N) - z) ** 2))
        return (best + offset) % RING_N
    corners, ring_pairs = base["corners"], base["ring_pairs"]
    carriage_y, race_y = base["carriage_y"], base["race_y"]

    # the yaw ring and its spokes come from the shared base -- authoring
    # them again here is what produced two "turret.yaw" bodies at the
    # same place, and therefore twice its mass, until node() was taught
    # to refuse a repeated identity.
    # THE TRAVERSE CLUTCH GOES AROUND THE RING, not up the middle. One
    # pack per ring station is what a slew drive with a brake actually
    # is; it can carry the overturning moment a shot puts into it,
    # where a single central member is a ball joint and carries none;
    # and it leaves the ejection shaft clear.
    for _i in range(RING_N):
        g.edge(f"turret.mount.{_i}", f"deck.ring.{_i}", f"turret.yaw.{_i}",
               "actuated-damped-clutch-gimbal-base",
               radius=0.034, frame_mount=True,
               load_path="turret-wrench-through-the-ring-beam-into-the-posts",
               clutch_engagement=1.0,
               angular_stiffness_nm_per_rad=92_000.0 / RING_N,
               angular_damping_nm_s_per_rad=7_400.0 / RING_N,
               holding_torque_nm=24_000.0 / RING_N,
               release_behavior="bounded-free-gimbal-after-commanded-clutch-release")

    # ---- MECHANISM 1: THE RING, MULTIPLE PINIONS ----
    # Three drives on the one ring gear -- electric, hydraulic, and
    # pneumatic, exactly slew_drive.py's own mixed set -- authored as
    # three parallel gimbal-yaw-bearing members between the stationary
    # carriage and the rotating race, each carrying which technology
    # drives it. The physics of load-sharing, redundancy and mode
    # selection (silent / precision / hazardous-atmosphere / urgent)
    # lives in `slew_drive.SlewDrive`; this is the hardware it drives.
    # THE PINIONS ARE REAL BODIES ON A REAL RING, not an abstract member
    # between two hubs. `author_captive_pinion_ring` emits the pinion,
    # its carrier, its bearing and its mesh(es), and takes the whole
    # arrangement as parameters -- so the same call with
    # `traverse_upper_ring=True` gives the captive sandwich, and with
    # `traverse_upper_ring_reference="frame"` gives the differential
    # planetary instead. See `slew_drive.CaptivePinionRing` for what
    # each of those choices actually costs and buys.
    traverse = author_captive_pinion_ring(
        g, prefix="turret.traverse",
        lower_ring_node=lambda x, z: f"turret.yaw.{_ring_near(x, z)}",
        # BRACED TO THE RING BESIDE IT. Tied to one far corner of a base
        # this wide, a pinion carrier's reaction brace ran right across
        # the machine and clipped the case shaft on the way.
        frame_node=lambda x, z: f"deck.ring.{_ring_near(x, z)}",
        ring_radius_m=ring_radius, ring_y=race_y,
        pinions=[{"identity": "elec-a", "technology": "electric",
                  "drive_torque_nm": 3_800.0, "teeth": 18},
                 {"identity": "elec-b", "technology": "electric",
                  "drive_torque_nm": 3_800.0, "teeth": 18},
                 {"identity": "hyd-a", "technology": "hydraulic",
                  "drive_torque_nm": 9_200.0, "teeth": 20}],
        upper_ring=traverse_upper_ring,
        upper_ring_reference=traverse_upper_ring_reference,
        upper_ring_tooth_delta=traverse_upper_ring_tooth_delta,
        plate_gap_m=0.075)

    # ---- MECHANISM 2: THE TRUNNION, AT THE BARREL'S OWN C.O.G. ----
    cal = _cannon_calibre(bore_mm)
    # ---- THE OUTER BARREL IS ONE TUBE, AND IT DOES NOT CHANGE ----
    # This was scaling with whatever liner was fitted -- 1.40 m for a
    # 20 mm, 8.40 m for a 120 -- which is the exact opposite of the
    # design. The whole point of a 120 mm outer barrel is that it is
    # the FIXED part: every smaller calibre goes down the same tube,
    # full length, held concentric by the washers. A liner that stops
    # a fifth of the way along leaves the shot crossing four metres of
    # 120 mm bore with nothing centring it, which is the case the
    # ballistics already showed destroys the round.
    #
    # So the tube is sized once, for the largest ordnance the mount
    # takes, and every liner runs its whole length. A 20 mm liner in it
    # is very long and that is the intent: lined the whole way the same
    # round makes 1005 m/s instead of 813.
    OUTER_BARREL_LENGTH_M = 8.40
    barrel_length_m = OUTER_BARREL_LENGTH_M
    barrel_radius_m = max(0.025, bore_mm / 1000.0 * 0.72)
    barrel_volume_m3 = math.pi * barrel_radius_m ** 2 * barrel_length_m
    barrel_mass_kg = _mass_of("gun-steel", barrel_volume_m3, hollow_fraction=0.18)
    # ---- THE STANDARD BREECH, AND WHAT ADAPTS A BARREL TO IT ----
    # ONE BREECH, SIZED FOR THE LARGEST ROUND THE FAMILY WILL EVER
    # FIRE, AND NEVER FOR THE BARREL THAT HAPPENS TO BE FITTED. That is
    # the whole return: one loading interface, one autoloader, one set
    # of structure, and the tube becomes the only thing that varies.
    # NATO's 120 mm works this way (L/44 and L/55 are tube swaps on one
    # breech), so do the 5"/54 to 5"/62 naval refits and the 155 mm /39
    # to /52 upgrades, and so does the Bushmaster family end to end.
    #
    # The breech ring's size is set by the CASE HEAD, not by the bore,
    # which is why a standard breech can fire anything smaller: a
    # sub-calibre round in a sabot has the same head and a smaller
    # payload. A 120 mm gun firing APFSDS is already doing exactly
    # this -- pushing a 27 mm penetrator down a 120 mm tube.
    breech_cal_geom = _cannon_calibre(standard_breech_bore_mm)
    breech_bore_m = standard_breech_bore_mm / 1000.0
    # a breech ring is mostly solid steel around a chamber a little
    # larger than the bore, and it is long enough to hold the block,
    # the obturator and the extractor
    breech_radius_m = breech_bore_m * 2.6
    breech_length_m = max(0.35, breech_bore_m * 7.0)
    breech_mass_kg = _mass_of("gun-steel",
                              math.pi * breech_radius_m ** 2 * breech_length_m,
                              hollow_fraction=0.42)
    # WHERE THE BREECH SITS IS THE POINT. Behind the trunnion, inside
    # the turret, over the fine platform -- so that reloading happens
    # in cover instead of out in the world behind a tube sticking two
    # metres past the back of the mount, which is what this geometry
    # did before the breech existed.
    BREECH_Z = 0.10
    barrel_start_z = BREECH_Z + breech_length_m / 2.0
    barrel_centre_z = barrel_start_z + barrel_length_m / 2.0

    # THE COLLAR ADAPTER. Every barrel terminates at the same interface,
    # so the difference between this tube and the standard one is taken
    # up in a collar: bore out a thinner wall for a fat barrel, leave
    # more steel for a thin one.
    #
    # It is tempting to call this an interface and a balance weight in
    # one part, and measuring it says otherwise, so it is written down
    # here instead: across the whole family the collar runs about 93 to
    # 101 kg. The bore is small against the interface diameter, so
    # squaring them leaves almost nothing to vary. The collar is an
    # INTERFACE and nothing else -- THE BREECH is the counterweight, and
    # it is the breech's fixed mass behind the pin that actually
    # balances the tube in front of it.
    collar_outer_m = breech_radius_m * 0.92
    collar_length_m = breech_bore_m * 2.2
    collar_mass_kg = _mass_of("hardened-steel",
                              math.pi * (collar_outer_m ** 2 - barrel_radius_m ** 2)
                              * collar_length_m, hollow_fraction=0.25)

    # THE TRUNNION GOES TO THE CENTRE OF GRAVITY OF THE WHOLE RECOILING
    # ASSEMBLY, not of the tube. That is the better invariant, and the
    # breech is what makes it reachable: a heavy block behind the pin
    # balances a long tube in front of it, so the pivot lands between
    # them instead of out along the barrel. Real mounts carry a
    # counterweight bustle for precisely this, and here the counterweight
    # is not dead mass -- it is the breech, doing its own job.
    #
    # A collar sitting ON the trunnion cannot move the trunnion, so it
    # drops out of the balance and the pin lands at the mass-weighted
    # mean of the tube and the block.
    #
    # EXCEPT THAT THE TUBE AND THE BLOCK ARE NOT THE ELEVATING MASS.
    # They are the two heaviest parts of it and they were the only two
    # in this sum, which put the pin at z=1.276 while the thing it is
    # supposed to balance -- tube, breech, spine, jacket, evacuator,
    # rail and absorber, 2094 kg of it -- has its centre of gravity at
    # z=3.313. Two metres of lever, 41.8 kN.m of dead moment held by the
    # elevation ram and the equilibrator every second the machine is
    # switched on, and nothing said a word because the sum agreed with
    # itself.
    #
    # The balance point is a property of the ASSEMBLED machine, so it
    # takes an assembled machine to find. `trunnion_z` carries it: left
    # None the builder uses this two-part estimate, and
    # `balanced_station()` builds once, weighs what it built, and builds
    # again with the answer. A number that can only be known after the
    # fact is passed in after the fact rather than guessed at up front.
    TRUNNION_Z = (float(trunnion_z) if trunnion_z is not None else
                  (barrel_mass_kg * barrel_centre_z + breech_mass_kg * BREECH_Z)
                  / max(barrel_mass_kg + breech_mass_kg, 1e-9))
    g.motion_group = "traverse"
    g.assembly = "gun-gimbal"
    g.node("turret.pitch", [0.0, carriage_y + 0.42, TRUNNION_Z], "pitch-bearing",
           material="hardened-steel")
    g.motion_group = "elevation"
    # THE TUBE IS NOT PART OF THE ELEVATING MASS. It rides the fine
    # platform through the recoil slide, so it moves with the six legs'
    # twitch AND slides along its own bore -- both declared here rather
    # than inferred, so posing the mesh needs to recognise nothing.
    gun_y = carriage_y + 0.44
    g.assembly = "gun-tube"
    # A STRUCTURAL PORT, DECLARED ON THE FACE IT BELONGS TO.
    # Where the fine platform bolts is not a number to be chosen twice.
    # The breech says where it can be picked up -- an annular face at its
    # REAR end, looking aft, with its middle left open -- and the
    # platform is authored against that. Same idea as the fluid ports in
    # `assembly_ports`: the part carries its own mounting faces, and
    # anything that bolts to it asks rather than guesses.
    #
    # ANNULAR, because the blast panel vents rearward through the centre
    # of that same face. A solid plate there would be a platform bolted
    # across the one path the over-recoil energy is supposed to take.
    BREECH_REAR_Z = BREECH_Z - breech_length_m / 2.0
    # ---- THE BREECH HAS A MACHINED FLAT BOTTOM ----
    # A gun breech is not a bare cylinder. It has a flat machined under
    # its belly because that is where it is held, and a clear opening
    # through that flat because that is where spent cases go. Feed is
    # from the TOP, ejection is DOWN and out through the mounting
    # plate, and those two facts decide the whole layout of everything
    # under the gun: the plate needs a hole in it, and nothing may be
    # bolted across that hole.
    BREECH_FLAT_HALF_W = breech_radius_m * 0.62     # the chord the flat cuts
    BREECH_FLAT_Y = gun_y - breech_radius_m          # the flat itself
    CASE_CHUTE_HALF_W = breech_bore_m * 0.75         # a spent case falls through
    CASE_CHUTE_AFT_Z = BREECH_REAR_Z + 0.045
    CASE_CHUTE_FWD_Z = BREECH_REAR_Z + breech_length_m * 0.40
    # AND THE BREECH DECLARES THE COLUMN THAT MUST STAY EMPTY. It owns
    # the requirement because it is the thing doing the ejecting; every
    # other assembly has to route around it, and `check()` now says so
    # rather than leaving it to whoever reads the comment.
    g.clear_volumes.append({
        "identity": "turret.case_ejection_column",
        "owner": "turret.breech",
        # down to the mouth of the bin, which is where it ends: below
        # that is the bin's own inside and things are supposed to be
        # in there
        "min": (-CASE_CHUTE_HALF_W, base["floor_y"] + 0.52, CASE_CHUTE_AFT_Z),
        "max": (CASE_CHUTE_HALF_W, BREECH_FLAT_Y, CASE_CHUTE_FWD_Z),
        "note": "spent cases fall from the breech flat, through the "
                "mounting plate, and out under the mount -- feed is "
                "from the top, so this is the only way out",
    })
    g.assembly = "breech"
    g.node("turret.breech", [0.0, gun_y, BREECH_Z], "standard-breech",
           motion_group="fine", recoils=True,
           flat_bottom_y_m=round(BREECH_FLAT_Y, 4),
           flat_bottom_half_width_m=round(BREECH_FLAT_HALF_W, 4),
           case_chute_half_width_m=round(CASE_CHUTE_HALF_W, 4),
           case_chute_z_m=(round(CASE_CHUTE_AFT_Z, 4), round(CASE_CHUTE_FWD_Z, 4)),
           feed_from="top", eject_to="down-through-the-mounting-plate",
           mass_kg=round(breech_mass_kg, 1), mass_in_total=False,
           material="gun-steel", shape="drum",
           drum_axis=(0.0, 0.0, 1.0), drum_radius_m=breech_radius_m,
           drum_length_m=breech_length_m,
           breech_bore_mm=standard_breech_bore_mm,
           chamber_radius_m=round(breech_bore_m * 0.62, 4),
           accepts_bore_mm_up_to=standard_breech_bore_mm,
           sub_calibre_by_sabot=True,
           half_extent_m=(breech_radius_m, breech_radius_m, breech_length_m / 2.0))
    g.assembly = "gun-tube"
    g.node("turret.collar", [0.0, gun_y, TRUNNION_Z], "barrel-collar-adapter",
           motion_group="fine", recoils=True,
           mass_kg=round(collar_mass_kg, 1), mass_in_total=False,
           material="hardened-steel", shape="drum",
           drum_axis=(0.0, 0.0, 1.0), drum_radius_m=collar_outer_m,
           drum_length_m=collar_length_m,
           adapts_bore_mm=bore_mm, to_interface_radius_m=round(collar_outer_m, 4),
           half_extent_m=(collar_outer_m, collar_outer_m, collar_length_m / 2.0))
    # THE BLAST PANEL. The over-recoil backstop is a chosen failure
    # path: if the tube ever comes back further than the slide can
    # absorb, the energy leaves through a frangible panel at the rear of
    # the breech instead of through the mount. Blowout panels are
    # standard practice (an Abrams' ammunition compartment has them) and
    # they are far easier to justify here than on a crewed vehicle,
    # because this turret has nobody inside it to vent into.
    g.assembly = "breech"
    g.node("turret.blast_panel", [0.0, gun_y, BREECH_Z - breech_length_m / 2.0 - 0.05],
           "frangible-blast-panel", motion_group="fine", recoils=True,
           mass_kg=14.0, mass_in_total=False, material="steel-plate",
           structural_participation=False,
           solver_condensed_into="turret.breech", solver_condensed_mass=True,
           condensed_while="frangible-seats-are-latched",
           half_extent_m=(breech_radius_m * 0.8, breech_radius_m * 0.8, 0.035),
           vents="rearward", release_pressure_pa=3.5e6,
           opens_on="over-recoil-past-the-backstop")
    g.assembly = "gun-tube"
    # ================= THE CONCENTRIC BARREL =================
    # THE OUTER BARREL IS THE FIXED PART OF THIS MOUNT. Its bore is 120
    # mm, which is the largest ordnance this station will ever carry,
    # and it never changes. Every smaller barrel the mount can shoot is
    # an INNER BARREL that goes down inside it and is held concentric by
    # screw-in washers along its length. Changing calibre is changing
    # the inner barrel and its washers; the outer barrel, the collar
    # that clamps it, the cradle and the recoil gear are all untouched.
    #
    # THE WASHERS ARE THE COOLING SYSTEM. They thread into the outer
    # barrel, they are chambered rather than solid, and the space they
    # leave between the two tubes is a water jacket running the whole
    # length of the gun. The smaller the inner barrel, the bigger that
    # annulus and the more water it takes -- a 20 mm liner in a 120 mm
    # outer barrel has a 38 mm annulus around it, which is a serious
    # amount of coolant right against the hottest metal on the machine.
    # That is what lets a C18 carry this gun's heat away instead of the
    # barrel standing there radiating it.
    #
    # A solid washer would be a dam. These are chambered: the open
    # fraction is declared, and it is what the flow is computed against.
    from prism_bodies import Prism, emit_prism
    OUTER_BARREL_BORE_M = 0.120                 # the max ordnance size
    MAX_ORDNANCE_MM = 120.0
    assert bore_mm <= MAX_ORDNANCE_MM, (
        f"a {bore_mm:.0f} mm barrel does not go down a "
        f"{MAX_ORDNANCE_MM:.0f} mm outer barrel")
    # gun-tube wall as a fraction of bore: a 76 mm tube is about 20 mm
    # in the wall, not 26 -- and the difference decides whether it
    # sleeves into the 120 at all
    liner_wall_m = max(0.005, bore_mm / 1000.0 * 0.22)
    liner_outer_r = bore_mm / 2000.0 + liner_wall_m
    # AT MAX ORDNANCE THERE IS NO INNER BARREL. A 120 mm round is fired
    # out of the 120 mm tube itself -- there is nothing left to sleeve
    # it with, and that is what "maximum ordnance size" means. The
    # water jacket goes with it: no annulus, no coolant, and the gun is
    # thermally limited instead, which is a real and declared cost of
    # shooting the biggest thing the mount takes.
    sleeved = liner_outer_r <= OUTER_BARREL_BORE_M / 2.0 - 0.004
    if not sleeved:
        liner_outer_r = OUTER_BARREL_BORE_M / 2.0
        liner_wall_m = 0.0
    outer_wall_m = 0.022
    outer_r = OUTER_BARREL_BORE_M / 2.0 + outer_wall_m
    annulus_m = OUTER_BARREL_BORE_M / 2.0 - liner_outer_r
    WASHER_PITCH_M = 0.420                      # how far apart they screw in
    N_WASHER = max(3, int(barrel_length_m / WASHER_PITCH_M) + 1)
    barrel_z0 = barrel_centre_z - barrel_length_m / 2.0
    g.assembly = "gun-tube"
    g.motion_group = "elevation"

    # ---- the two tubes, as prisms with ports on their real surfaces --
    # A tube has a BORE as well as an outside, and a washer bears on
    # both: on the inner barrel's outside and on the outer barrel's
    # inside. Written as two nodes on the axis those two faces are the
    # same point, and every washer came out zero length -- which is to
    # say there was no annulus at all, which is to say no cooling.
    washer_z = [barrel_z0 + barrel_length_m * (i / (N_WASHER - 1))
                for i in range(N_WASHER)]
    WASHER_ANGLES = (0.0, 1.5708, 3.14159, 4.71239)
    liner_ports, outer_ports = {}, {}
    for i in range(N_WASHER):
        along = -1.0 + 2.0 * (i / (N_WASHER - 1))
        for j, ang in enumerate(WASHER_ANGLES):
            liner_ports[f"w{i}.{j}"] = None
            outer_ports[f"w{i}.{j}"] = None

    # ---- THE LINER WEIGHS WHAT THE LINER IS ----
    # `barrel_mass_kg` estimates A WHOLE BARREL FOR THE CALIBRE -- a
    # cylinder of 0.72 x bore, 82% filled -- and handing that to the
    # liner node was wrong in both directions. At 20 mm it charged 106 kg
    # for a tube whose own geometry is 26 kg. At 120 mm, where there is
    # no liner at all and `turret.weapon` is the outer barrel's own bore
    # described a second time, it charged 1268 kg for a tube of zero wall
    # thickness -- so the recoiling mass counted the barrel twice and the
    # 120 mm came out HEAVIER than the 20 mm, which is backwards: the
    # maximum-ordnance configuration has no liner and no filler sleeve
    # and is the LIGHTEST of the family.
    #
    # That error flattered the design. Recoil energy is p^2/2m, so a
    # lighter recoiling mass is a HARDER recoil for the same shot, and
    # the duplicate was hiding it.
    #
    # Unsleeved, the node carries no mass at all: it is a surface of the
    # outer barrel, and the `rigid-offset` between them already declares
    # them one piece. Sleeved, it weighs its own annulus.
    _liner_mass_kg = 0.0
    if sleeved:
        _liner_area = math.pi * (liner_outer_r ** 2
                                 - (bore_mm / 2000.0) ** 2)
        _liner_mass_kg = _liner_area * barrel_length_m * 7850.0
    inner_prism = Prism(
        identity="turret.weapon", kind="recoiling-weapon-mass",
        centre=(0.0, gun_y, barrel_centre_z), shape="tube",
        axis=(0.0, 0.0, 1.0), radius=liner_outer_r,
        inner_radius=bore_mm / 2000.0, length=barrel_length_m,
        material="gun-steel", mass_kg=round(_liner_mass_kg, 1),
        attributes=dict(
            motion_group="fine", recoils=True, mass_in_total=False,
            part_role="inner-barrel" if sleeved else "bore-of-the-outer-barrel",
            barrel_bore_mm=bore_mm,
            wall_m=round(liner_wall_m, 4),
            interchangeable=sleeved, wear_part=True,
            sleeved=sleeved,
            goes_inside="turret.outer_barrel" if sleeved else None,
            terminates_at="turret.collar",
            muzzle_z_m=round(barrel_z0 + barrel_length_m, 4)))
    outer_prism = Prism(
        identity="turret.outer_barrel", kind="recoiling-weapon-mass",
        centre=(0.0, gun_y, barrel_centre_z), shape="tube",
        axis=(0.0, 0.0, 1.0), radius=outer_r,
        inner_radius=OUTER_BARREL_BORE_M / 2.0, length=barrel_length_m,
        material="gun-steel",
        mass_kg=round(_mass_of("gun-steel", math.pi
                               * (outer_r ** 2 - (OUTER_BARREL_BORE_M / 2) ** 2)
                               * barrel_length_m), 1),
        attributes=dict(
            motion_group="fine", recoils=True, mass_in_total=False,
            part_role="outer-barrel",
            bore_mm=OUTER_BARREL_BORE_M * 1000.0,
            max_ordnance_mm=MAX_ORDNANCE_MM,
            accepts_inner_barrel_up_to_mm=MAX_ORDNANCE_MM,
            wall_m=outer_wall_m,
            coolant="water" if sleeved else None,
            water_jacket=sleeved,
            no_jacket_because=None if sleeved else
                "at maximum ordnance the round is fired from this tube "
                "itself; there is no annulus to cool it with",
            coolant_annulus_m=round(annulus_m, 4) if sleeved else 0.0,
            washer_thread="screw-in-chambered-washer" if sleeved else None,
            washer_pitch_m=WASHER_PITCH_M if sleeved else None,
            carries="bending-mounting-and-recoil-loads"))

    _lp, _op = {}, {}
    for i in range(N_WASHER):
        along = -1.0 + 2.0 * (i / (N_WASHER - 1))
        for j, ang in enumerate(WASHER_ANGLES):
            _lp[f"w{i}.{j}"] = inner_prism.side_point(ang, along=along,
                                                      surface="outer")
            _op[f"w{i}.{j}"] = outer_prism.side_point(ang, along=along,
                                                      surface="inner")
    liner_nodes = emit_prism(g, inner_prism, _lp, motion_group="fine",
                             assembly="gun-tube",
                             port_kwargs=dict(recoils=True, mass_in_total=False))
    outer_nodes = emit_prism(g, outer_prism, _op, motion_group="fine",
                             assembly="gun-tube",
                             port_kwargs=dict(recoils=True, mass_in_total=False))

    # ---- THE SCREW-IN WASHERS ----
    # Each is a threaded annular ring: it screws into the outer barrel
    # and closes on the inner one. Chambered, so water passes; the open
    # fraction is declared and is what the flow is sized against.
    washer_area = math.pi * ((OUTER_BARREL_BORE_M / 2.0) ** 2
                             - liner_outer_r ** 2)
    for i in range(N_WASHER if sleeved else 0):
        for j, ang in enumerate(WASHER_ANGLES):
            g.edge(f"turret.washer.{i}.{j}", liner_nodes[f"w{i}.{j}"],
                   outer_nodes[f"w{i}.{j}"], "rigid-distance",
                   radius=max(0.005, annulus_m * 0.42),
                   palette="rollbar-silver",
                   part_role="screw-in-chambered-washer",
                   thread="screwed-into-the-outer-barrel",
                   chambered=True, open_flow_fraction=0.62,
                   annulus_m=round(annulus_m, 4),
                   flow_area_m2=round(washer_area * 0.62, 6),
                   holds_concentric=("turret.weapon", "turret.outer_barrel"),
                   load_path="inner-barrel-held-concentric-and-load-shared-"
                             "into-the-outer-barrel")

    if not sleeved:
        # the two surfaces are the same surface, so they are welded
        # rather than washered: this is one tube described twice, and
        # saying so is better than pretending a zero-length spacer is a
        # spacer
        for i in range(N_WASHER):
            for j in range(len(WASHER_ANGLES)):
                g.edge(f"turret.bore_weld.{i}.{j}", liner_nodes[f"w{i}.{j}"],
                       outer_nodes[f"w{i}.{j}"], "rigid-offset",
                       radius=0.010, palette="rollbar-silver",
                       load_path="at-max-ordnance-the-bore-IS-the-outer-barrel")

    # ================= THE UNDER-BARREL SPINE =================
    # One square steel channel under the tube, doing six jobs, because
    # once you have committed to a stiff hollow section running the
    # length of the gun it is absurd for it to carry only gas.
    #
    #   STIFFNESS. A square hollow section puts material further from
    #   the neutral axis than a round one of the same envelope -- 1.70x
    #   the second moment for the same outside dimension and wall. Slung
    #   underneath, where the tube droops, it is material on the side
    #   that is in tension.
    #
    #   THE OPTICAL TUNNEL. A laser fires from the breech end down a
    #   sealed, armoured bore to a reference target at the muzzle. That
    #   is a muzzle reference system that does not have to see past the
    #   muzzle through its own blast and fume -- it reads the barrel's
    #   true shape from inside a protected tube that bends with it.
    #
    #   THE EVACUATOR. A 1 litre tube inside the channel, ported to the
    #   bore with forward rake. The scavenging job needs litres, not
    #   the forty-six the structural envelope happens to enclose --
    #   sizing the gas chamber by what the beam wanted was two problems
    #   sharing one number.
    #
    #   DISPERSAL. A line down the channel to the muzzle end, fired by a
    #   blank or by the shot itself, to put whatever is packed in it out
    #   in front of the gun.
    #
    #   SENSORS, run in a protected conduit rather than strapped to a
    #   barrel that gets hot enough to cook them.
    #
    #   SUPPORT. Hard points at intervals. Firing steeply upward the
    #   barrel is a mast, and a mast wants to be held somewhere other
    #   than its own trunnion.
    #
    # MODULAR IN LENGTH, in sections, so the same channel serves a long
    # 120 tube and a nimble autocannon by fitting fewer of them.
    g.assembly = "under-barrel-spine"
    g.motion_group = "fine"
    SPINE_SECTION_M = 1.05               # one module
    # ================= THE SPINE IS THE RECOIL TANK =================
    # It was a 110 mm square channel under the barrel, and the absorber
    # was a separate 218 mm casing slung below it -- two bodies doing
    # one job, and the fatter of them hanging off the thinner.
    #
    # THE SECTION IS SET BY WHAT HAS TO FIT THROUGH IT, in this order,
    # none of it chosen:
    #
    #   THREE 84 mm BORES. Each absorber element alone takes a full
    #   charge at the frame limit, which at 200 bar is 55 cm2 of piston.
    #   On a triangle they leave 27 mm of clear centre, and 27 mm is not
    #   a channel.
    #
    #   A CHANNEL UP THE MIDDLE. The trunnion hangs off the piece BELOW
    #   the tank and has to reach the piece ABOVE it, so a strut passes
    #   through the tank on a slot long enough for the whole stroke.
    #   Pushing the bores outboard to clear a 90 mm channel takes the
    #   pitch radius to 99 mm and the section to 305 mm square.
    #
    # AND THE SECTION IS WHY THE OFFSET IS AFFORDABLE. A trunnion under
    # the tank sits 440 mm below the bore, so full recoil is 48 kN.m of
    # couple -- six times what the old arrangement had. Taken by two
    # discrete carriages that would be ruinous. Taken by the whole
    # square as a JOURNAL, over a two-metre engagement, it is a 24 kN
    # pair at 0.26 MPa of bearing pressure. The distribution is not a
    # refinement of the design, it is the thing that makes the layout
    # legal.
    ABSORBER_BORE_M = 0.0837            # 110 kN at 200 bar, each alone
    TANK_WALL_M = 0.012
    TANK_CHANNEL_M = 0.090              # the strut's slot
    _pitch = max((ABSORBER_BORE_M + TANK_WALL_M)
                 / (2.0 * math.sin(math.radians(60.0))),
                 TANK_CHANNEL_M / 2.0 + TANK_WALL_M + ABSORBER_BORE_M / 2.0)
    ABSORBER_PITCH_M = _pitch
    spine_side_m = 2.0 * (_pitch + ABSORBER_BORE_M / 2.0 + TANK_WALL_M)
    #: cylinder closed length plus the stroke: past this the tank is a
    #: hollow journal of the same outside, because there is nothing left
    #: to bore.
    _TANK_BORED_LENGTH = 0.844 + 0.220 + 0.844

    def _tank_section_area_m2(z_here: float) -> float:
        """Metal in the section at this station."""
        outer = spine_side_m ** 2
        if z_here <= spine_z0 + _TANK_BORED_LENGTH:
            return (outer
                    - 3.0 * math.pi * (ABSORBER_BORE_M / 2.0) ** 2
                    - math.pi * (TANK_CHANNEL_M / 2.0) ** 2)
        inner_side = spine_side_m - 2.0 * (TANK_WALL_M * 2.0)
        return outer - inner_side ** 2 - math.pi * (TANK_CHANNEL_M / 2.0) ** 2
    spine_wall_m = 0.008
    spine_len = barrel_length_m * 0.62
    spine_modules = max(2, int(round(spine_len / SPINE_SECTION_M)))
    spine_len = spine_modules * SPINE_SECTION_M
    spine_y = gun_y - outer_r - spine_side_m / 2.0 - 0.012
    spine_z0 = barrel_z0 + barrel_length_m * 0.30
    inner = spine_side_m - 2 * spine_wall_m
    for i in range(spine_modules + 1):
        z = spine_z0 + SPINE_SECTION_M * i
        g.node(f"turret.spine.{i}", [0.0, spine_y, z],
               "load-bearing-structure", material="steel-plate",
               mass_in_total=False,
               # A FORGING WHERE THE BORES ARE, A JOURNAL EVERYWHERE
               # ELSE. The outside is 305 mm square for the whole
               # length because the whole length is bearing surface --
               # that is what "all of it is a journal" means. But the
               # three cylinders only exist where the absorber does,
               # which is the closed length plus the stroke, and past
               # that there is nothing to bore. Making the forging run
               # the full 5.25 m put 3.5 TONNES of solid steel under a
               # 20 mm gun, which the balance then dutifully chased,
               # dragging the trunnion out to z=4.79.
               mass_kg=round(_tank_section_area_m2(z) * 7850.0
                             * SPINE_SECTION_M, 2),
               bored_here=bool(z <= spine_z0 + _TANK_BORED_LENGTH),
               part_role="recoil-tank-module",
               section="square-journal-with-three-bores",
               recoil_tank=True, journal_face_m=spine_side_m,
               absorber_bore_m=ABSORBER_BORE_M,
               absorber_pitch_m=ABSORBER_PITCH_M,
               channel_diameter_m=TANK_CHANNEL_M,
               section_across_m=spine_side_m, wall_m=spine_wall_m,
               module_length_m=SPINE_SECTION_M,
               modular=True, armoured=True,
               carries=("optical-reference-tunnel", "bore-evacuator-tube",
                        "dispersal-line", "sensor-conduit"),
               beam_solvable=True,
               half_extent_m=(spine_side_m / 2.0, spine_side_m / 2.0,
                              SPINE_SECTION_M / 2.0))
    for i in range(spine_modules):
        g.edge(f"turret.spine_joint.{i}", f"turret.spine.{i}",
               f"turret.spine.{i + 1}", "rigid-distance",
               radius=spine_side_m / 2.0, wall_m=spine_wall_m, alloy="a36",
               palette="rollbar-silver", beam_solvable=True,
               part_role="spine-module-joint",
               load_path="square-hollow-section-stiffening-the-tube")
    # One fabricated clamp strap per module.  This used to be two abstract
    # centre-to-surface spokes called "bands".  They had neither a perimeter,
    # a hinge, a tensioning screw nor contact surfaces, so fixed beam ends were
    # standing in for all four pieces of real hardware.
    from fabrication import WrapProfile, emit_shaped_iron_strap
    for i in range(spine_modules + 1):
        z = spine_z0 + SPINE_SECTION_M * i
        emit_shaped_iron_strap(
            g, f"turret.spine_strap.{i}",
            (WrapProfile("turret.outer_barrel", "circle", (0.0, gun_y),
                         radius_m=outer_r),
             WrapProfile(f"turret.spine.{i}", "box", (0.0, spine_y),
                         half_extent_m=(spine_side_m / 2.0,
                                        spine_side_m / 2.0))),
            station_m=z, width_m=0.100, thickness_m=0.020,
            clearance_m=0.002, material="4130n", bolt="M20",
            bolt_class="10.9", set_torque_nm=430.0)
        # SUPPORT HARD POINTS, at every module joint. Firing near
        # vertical the tube is a mast and wants holding somewhere other
        # than the trunnion it is already hanging off.
        g.node(f"turret.spine.hardpoint.{i}",
               [0.0, spine_y - spine_side_m / 2.0 - 0.010,
                spine_z0 + SPINE_SECTION_M * i],
               "chassis-load-node", material="hardened-steel",
               mass_in_total=False, mass_kg=1.4,
               part_role="spine-hardpoint",
               takes="support-strut-or-utility-mount",
               rated_n=40_000.0,
               half_extent_m=(0.035, 0.010, 0.035))
        g.edge(f"turret.spine.hardpoint_web.{i}",
               f"turret.spine.hardpoint.{i}", f"turret.spine.{i}",
               "rigid-distance", radius=0.018, palette="rollbar-silver",
               load_path="hardpoint-into-the-spine")
        # triangulate BOTH ways along the spine -- a hardpoint tied
        # only forward is a hinge at the last module, which is exactly
        # where a support strut would be loading it
        for j in {min(i + 1, spine_modules), max(i - 1, 0)}:
            if j != i:
                g.edge(f"turret.spine.hardpoint_tie.{i}.{j}",
                       f"turret.spine.hardpoint.{i}", f"turret.spine.{j}",
                       "rigid-distance", radius=0.012,
                       palette="rollbar-silver",
                       load_path="hardpoint-triangulated-along-the-spine")

    # ---- WHAT LIVES INSIDE THE SPINE ----
    g.assembly = "bore-evacuator"
    # THE SECRET CLEARER: one litre, not forty-six. Sized for the job it
    # does -- trapping a breath of bore gas and blowing it forward when
    # the bore drops to atmosphere -- rather than for the envelope the
    # beam happened to need.
    evac_r = math.sqrt(0.001 / (math.pi * spine_len))
    for i in range(spine_modules + 1):
        g.node(f"turret.evacuator.{i}",
               [0.0, spine_y + 0.018, spine_z0 + SPINE_SECTION_M * i],
               "load-bearing-structure", material="steel-pipe",
               mass_in_total=False, mass_kg=0.9,
               part_role="bore-evacuator-tube",
               fluid="propellant-gas",
               fluid_volume_l=round(1.0 / (spine_modules + 1), 4),
               inside="turret.spine",
               beam_solvable=False,
               half_extent_m=(evac_r, evac_r, SPINE_SECTION_M / 2.0))
        g.edge(f"turret.evacuator_housed.{i}", f"turret.evacuator.{i}",
               f"turret.spine.{i}", "rigid-distance", radius=0.010,
               palette="rollbar-silver",
               load_path="evacuator-tube-carried-inside-the-spine")
        g.edge(f"turret.evacuator_housed_b.{i}", f"turret.evacuator.{i}",
               f"turret.spine.{max(i - 1, 0) if i else 1}",
               "rigid-distance", radius=0.008, palette="rollbar-silver",
               load_path="second-carrier-so-the-tube-is-held-not-hung")
    for i in range(spine_modules):
        # A GAS LINE, not a coolant line. It was on `coolant-line`
        # because that was the only kind I had checked the engine
        # discovers -- which made the propellant-gas plenum come back
        # from `_discover_fluid_circuits` classified as thermal-liquid,
        # and it would have been stepped as though it were full of
        # water. `exhaust-flow-path` is what this actually is: hot dirty
        # combustion gas on its way out.
        g.edge(f"turret.evacuator_run.{i}", f"turret.evacuator.{i}",
               f"turret.evacuator.{i + 1}", "exhaust-flow-path",
               radius=evac_r, circuit_identity="bore-evacuator",
               material="steel-pipe", wall_m=0.002,
               palette="rollbar-silver",
               load_path="the-evacuator-plenum-along-the-spine")
    # the ports: forward-raked, which IS the scavenging mechanism
    for i in range(0, spine_modules + 1, 2):
        f = i / spine_modules
        w = min(int(f * (N_WASHER - 1)), N_WASHER - 1)
        # This is a drilled/raked void through metal already owned by the
        # evacuator and outer-barrel bodies. It has flow area, but no separate
        # mass, section or material capacity of its own.
        g.edge(f"turret.evacuator_port.{i}", f"turret.evacuator.{i}",
               outer_nodes[f"w{w}.3"], "exhaust-flow-path",
               radius=0.006, wall_m=0.0, material="propellant-gas",
               circuit_identity="bore-evacuator", palette="service-line",
               part_role="evacuator-port", rake_deg=35.0,
               port_area_m2=round(math.pi * 0.006 ** 2, 6),
               cleanliness_fraction=1.0, fouling_mass_g=0.0,
               occluded_area_fraction=0.0,
               service_state="clean-open",
               service_action="bore-brush-and-solvent-flush",
               blows="toward-the-muzzle-once-the-bore-falls-to-atmosphere",
               load_path="evacuator-ported-into-the-bore")

    # ---- THE OPTICAL REFERENCE TUNNEL ----
    # A laser at the breech, a target at the muzzle, and a sealed
    # armoured bore between them that bends with the barrel it is
    # strapped to. It never has to look through muzzle blast.
    g.assembly = "muzzle-reference"
    for tag, i, role in (("emitter", 0, "reference-laser"),
                         ("target", spine_modules, "reference-target")):
        g.node(f"turret.mrs.{tag}",
               [0.0, spine_y - 0.020, spine_z0 + SPINE_SECTION_M * i],
               "load-bearing-structure", material="hardened-steel",
               mass_in_total=False, mass_kg=2.2,
               part_role=role, measures="barrel-true-shape",
               resolution_rad=20.0e-6,
               half_extent_m=(0.028, 0.028, 0.040))
        g.edge(f"turret.mrs.{tag}_mount", f"turret.mrs.{tag}",
               f"turret.spine.{i}", "rigid-distance", radius=0.014,
               palette="rollbar-silver",
               load_path="reference-optic-mounted-in-the-spine")
        g.edge(f"turret.mrs.{tag}_stay", f"turret.mrs.{tag}",
               f"turret.spine.hardpoint.{i}", "rigid-distance",
               radius=0.010, palette="rollbar-silver",
               load_path="reference-optic-stay")
        # AN OPTIC ON TWO POINTS CAN ROTATE ABOUT THE LINE BETWEEN THEM,
        # which for a reference instrument is the one motion that
        # ruins it: it would read a bend that is its own mounting.
        g.edge(f"turret.mrs.{tag}_third", f"turret.mrs.{tag}",
               f"turret.spine.{max(i - 1, 0) if i else 1}",
               "rigid-distance", radius=0.010, palette="rollbar-silver",
               load_path="third-point-so-the-optic-cannot-roll")
    g.assembly = "gun-tube"

    # the collar clamps the OUTER barrel, which is the tube that stays
    g.edge("turret.collar_to_outer_barrel", outer_nodes["w0.3"],
           "turret.collar", "rigid-distance", radius=outer_r * 0.7,
           palette="rollbar-silver",
           load_path="the-outer-barrel-is-what-the-collar-clamps")

    # ---- THE WATER JACKET AND ITS PORTS ----
    # Declared through the project's own port system, so they read as
    # OPEN until something is plumbed to them and the report says which.
    from assembly_ports import PartPort, mate_ports, emit_ports_graph
    _cool = []
    for tag, along, dirn in (("in", -0.86, (0.0, -1.0, 0.0)),
                             ("out", 0.86, (0.0, 1.0, 0.0))):
        ang = 4.71239 if dirn[1] < 0 else 1.5708
        at = outer_prism.side_point(ang, along=along, surface="outer")
        _cool.append(PartPort(
            f"turret.outer_barrel.water_{tag}", "turret.outer_barrel",
            "coolant", at.position.copy(), np.asarray(dirn, dtype=float),
            max(0.010, annulus_m * 0.55), False, fluid="coolant",
            joint="bolted-flange"))
    if not sleeved:
        _cool = []          # no annulus, so no water ports to declare
    _cres = mate_ports(_cool)
    emit_ports_graph(
        _cool, _cres,
        lambda ident, pos, kind, **kw: g.node(
            ident, pos, "engine-block-port", motion_group="fine", recoils=True,
            material="steel-pipe", mass_in_total=False,
            half_extent_m=(0.030, 0.030, 0.030), **kw),
        lambda ident, a_, b_, constraint, **kw: g.edge(
            ident, a_, b_, constraint, palette="rollbar-silver"),
        prefix="turret")
    for tag, i in ((("in", 0), ("out", N_WASHER - 1)) if sleeved else ()):
        for j in (0, 1, 3):
            g.edge(f"turret.water_boss.{tag}.{j}",
                   f"turret.turret.outer_barrel.water_{tag}",
                   outer_nodes[f"w{i}.{j}"], "rigid-distance", radius=0.016,
                   palette="rollbar-silver",
                   load_path="water-boss-welded-through-the-outer-barrel-wall")

    # ---- THE WATER JACKET AS A REAL FLUID CIRCUIT ----
    # The washers already divide the annulus: four legs per bay at 0,
    # 90, 180 and 270 degrees, so every bay is four channels around the
    # tube. Sealed between bays lengthwise, each of those is a CHAMBER
    # with a volume, a wall and a temperature.
    #
    # These go in the graph as nodes and coolant-line edges, which means
    # `_discover_fluid_circuits` finds them by its own union-find over
    # connectivity and steps them with the engine's own thermal-liquid
    # physics -- the same code path the powertrain's coolant loop uses.
    # Nothing here simulates a temperature; it describes the plumbing
    # and lets the fluid engine do it.
    #
    # COUNTERFLOW, in the routing rather than in a comment: the supply
    # enters at the BREECH end where the metal is hottest and works
    # forward to the muzzle, so the coldest water always meets the
    # hottest steel.
    if sleeved:
        # ---- THE FILLER SLEEVE ----
        # The annulus between a small liner and the 120 mm outer bore is
        # a BATH, not a jacket: 45 mm of nearly still water giving 26 W/K
        # where 919 W/K is needed to hold the tube at 40 C. The sleeve
        # packs it out to a narrow channel, and that is where all the
        # heat transfer comes from -- same pump, same water, 130x the
        # coefficient, because the film coefficient goes as Reynolds to
        # the 0.8 and Reynolds goes with velocity.
        JACKET_GAP_M = 0.005
        sleeve_outer_r = OUTER_BARREL_BORE_M / 2.0 - JACKET_GAP_M
        sleeve_inner_r = liner_outer_r
        sleeve_ok = sleeve_outer_r > sleeve_inner_r + 0.002
        jacket_wall_m = 0.004
        chamber_len = barrel_length_m / max(N_WASHER - 1, 1)
        quadrants = (("top", 1.5708), ("right", 0.0),
                     ("bottom", 4.71239), ("left", 3.14159))
        g.assembly = "barrel-cooling"
        g.motion_group = "fine"
        chamber_r = (OUTER_BARREL_BORE_M / 2.0 + liner_outer_r) / 2.0
        for i in range(N_WASHER - 1):
            zc = washer_z[i] + chamber_len / 2.0
            for qname, ang in quadrants:
                # the annulus quarter this chamber actually is
                _inner = sleeve_outer_r if sleeve_ok else liner_outer_r
                vol_l = (math.pi * ((OUTER_BARREL_BORE_M / 2.0) ** 2
                                    - _inner ** 2)
                         * chamber_len / 4.0) * 1000.0
                g.node(f"barrel.coolant.{i}.{qname}",
                       [chamber_r * math.cos(ang),
                        gun_y + chamber_r * math.sin(ang), zc],
                       "coolant-chamber", material="gun-steel",
                       mass_in_total=False,
                       solver_condensed_into="turret.outer_barrel",
                       solver_condensed_mass=False,
                       part_role="barrel-jacket-chamber",
                       fluid="coolant", fluid_volume_l=round(vol_l, 4),
                       station=i, quadrant=qname,
                       between_washers=(i, i + 1),
                       annulus_m=round(JACKET_GAP_M if sleeve_ok
                                       else annulus_m, 4),
                       sleeved_by="turret.filler_sleeve" if sleeve_ok else None,
                       cools="turret.weapon",
                       half_extent_m=(0.020, 0.020, chamber_len / 2.0))
        # the supply manifold feeds every quadrant at the breech end,
        # and each quadrant runs forward chamber to chamber
        for qname, ang in quadrants:
            g.edge(f"barrel.coolant_in.{qname}",
                   f"turret.turret.outer_barrel.water_in",
                   f"barrel.coolant.0.{qname}", "coolant-line",
                   radius=max(0.006, annulus_m * 0.4),
                   circuit_identity="barrel-coolant",
                   material="steel-pipe", wall_m=jacket_wall_m,
                   palette="rollbar-silver",
                   valve="quadrant-trim",
                   load_path="coldest-water-to-the-breech-end")
            for i in range(N_WASHER - 2):
                g.edge(f"barrel.coolant_run.{i}.{qname}",
                       f"barrel.coolant.{i}.{qname}",
                       f"barrel.coolant.{i + 1}.{qname}", "coolant-line",
                       radius=max(0.005, annulus_m * 0.35),
                       circuit_identity="barrel-coolant",
                       material="steel-pipe", wall_m=jacket_wall_m,
                       palette="rollbar-silver",
                       load_path="through-the-chambered-washer-forward")
            g.edge(f"barrel.coolant_out.{qname}",
                   f"barrel.coolant.{N_WASHER - 2}.{qname}",
                   f"turret.turret.outer_barrel.water_out", "coolant-line",
                   radius=max(0.006, annulus_m * 0.4),
                   circuit_identity="barrel-coolant",
                   material="steel-pipe", wall_m=jacket_wall_m,
                   palette="rollbar-silver",
                   load_path="warmed-water-out-at-the-muzzle-end")
        # each chamber is against the steel it cools: this is the path
        # the heat actually takes, and it is a member like any other
        for i in range(N_WASHER - 1):
            for qname, ang in quadrants:
                g.edge(f"barrel.coolant_wet.{i}.{qname}",
                       f"barrel.coolant.{i}.{qname}",
                       outer_nodes[f"w{i}.0"], "rigid-offset",
                       radius=0.010, palette="rollbar-silver",
                       part_role="jacket-wetted-wall",
                       load_path="chamber-against-the-tube-it-cools")
        if sleeve_ok:
            sleeve_vol = (math.pi * (sleeve_outer_r ** 2 - sleeve_inner_r ** 2)
                          * barrel_length_m)
            g.node("turret.filler_sleeve", [0.0, gun_y, barrel_centre_z],
                   "load-bearing-structure", motion_group="fine", recoils=True,
                   # ALUMINIUM, and the label now matches the density the
                   # mass was computed at. It said pressed-steel and
                   # weighed 2700 kg/m3, so the graph carried a steel
                   # part with an aluminium mass -- 33 kg where steel
                   # would be 96.
                   mass_in_total=False, material="6061t6",
                   youngs_modulus_pa=69.0e9,
                   mass_kg=round(sleeve_vol * 2700.0, 2),
                   solver_condensed_into="turret.outer_barrel",
                   solver_condensed_mass=True,
                   part_role="jacket-filler-sleeve",
                   shape="tube", tube_axis=(0.0, 0.0, 1.0),
                   tube_outer_radius_m=round(sleeve_outer_r, 4),
                   tube_inner_radius_m=round(sleeve_inner_r, 4),
                   drum_length_m=barrel_length_m,
                   leaves_gap_m=JACKET_GAP_M,
                   half_extent_m=(sleeve_outer_r, sleeve_outer_r,
                                  barrel_length_m / 2.0))
            for i in (0, N_WASHER - 1):
                for j in range(len(WASHER_ANGLES)):
                    g.edge(f"turret.sleeve_seat.{i}.{j}", "turret.filler_sleeve",
                           outer_nodes[f"w{i}.{j}"], "rigid-distance",
                           radius=0.012, palette="rollbar-silver",
                           load_path="sleeve-seated-in-the-outer-barrel")
        g.assembly = "gun-tube"


    g.edge("turret.barrel_to_collar", liner_nodes["w0.1"], "turret.collar", "rigid-distance",
           radius=max(0.022, barrel_radius_m * 0.55), palette="rollbar-silver",
           load_path="every-barrel-terminates-at-the-same-collar-interface")
    g.edge("turret.collar_to_breech", "turret.collar", "turret.breech", "rigid-distance",
           radius=breech_radius_m * 0.5, palette="rollbar-silver",
           load_path="chamber-pressure-and-recoil-thrust-into-the-breech-ring")
    g.edge("turret.breech_to_barrel", "turret.breech", liner_nodes["w0.0"], "rigid-distance",
           radius=max(0.020, barrel_radius_m * 0.45), palette="rollbar-silver",
           load_path="breech-and-tube-are-one-recoiling-body")
    g.assembly = "breech"
    for k in range(3):
        g.edge(f"turret.blast_panel_seat.{k}", "turret.blast_panel", "turret.breech",
               "tension-limit-strap", radius=0.016, palette="actuator-yellow",
               release_force_n=3.5e6 * math.pi * (breech_radius_m * 0.8) ** 2 / 3.0,
               load_path="frangible-seat-that-lets-the-breech-open-rearward")
    g.assembly = "gun-support-structure"
    # THE STANDARD STANDS ON THE RING. Run from a node on the bore axis
    # it went straight through the breech -- 22 of 41 samples -- and
    # through the collar as well. A real trunnion standard reaches up
    # the side of the gun to the trunnion, which is both out of the
    # metal and the shape that resists the firing moment.
    # ============ THE TURNTABLE CARRIES ITS OWN FLOOR ============
    # What was here: four struts from the ring to the trunnion. Two
    # "cheeks" that were supposed to straddle the case shaft and both
    # resolved to turret.yaw.6 -- the ring has eight nodes, so +-0.2 m
    # of intended separation rounded to the same one and the pair became
    # one strut drawn twice. And two braces reaching 3.6 and 4.7 metres
    # diagonally back to the race corners. The frame solve had them at
    # 1815 kN, which is what happens when a four-metre overhang is hung
    # off pin-ended sticks.
    #
    # A TURNTABLE DOES NOT HANG OFF STICKS. It has a floor, and the
    # floor rests on the ring below it over the whole annulus:
    #
    #   THE TRACK IS THE TOP FACE OF THE LOWER RING, greased, and the
    #   floor sits on it. Free in exactly one rotation -- about the
    #   vertical -- and rigid in the other five, which is the joint now
    #   declared in joints.py rather than approximated with a strut.
    #
    #   HEAVY ROLLERS TAKE WHAT THE FILM WILL NOT. Distributed round
    #   the track so no part of it is asked for more than it can hold.
    #
    #   AND HOLD-DOWNS UNDER THE LIP, because a flat track carries
    #   compression and nothing else. The gun overhangs forward, so the
    #   aft side of the ring is in LIFT, and without hold-downs the
    #   whole turntable is a hinge.
    #
    # THE OVERHANG IS THE WHOLE PROBLEM and it is worth writing down:
    # the trunnion is 3.4 m forward of the ring's edge, carrying 5066 kg
    # of elevating mass. That is 204 kN.m about the ring centre, which
    # the ring reacts as a couple across its own diameter: 152 kN
    # pressing down at the front and 152 kN LIFTING at the back, before
    # a shot is fired. The floor exists to turn that into a distributed
    # load on a large-diameter face instead of an axial load in a stick.
    # ============ THE TURNTABLE IS A BOXED DRUM ============
    # What was here: a ring of eight pads on a flat greased face, eight
    # rollers, eight hold-downs, and a rim ring of beams tying them
    # together. It worked and it was a track, not a bearing -- a flat
    # face carries compression and has no answer to lift, so under the
    # gun's overhang it separated over half its annulus and put 218 kN
    # into the hold-downs, which were then the load path rather than
    # retainers.
    #
    # A BOX FIXES THAT BY HAVING TWO FACES. Top annulus, bottom annulus,
    # cylinder wall between them: one closed cell, and the overturning
    # is reacted as a push/pull PAIR across the box's full depth instead
    # of as compression on a single face that can only push. Closing the
    # cell is also the largest structural return available anywhere in
    # this machine -- measured on this section, 6.6x stiffer out of
    # plane for the price of a wall.
    #
    # AND EVERY FITTING NOW HAS A SURFACE IT BELONGS TO:
    #   the top annulus      the floor truss welds to it, anywhere
    #   both outer rims      captive races, pressed
    #   the middle band      the gear the pinions run in
    #   the bottom annulus   what the whole thing stands on
    from surfaces import BoxedRing, BearingRace, emit_boxed_ring, emit_race
    g.assembly = "turntable-drum"
    g.motion_group = "traverse"
    #: What the seat is designed against, from ring_joints.turret_over-
    #: turning on the built graph: the mass above the ring and the
    #: overturning it develops, dead plus recoil. Stated as the DESIGN
    #: load so changing the gun changes the bearing rather than
    #: silently overloading it.
    RING_DRUM_MOMENT_NM = 259_800.0
    DRUM_DEPTH_M = 0.300
    DRUM_PLATE_M = 0.040
    DRUM_WALL_M = 0.025
    # A METRE OF ANNULUS. The radial width is the drum's whole reason
    # for existing: it is the lever the two races work at, the depth of
    # plate the truss can weld anywhere on, and the span the inner and
    # outer walls close a cell across. A 300 mm ring was a bearing with
    # a seat; a 1000 mm one is a floor that happens to rotate.
    #
    # THE OUTER RADIUS FOLLOWS THE STATIC RING, because that is what the
    # races seat against -- set them independently and the seat members
    # become 800 mm struts reaching for a ring that is not there.
    DRUM_RADIAL_WIDTH_M = 1.00
    _drum_r = ring_radius * 0.98
    _drum_y = race_y + DRUM_DEPTH_M / 2.0
    _drum = BoxedRing(identity="turret.drum",
                      centre=(0.0, _drum_y, 0.0),
                      inner_radius_m=_drum_r - DRUM_RADIAL_WIDTH_M,
                      outer_radius_m=_drum_r,
                      height_m=DRUM_DEPTH_M,
                      plate_thickness_m=DRUM_PLATE_M,
                      wall_thickness_m=DRUM_WALL_M,
                      segments=RING_N)
    drum = emit_boxed_ring(g, _drum, motion_group="traverse",
                           assembly="turntable-drum")
    _floor_y = _drum_y + DRUM_DEPTH_M / 2.0
    # ---- TWO CAPTIVE RACES, ONE ABOVE AND ONE BELOW ----
    # Closed arcs of the same BearingRace a straight journal uses: the
    # shape is a parameter. Preloaded, so each carries tension as well
    # as compression and the pair can be a couple rather than a hinge.
    g.assembly = "slew-race"
    g.motion_group = "frame"
    _race_preload = round(RING_DRUM_MOMENT_NM / max(DRUM_DEPTH_M, 0.1)
                          / RING_N, 0)
    races = {}
    for tag, rim in (("upper", drum["upper"]["outer"]),
                     ("lower", drum["lower"]["outer"])):
        race = BearingRace(
            identity=f"turret.race.{tag}", stations=RING_N,
            centre=(0.0, _drum_y + (DRUM_DEPTH_M / 2.0
                                    if tag == "upper" else -DRUM_DEPTH_M / 2.0),
                    0.0),
            radius_m=_drum_r + 0.05, sweep_deg=360.0, axis=(0.0, 1.0, 0.0),
            element="slew-thrust-roller", preload_n=_race_preload)
        races[tag] = emit_race(g, race, lambda i, _r=rim: _r[i],
                               motion_group="frame", assembly="slew-race")
        # the race body itself is carried by the static ring below it
        for i in range(RING_N):
            g.edge(f"turret.race.{tag}.seat.{i}",
                   f"turret.race.{tag}.element.{i}", f"deck.ring.{i}",
                   "rigid-distance", radius=0.042, palette="chassis-grey",
                   load_path=f"{tag}-race-seated-on-the-static-ring")
    # ============ THE ROOM UNDER THE TURRET ============
    # The drum's bore is not a hole, it is the way IN. At a metre and a
    # seventy-five clear it takes a soldier in full kit climbing, which
    # is the dimension that decided it -- not a bearing clearance.
    #
    # AND IT IS PART OF THE LOWER DECK, not of the turret: bolted at its
    # rim to the BOTTOM annulus and standing on the bay floor, so the
    # turret turns around it and it never turns. That is also why the
    # bottom annulus is the room's CEILING, and why the pillars below
    # carry its outer rim: a ceiling three metres across with a gun on
    # it is not a plate, it is a floor upside down.
    #
    # BOLTED, NOT WELDED. The body is cast 6061 and the whole point of
    # casting it on site is that a cracked section can be replaced --
    # which a weld undoes. Every joint here is declared `bolted` so the
    # graph says what can be taken apart.
    g.assembly = "access-trunk"
    g.motion_group = "frame"
    _trunk_r = _drum_r - DRUM_RADIAL_WIDTH_M
    _trunk_top = race_y
    _floor_y_bay = -1.344
    _trunk_bottom = _floor_y_bay + 0.62          # where it opens out
    _TRUNK_N = 12
    _levels = (_trunk_top, (_trunk_top + _trunk_bottom) / 2.0, _trunk_bottom)
    for li, y in enumerate(_levels):
        for i in range(_TRUNK_N):
            a = 2.0 * math.pi * i / _TRUNK_N
            g.node(f"trunk.shell.{li}.{i}",
                   [_trunk_r * math.cos(a), y, _trunk_r * math.sin(a)],
                   "load-bearing-structure", material="6061t6",
                   mass_in_total=False, mass_kg=14.0,
                   part_role="access-trunk-shell", level=li,
                   clear_diameter_m=round(2.0 * _trunk_r - 0.05, 3),
                   climbable=True, fastening="bolted",
                   half_extent_m=(0.06, 0.18,
                                  max(math.pi * _trunk_r / _TRUNK_N, 0.02)))
    for li in range(len(_levels)):
        for i in range(_TRUNK_N):
            j = (i + 1) % _TRUNK_N
            g.edge(f"trunk.hoop.{li}.{i}", f"trunk.shell.{li}.{i}",
                   f"trunk.shell.{li}.{j}", "rigid-distance",
                   radius=0.028, alloy="6061t6", palette="chassis-grey",
                   beam_solvable=True, fastening="bolted",
                   load_path="access-trunk-hoop")
            if li:
                g.edge(f"trunk.stave.{li}.{i}", f"trunk.shell.{li - 1}.{i}",
                       f"trunk.shell.{li}.{i}", "rigid-distance",
                       radius=0.032, alloy="6061t6", palette="chassis-grey",
                       beam_solvable=True, fastening="bolted",
                       load_path="access-trunk-stave")
    # BOLTED AT THE RIM TO THE BOTTOM ANNULUS. Zero length, because the
    # two flanges are the same face seen from both sides.
    for i in range(_TRUNK_N):
        _near = int(round(i * RING_N / _TRUNK_N)) % RING_N
        g.edge(f"trunk.rim_bolt.{i}", f"trunk.shell.0.{i}",
               drum["lower"]["mean"][_near], "rigid-distance",
               radius=0.026, alloy="6061t6", palette="rollbar-silver",
               fastening="bolted",
               load_path="trunk-bolted-to-the-bottom-annulus-at-its-rim")
    # ---- SUPPORT FINS: TRIANGLES OFF THE TUBE ----
    # A tube in compression with a three-metre plate on top of it wants
    # hoop support at height, and a fin is the cheapest thing that gives
    # it: a triangle out from the shell to the deck, so the load path is
    # in the plane of the plate rather than through its flat.
    _FIN_ANGLES = (0.25, 0.75, 1.25, 1.75)
    for fi, frac in enumerate(_FIN_ANGLES):
        a = 2.0 * math.pi * frac / 2.0
        i0 = int(round(frac / 2.0 * _TRUNK_N)) % _TRUNK_N
        tip = f"trunk.fin.{fi}"
        g.node(tip, [(_trunk_r + 0.52) * math.cos(a), _trunk_bottom,
                     (_trunk_r + 0.52) * math.sin(a)],
               "load-bearing-structure", material="6061t6",
               mass_in_total=False, mass_kg=22.0,
               part_role="access-trunk-fin", fastening="bolted",
               half_extent_m=(0.26, 0.012, 0.26))
        for li, r in ((0, 0.026), (1, 0.030), (2, 0.026)):
            g.edge(f"trunk.fin_edge.{fi}.{li}", tip,
                   f"trunk.shell.{li}.{i0}", "rigid-distance",
                   radius=r, alloy="6061t6", palette="chassis-grey",
                   beam_solvable=True, fastening="bolted",
                   load_path="fin-triangle-into-the-trunk")
    # ---- AND AT THE BOTTOM IT STOPS BEING A PIPE ----
    # Four legs and a ladder. A closed tube all the way down would be a
    # shaft with no way out at the bottom and nowhere for the feed to
    # cross; opening it out puts the load into four columns and leaves
    # three sides of the base clear.
    _LEG_AT = (1, 4, 7, 10)
    for k, i in enumerate(_LEG_AT):
        a = 2.0 * math.pi * i / _TRUNK_N
        foot = f"trunk.leg.{k}"
        g.node(foot, [_trunk_r * math.cos(a) * 1.18, _floor_y_bay,
                      _trunk_r * math.sin(a) * 1.18],
               "structural-body-pin-frame-foot", material="6061t6",
               mass_in_total=False, mass_kg=26.0,
               part_role="access-trunk-leg-foot", fastening="bolted")
        g.edge(f"trunk.leg.{k}.col", f"trunk.shell.2.{i}", foot,
               "rigid-distance", radius=0.048, alloy="6061t6",
               palette="chassis-grey", beam_solvable=True,
               fastening="bolted",
               load_path="the-trunk-opening-into-four-columns")
        g.edge(f"trunk.leg.{k}.tie", foot,
               f"trunk.shell.2.{(i + 2) % _TRUNK_N}",
               "rigid-distance", radius=0.030, alloy="6061t6",
               palette="chassis-grey", fastening="bolted",
               load_path="leg-braced-back-up-to-the-trunk")
    for k in range(len(_LEG_AT)):
        g.edge(f"trunk.leg_ring.{k}", f"trunk.leg.{k}",
               f"trunk.leg.{(k + 1) % len(_LEG_AT)}", "rigid-distance",
               radius=0.030, alloy="6061t6", palette="chassis-grey",
               fastening="bolted",
               load_path="the-four-feet-closed-into-a-square")
    g.assembly = "access-ladder"
    _lad_a = 2.0 * math.pi * 10 / _TRUNK_N
    _lad_x, _lad_z = _trunk_r * math.cos(_lad_a), _trunk_r * math.sin(_lad_a)
    _rungs = 7
    for k in range(_rungs):
        f = k / (_rungs - 1)
        y = _floor_y_bay + (_trunk_top - _floor_y_bay) * f
        g.node(f"ladder.rung.{k}", [_lad_x * 0.93, y, _lad_z * 0.93],
               "chassis-load-node", material="6061t6",
               mass_in_total=False, mass_kg=2.4,
               part_role="access-ladder-rung", fastening="bolted",
               half_extent_m=(0.20, 0.016, 0.02))
        if k:
            g.edge(f"ladder.stringer.{k}", f"ladder.rung.{k - 1}",
                   f"ladder.rung.{k}", "rigid-distance", radius=0.018,
                   alloy="6061t6", palette="chassis-grey",
                   fastening="bolted", load_path="ladder-stringer")
        if k in (0, _rungs - 1):
            # THE TOP AND BOTTOM RUNGS ARE THE ANCHORS. A stringer is a
            # chain and a chain needs both ends held; the middle rungs
            # are carried by the stringer, the ends are not.
            g.edge(f"ladder.anchor.{k}", f"ladder.rung.{k}",
                   f"trunk.shell.{0 if k else 2}.9", "rigid-distance",
                   radius=0.016, alloy="6061t6", palette="rollbar-silver",
                   fastening="bolted", load_path="ladder-end-anchored")
        _lvl = 0 if f > 0.66 else (1 if f > 0.33 else 2)
        g.edge(f"ladder.tie.{k}", f"ladder.rung.{k}",
               f"trunk.shell.{_lvl}.10", "rigid-distance", radius=0.014,
               alloy="6061t6", palette="rollbar-silver", fastening="bolted",
               load_path="ladder-hung-off-the-trunk")
    # ---- THE PILLARS THAT HOLD THE CEILING ----
    # The bottom annulus is the room's ceiling and its OUTER rim is the
    # far end of a three-metre cantilever from the trunk. Pillars carry
    # that rim straight down, in the middle of the room, which is where
    # the load is and not where it is convenient.
    g.assembly = "deck-pillars"
    _PILLARS = 8
    for k in range(_PILLARS):
        i = int(round(k * RING_N / _PILLARS)) % RING_N
        a = 2.0 * math.pi * k / _PILLARS
        base = f"deck.pillar.{k}"
        g.node(base, [_drum_r * math.cos(a), _floor_y_bay,
                      _drum_r * math.sin(a)],
               "structural-body-pin-frame-foot", material="steel-plate",
               mass_in_total=False, mass_kg=34.0,
               part_role="lower-deck-pillar-foot", fastening="bolted")
        g.edge(f"deck.pillar.{k}.col", base, drum["lower"]["outer"][i],
               "rigid-distance", radius=0.056, alloy="a36",
               palette="chassis-grey", beam_solvable=True,
               fastening="bolted",
               load_path="pillar-carrying-the-ceiling's-outer-rim")
    # A COLUMN STANDING ON NOTHING IS A PENDULUM. Eight pillars each
    # with one connection were exactly that: perfectly good in
    # compression and free to fall over in any direction, which a
    # static solve will happily report as a very soft structure rather
    # than as the mechanism it is. They are tied into a ring at the
    # floor and braced back to the trunk, so the load has somewhere to
    # go sideways as well as down.
    for k in range(_PILLARS):
        g.edge(f"deck.pillar_ring.{k}", f"deck.pillar.{k}",
               f"deck.pillar.{(k + 1) % _PILLARS}", "rigid-distance",
               radius=0.034, alloy="a36", palette="chassis-grey",
               beam_solvable=True, fastening="bolted",
               load_path="pillar-feet-tied-into-a-ring-on-the-bay-floor")
        g.edge(f"deck.pillar_brace.{k}", f"deck.pillar.{k}",
               f"trunk.shell.2.{int(round(k * _TRUNK_N / _PILLARS)) % _TRUNK_N}",
               "rigid-distance", radius=0.028, alloy="a36",
               palette="chassis-grey", fastening="bolted",
               load_path="pillar-braced-back-to-the-trunk")
    g.assembly = "turntable-floor"
    g.motion_group = "traverse"

    # ---- THE FLOOR RIM IS THE DRUM'S OWN TOP ANNULUS ----
    # Not a separate ring of nodes tied to it: the same plate. A truss
    # root that lands on the drum lands on structure, not on a fitting.
    g.assembly = "turntable-floor"
    g.motion_group = "traverse"
    for i in range(_n_ring if False else RING_N):
        g.edge(f"turret.floor.to_ring.{i}", drum["upper"]["mean"][i],
               f"turret.yaw.{i}", "rigid-distance",
               radius=0.046, palette="rollbar-silver",
               load_path="the-drum-and-the-yaw-ring-are-one-body")
    _n_ring = RING_N
    _rim_of = drum["upper"]["mean"]
    #: the truss depth the overhang needs, and where the beams sit
    #: across the case shaft
    FLOOR_DEPTH_M = 0.550
    FLOOR_BEAM_X = CASE_CHUTE_HALF_W + 0.28

    # ---- AND THE FLOOR REACHES FORWARD UNDER THE GUN ----
    # Not a cantilever of one member: a truss with real depth, because
    # 171 kN.m at the root through a 0.55 m depth is 311 kN in the
    # chords, and 311 kN is a section rather than a stick.
    _bays = 5
    _z0, _z1 = -_drum_r * 0.7, TRUNNION_Z
    for side, sx in (("L", -1.0), ("R", 1.0)):
        for k in range(_bays + 1):
            f = k / _bays
            z = _z0 + (_z1 - _z0) * f
            for chord, yy in (("top", _floor_y), ("bot", _floor_y - FLOOR_DEPTH_M)):
                g.node(f"turret.floor.{chord}.{side}.{k}",
                       [sx * FLOOR_BEAM_X, yy, z],
                       "load-bearing-structure", material="steel-plate",
                       mass_in_total=False, mass_kg=26.0,
                       part_role=f"turntable-floor-{chord}-chord",
                       half_extent_m=(0.070, 0.070, 0.070))
            g.edge(f"turret.floor.post.{side}.{k}",
                   f"turret.floor.top.{side}.{k}",
                   f"turret.floor.bot.{side}.{k}", "rigid-distance",
                   radius=0.034, palette="rollbar-silver",
                   load_path="floor-truss-post")
            if k:
                for chord, r in (("top", 0.045), ("bot", 0.045)):
                    g.edge(f"turret.floor.{chord}_run.{side}.{k}",
                           f"turret.floor.{chord}.{side}.{k - 1}",
                           f"turret.floor.{chord}.{side}.{k}",
                           "rigid-distance", radius=r,
                           palette="rollbar-silver", beam_solvable=True,
                           load_path="floor-truss-chord-carrying-the-overhang")
                g.edge(f"turret.floor.diag.{side}.{k}",
                       f"turret.floor.bot.{side}.{k - 1}",
                       f"turret.floor.top.{side}.{k}", "rigid-distance",
                       radius=0.030, palette="rollbar-silver",
                       load_path="floor-truss-diagonal")
        # the truss roots into the rim ring, not into thin air
        for k in range(2):
            g.edge(f"turret.floor.root.{side}.{k}",
                   f"turret.floor.top.{side}.{k}",
                   _rim_of[6 if k == 0 else (5 if sx < 0 else 7)],
                   "rigid-distance", radius=0.040, palette="rollbar-silver",
                   load_path="floor-truss-welded-to-the-drum's-top-annulus")
    # PAIRED ACROSS, but never through the case shaft: the one volume in
    # this machine that has to stay empty runs z -0.12..0.05, so the
    # cross ties start forward of it.
    for k in range(_bays + 1):
        f = k / _bays
        z = _z0 + (_z1 - _z0) * f
        if -0.20 < z < 0.12:
            continue
        for chord in ("top", "bot"):
            g.edge(f"turret.floor.cross.{chord}.{k}",
                   f"turret.floor.{chord}.L.{k}",
                   f"turret.floor.{chord}.R.{k}", "rigid-distance",
                   radius=0.032, palette="rollbar-silver",
                   clears="turret.case_ejection_column",
                   load_path="floor-truss-cross-tie")

    # ---- THE COUNTERWEIGHT, ON THE BACK OF THE PLATFORM ----
    # The gun is five tonnes hanging four metres forward of the traverse
    # axis, so the turntable is not a balanced body and never will be:
    # the trunnion balances the gun about its OWN pivot, which says
    # nothing about where the turret's mass sits relative to the axis it
    # turns on. Slewing an unbalanced turntable throws that offset
    # around as a rotating load, and it is the load case the drum, the
    # races and the drive actually live with.
    #
    # A counterweight on the back rim does not remove that -- it is not
    # meant to. It moves the centre of mass back toward the axis so the
    # rotating offset is something the structure can carry, while
    # leaving enough of it that the machine still has to work for a
    # living. Put on the RIM because the lever is longest there, so the
    # least dead weight does the most good.
    g.assembly = "counterweight"
    g.motion_group = "traverse"
    COUNTERWEIGHT_KG = 1400.0
    _cw_z = -_drum_r * 0.92
    for k in range(3):
        _x = (k - 1) * _drum_r * 0.42
        g.node(f"turret.counterweight.{k}",
               [_x, _floor_y + 0.10, _cw_z],
               "load-bearing-structure", material="steel-plate",
               mass_in_total=False, mass_kg=COUNTERWEIGHT_KG / 3.0,
               part_role="turntable-counterweight", fastening="bolted",
               half_extent_m=(0.28, 0.16, 0.22))
        _near = int(round((math.atan2(_cw_z, _x) % (2 * math.pi))
                          / (2 * math.pi) * RING_N)) % RING_N
        for d in (0, 1):
            g.edge(f"turret.counterweight_seat.{k}.{d}",
                   f"turret.counterweight.{k}",
                   drum["upper"]["mean"][(_near + d) % RING_N],
                   "rigid-distance", radius=0.046, palette="rollbar-silver",
                   fastening="bolted",
                   load_path="counterweight-bolted-to-the-back-of-the-drum")
        if k:
            g.edge(f"turret.counterweight_tie.{k}",
                   f"turret.counterweight.{k - 1}",
                   f"turret.counterweight.{k}", "rigid-distance",
                   radius=0.034, palette="rollbar-silver",
                   fastening="bolted",
                   load_path="counterweight-blocks-tied-across")

    # ---- THE TRUNNION STANDS ON THE END OF ITS OWN FLOOR ----
    g.assembly = "gun-support-structure"
    for side in ("L", "R"):
        g.edge(f"turret.standard.{side}", f"turret.floor.top.{side}.{_bays}",
               "turret.pitch", "rigid-distance", radius=0.052,
               palette="rollbar-silver",
               clears="turret.case_ejection_column",
               load_path="trunnion-standard-on-the-floor-that-reaches-it")
        g.edge(f"turret.standard_brace.{side}",
               f"turret.floor.top.{side}.{_bays - 1}", "turret.pitch",
               "rigid-distance", radius=0.038, palette="rollbar-silver",
               load_path="standard-braced-back-along-the-floor")

    # WHAT MOVES WHEN THE GUN ELEVATES, said rather than inherited.
    # `g.motion_group` is sticky by design -- a builder sets it once
    # and every body in the block says so for itself -- which makes a
    # block that forgets to put it back a trap. The turntable block
    # ends in "traverse", so the cradle, the elevation lug, the
    # anchor and the trunnion were all authored as traverse: the
    # elevation bearing then joined traverse to traverse, which is a
    # bearing between two halves of ONE body, and the gun could not
    # elevate at all. Nothing complained, because a joint inside a
    # body is perfectly legal -- it simply does nothing.
    g.motion_group = "elevation"
    cradle_y = carriage_y + 0.44
    g.assembly = "cradle"
    g.node("turret.cradle", [0.0, cradle_y, TRUNNION_Z + 0.18], "load-bearing-structure",
           material="steel-plate", half_extent_m=(0.13, 0.10, 0.30))
    # THE ELEVATING GEAR GOES DOWN ONE SIDE. On the bore axis its lug,
    # its brace and its ram all crossed the case ejection column, which
    # is the one volume in this machine that has to stay empty. Real
    # mounts put the elevating arc or ram off to the side of the
    # cradle for exactly this reason -- the middle under a gun is where
    # the brass goes.
    _lug_x = CASE_CHUTE_HALF_W + 0.16
    g.node("turret.elevation_lug", [_lug_x, cradle_y - 0.26, TRUNNION_Z - 0.14],
           "chassis-load-node", off_the_centreline_for="turret.case_ejection_column",
           material="steel-plate", half_extent_m=(0.05, 0.05, 0.05))
    g.motion_group = "traverse"
    g.node("turret.elevation_anchor", [_lug_x, cradle_y - 0.47, TRUNNION_Z + 0.22], "chassis-load-node",
           material="steel-plate", half_extent_m=(0.06, 0.06, 0.06))
    g.edge("turret.cradle_to_pitch", "turret.cradle", "turret.pitch", "rigid-distance",
           radius=0.038, palette="rollbar-silver")
    g.edge("turret.lug_to_cradle", "turret.elevation_lug", "turret.cradle", "rigid-distance",
           radius=0.026, palette="rollbar-silver")
    g.edge("turret.lug_brace", "turret.elevation_lug", "turret.pitch", "rigid-distance",
           radius=0.022, palette="rollbar-silver")
    g.edge("turret.anchor_to_yaw", "turret.elevation_anchor",
           f"turret.yaw.{_ring_near(0.0, ring_radius)}", "rigid-distance",
           radius=0.030, palette="rollbar-silver")
    g.edge("turret.anchor_brace", "turret.elevation_anchor", "turret.race.pn", "rigid-distance",
           radius=0.022, palette="rollbar-silver")
    g.edge("turret.elevation_ram", "turret.elevation_anchor", "turret.elevation_lug",
           "linear-hydraulic-actuator", radius=0.032, palette="actuator-yellow",
           frame_mount=True, family="linear-hydraulic-actuator",
           kind="commanded-rest-length-elevation-ram",
           commanded_rest_length_m=0.0, minimum_rest_length_m=0.0,
           maximum_rest_length_m=0.320, bore_m=0.063, rod_m=0.036,
           linear_stiffness_n_per_m=9.0e6, linear_damping_n_s_per_m=3.2e4,
           holding_force_n=48_000.0, relief_force_n=70_000.0,
           maximum_relief_stroke_m=0.012, elevation_range_deg=[-8.0, 42.0],
           energy_law="commanded-length-plus-series-relief; the equilibrator carries "
                      "the barrel moment so this carries only the difference")
    g.edge("turret.equilibrator", "turret.elevation_anchor", "turret.cradle",
           "spring-damper", radius=0.026, palette="actuator-yellow",
           stiffness_n_per_m=8_000.0, preload_force_n=400.0,
           compression_damping_n_s_per_m=300.0, rebound_damping_n_s_per_m=420.0,
           load_path="trims-whatever-small-moment-the-cradle-and-fine-stage-still-carry")
    g.edge("turret.gimbal.pitch", "turret.pitch", "turret.cradle", "gimbal-pitch-bearing",
           radius=0.045, driven_by="turret.elevation_ram", elevation_range_deg=[-8.0, 42.0],
           load_path="trunnion-pin-carries-the-cradle")

    # ---- MECHANISM 3: THE SIX-ACTUATOR FINE-AIM PLATFORM ----
    # A small Stewart platform slung under the cradle, at the breech.
    # Six short, fast hydraulic legs between a base ring fixed to the
    # cradle and a platform ring that carries the recoil slide -- the
    # same six-leg, six-freedom geometry proven this session
    # (armature.py's hexapod, 1e-6 rad error in single-digit iterations)
    # gives genuine sub-degree "twitch" correction without moving the
    # heavy ring or trunnion at all.
    # THE PLATFORM PICKS THE GUN UP AT THE BREECH'S REAR PORT.
    #
    # ---- EVERYTHING MEETS ON THE FLAT BOTTOM OF THE BREECH ----
    # The rear face was the right END of the gun and the wrong FACE of
    # it. A bolt circle on the back disc puts hardware directly in the
    # path of the blast panel vent and directly in the way of a case
    # coming out, and it hangs the whole fine stage off a face that is
    # trying to move aft under recoil.
    #
    # The belly is where a gun is held. All six pickups land on the
    # machined flat, in two rails either side of a clear chute, so:
    #   - the load into the gun is a plate pressing up on a flat,
    #     which is what the flat is for;
    #   - the middle stays open and a spent case falls straight down
    #     through the mounting plate and out;
    #   - feed comes in over the top, which nothing now obstructs;
    #   - and the legs get the full depth of the well to work in.
    plate_thk = 0.045
    pad_x = (CASE_CHUTE_HALF_W + BREECH_FLAT_HALF_W) / 2.0
    assert CASE_CHUTE_HALF_W < pad_x < BREECH_FLAT_HALF_W, (
        "the pads live between the chute edge and the edge of the "
        "machined flat: inboard of that and they are over the hole, "
        "outboard and there is no flat under them")
    plate_y = BREECH_FLAT_Y - plate_thk
    # three stations along the belly; the aft bay between the first two
    # is the chute and stays open
    pad_z = (CASE_CHUTE_AFT_Z, CASE_CHUTE_FWD_Z,
             BREECH_REAR_Z + breech_length_m * 0.74)
    # THE BASE HANGS BELOW THE RING, ON A BASKET, reaching down into the
    # well. With the wider ring pair there is real depth to work in, so
    # the legs are long enough to have authority and to allow long
    # recoil above them.
    base_half_w = ring_radius * 0.40
    base_y = race_y - 0.330
    for k, zs in enumerate(pad_z):
        for sd, sgn in (("L", -1.0), ("R", 1.0)):
            g.motion_group = "traverse"
            g.assembly = "fine-rig-base"
            # THE STAGGER GOES THE SAME WAY ON BOTH SIDES, so a cross
            # tie at a station is entirely aft of the chute or entirely
            # forward of it. Staggered L one way and R the other, every
            # tie ran diagonally across the open bay -- six braces
            # through the one column a case has to fall down.
            g.node(f"fine.base.{k}{sd}",
                   [sgn * base_half_w, base_y, zs + (-0.085 if k == 0 else 0.085)],
                   "chassis-load-node", material="hardened-steel",
                   half_extent_m=(0.026, 0.026, 0.026))
            g.motion_group = "fine"
            g.assembly = "fine-rig-platform"
            g.node(f"fine.plate.{k}{sd}", [sgn * pad_x, plate_y, zs],
                   "chassis-load-node", material="hardened-steel",
                   half_extent_m=(0.028, plate_thk / 2.0, 0.028))
    # the basket: hangs off the slew ring and braces to the race pads,
    # so it turns with the mount and is a frame rather than six points
    g.assembly = "fine-rig-base"
    # HUNG FROM THE RACE PADS, NOT FROM THE CENTRE. Every hanger used to
    # run to `turret.yaw`, one node on the bore axis -- so six braces
    # converged on precisely the column a spent case falls down, and the
    # ejection path was blocked by the thing holding the ejector.
    # The race pads are out at radius where the ring actually is, which
    # is both where the load wants to go and out of the way.
    # NEAREST PAD, not a pad picked by counting. Three of the six
    # hangers reached across the machine to a pad on the far side and
    # crossed the ejection shaft doing it.
    _pads = {n: (sx * ring_radius * 0.72, sz * ring_radius * 0.72)
             for n, sx, sz in corners}
    for k, zs in enumerate(pad_z):
        for sd, sgn in (("L", -1.0), ("R", 1.0)):
            bx = sgn * base_half_w
            bz = zs + (-0.085 if k == 0 else 0.085)
            near = min(_pads, key=lambda n: (_pads[n][0] - bx) ** 2
                       + (_pads[n][1] - bz) ** 2)
            g.edge(f"fine.base_to_platform.{k}{sd}", f"fine.base.{k}{sd}",
                   f"turret.race.{near}", "rigid-distance",
                   radius=0.026, palette="rollbar-silver",
                   load_path="basket-hung-from-the-nearest-race-pad")
    for k, pad in enumerate(("nn", "np", "pp")):
        g.edge(f"fine.basket_brace.{k}", f"fine.base.{k}L", f"turret.race.{pad}",
               "rigid-distance", radius=0.020, palette="rollbar-silver",
               load_path="basket-braced-to-the-race-so-it-is-a-frame")

    g.assembly = "fine-rig-mount"
    # ---- THE PLATE BOLTS TO THE FLAT THROUGH REAL PORTS ----
    # `assembly_ports` is the project own mechanism for this: a port is
    # a point on a part SURFACE with an outward normal, derived from
    # that part geometry; `mate_ports` pairs facing ports on different
    # parts; a mated pair is ZERO LENGTH, because both ends are on the
    # two faces that touch. That is why a ported joint cannot reach
    # into the middle of a body.
    from assembly_ports import PartPort, mate_ports, emit_ports_graph
    DOWN = np.array([0.0, -1.0, 0.0])
    # the bore sweeps in the plane spanned by the bore axis and
    # vertical, so rotation IN that plane is rotation ABOUT the
    # transverse horizontal axis -- the trunnion axis
    TRUNNION_AXIS = (1.0, 0.0, 0.0)
    mount_ports_list = []
    for k, zs in enumerate(pad_z):
        for sd, sgn in (("L", -1.0), ("R", 1.0)):
            # the same point reached from each part own dimensions: the
            # breech machined flat, and the plate top face
            at = np.array([sgn * pad_x, BREECH_FLAT_Y, zs])
            top_of_plate = np.array([sgn * pad_x, plate_y + plate_thk, zs])
            assert np.linalg.norm(at - top_of_plate) < 1e-9, (at, top_of_plate)
            for ident, part, facing in (
                    (f"turret.breech.pad.{k}{sd}", "turret.breech", DOWN),
                    (f"fine.plate.{k}{sd}.pad", f"fine.plate.{k}{sd}", -DOWN)):
                mount_ports_list.append(PartPort(
                    ident, part, "structural-mount", at.copy(), facing,
                    0.026, True, fluid="structure",
                    joint="pinned-clevis", joint_axis=TRUNNION_AXIS))
    _mated = mate_ports(mount_ports_list)
    emit_ports_graph(
        mount_ports_list, _mated,
        lambda ident, pos, kind, **kw: g.node(
            ident, pos, "structural-mount-port", motion_group="fine",
            recoils=True, material="hardened-steel",
            half_extent_m=(0.026, 0.012, 0.026), **kw),
        # the constraint is whatever the PORT declared; passing one in
        # here would be the caller overruling the hardware
        lambda ident, a_, b_, constraint, **kw: g.edge(
            ident, a_, b_, constraint, palette="rollbar-silver",
            load_path="plate-pressing-up-on-the-breech-machined-flat",
            **{k2: v2 for k2, v2 in kw.items()
               if k2 not in ("circuit_identity", "gasket")}),
        prefix="turret")
    # A PAD IS PART OF ITS CASTING. Each port webs into its own body and
    # along to the next pad on the same rail, which is the strip of
    # metal a real mounting flat is machined as.
    for k in range(3):
        for sd in ("L", "R"):
            g.edge(f"turret.breech_pad_web.{k}{sd}",
                   f"turret.turret.breech.pad.{k}{sd}", "turret.breech",
                   "rigid-distance", radius=0.030, palette="rollbar-silver",
                   load_path="pad-is-part-of-the-breech-body")
            g.edge(f"fine.plate_pad_web.{k}{sd}",
                   f"turret.fine.plate.{k}{sd}.pad", f"fine.plate.{k}{sd}",
                   "rigid-distance", radius=0.022, palette="rollbar-silver",
                   load_path="pad-is-part-of-the-mounting-plate")
        if k < 2:
            for sd in ("L", "R"):
                g.edge(f"turret.breech_flat_rail.{k}{sd}",
                       f"turret.turret.breech.pad.{k}{sd}",
                       f"turret.turret.breech.pad.{k + 1}{sd}",
                       "rigid-distance", radius=0.024, palette="rollbar-silver",
                       load_path="the-machined-flat-the-pads-are-cut-into")
                # and the plate's own pads are a rail too: the bolted
                # face on the plate side is one machined strip, not six
                # loose bosses each hanging on a single stud
                g.edge(f"fine.plate_pad_rail.{k}{sd}",
                       f"turret.fine.plate.{k}{sd}.pad",
                       f"turret.fine.plate.{k + 1}{sd}.pad",
                       "rigid-distance", radius=0.020, palette="rollbar-silver",
                       load_path="the-plate-machined-mounting-strip")

    g.assembly = "fine-rig-platform"
    # ---- THE MOUNTING PLATE ----
    # Side rails down both edges, cross ties at the ends and at the
    # forward edge of the chute, and diagonals ONLY in the forward bay.
    # Nothing crosses the aft bay, because that is the hole a case
    # falls through; a plate braced across it would be a plate with no
    # ejection path, which is how this would quietly stop working.
    for k in range(2):
        for sd in ("L", "R"):
            g.edge(f"fine.plate_rail.{k}{sd}", f"fine.plate.{k}{sd}",
                   f"fine.plate.{k + 1}{sd}", "rigid-distance",
                   radius=0.018, palette="rollbar-silver",
                   load_path="mounting-plate-side-rail")
    for k in range(3):
        g.edge(f"fine.plate_tie.{k}", f"fine.plate.{k}L", f"fine.plate.{k}R",
               "rigid-distance", radius=0.018, palette="rollbar-silver",
               load_path="mounting-plate-cross-tie"
                         + ("-at-the-aft-lip-of-the-chute" if k == 0 else ""))
    for a_, b_ in (("1L", "2R"), ("1R", "2L")):
        g.edge(f"fine.plate_diag.{a_}_{b_}", f"fine.plate.{a_}",
               f"fine.plate.{b_}", "rigid-distance", radius=0.014,
               palette="rollbar-silver",
               load_path="forward-bay-diagonal-clear-of-the-case-chute")

    # ---- THE SIX ACTUATORS ----
    # One per pad, straight up the well from the basket to the plate.
    # They carry no firing load: the thrust seat takes the shot and
    # these sit in parallel with it.
    for k in range(3):
        for sd in ("L", "R"):
            g.edge(f"fine.leg.{k}{sd}", f"fine.base.{k}{sd}",
                   f"fine.plate.{k}{sd}",
                   "linear-hydraulic-actuator", radius=0.020,
                   palette="actuator-yellow",
                   family="linear-hydraulic-actuator",
                   kind="fine-aim-twitch-actuator",
                   carries_firing_load=False,
                   bypassed_by="turret.thrust_seat",
                   commanded_rest_length_m=0.0,
                   minimum_rest_length_m=-0.025,
                   maximum_rest_length_m=0.025,
                   bore_m=0.020, rod_m=0.011,
                   linear_stiffness_n_per_m=6.0e6,
                   linear_damping_n_s_per_m=2.4e4,
                   holding_force_n=7_900.0,
                   load_path="fine-aim-lifting-the-gun-off-its-belly-flat")
        g.assembly = "fine-rig-base"
        # cross ties only where the station itself is clear of the
        # chute, and side rails fore-and-aft -- the same rule the plate
        # above obeys, for the same reason
        g.edge(f"fine.base_pair.{k}", f"fine.base.{k}L", f"fine.base.{k}R",
               "rigid-distance", radius=0.012, palette="chassis-grey",
               load_path="basket-cross-tie-clear-of-the-chute")
        if k < 2:
            for sd in ("L", "R"):
                g.edge(f"fine.base_rail.{k}{sd}", f"fine.base.{k}{sd}",
                       f"fine.base.{k + 1}{sd}", "rigid-distance",
                       radius=0.012, palette="chassis-grey",
                       load_path="basket-side-rail")
        g.assembly = "fine-rig-platform"

    # ---- SIZING THE RECOIL SLIDE ----
    slide_stroke_m = max(0.28, bore_mm / 1000.0 * 8.0)
    # the recoiling assembly is tube PLUS collar PLUS breech, which is
    # what actually slides -- an earlier version used the tube and a guess
    # WHAT ACTUALLY SLIDES: the outer barrel, whatever liner is in it,
    # the collar and the breech. `barrel_mass_kg` is an estimate of a
    # barrel for the calibre and not the mass of anything in this graph.
    slide_mass_kg = round(647.2 + _liner_mass_kg + collar_mass_kg
                          + breech_mass_kg + 14.0, 1)
    # THE RECOIL PISTON IS SIZED TO THE GUN, not held at one diameter for
    # every bore. A 40 mm mount's recoil cylinders are about 95 mm; a
    # 120 mm gun's are twice that. Scaling the orifice off the BORE
    # against a piston that never changed made the damping nonsense at
    # both ends of the range, the square law turning every factor of
    # error into its square.
    slide_piston_bore_m = max(0.060, bore_mm / 1000.0 * 2.4)
    # the shot to absorb: shell plus propellant gas, which leaves at
    # about 1.5x muzzle velocity
    charge_kg = cal.mass_kg * 0.36
    shot_impulse = cal.mass_kg * cal.muzzle_m_s + charge_kg * cal.muzzle_m_s * 1.5
    # ONE PIECE OF ARITHMETIC, in actuators, used by everything that
    # sizes a recoil system -- the builder does not get its own opinion
    slide = _size_recoil_slide(impulse_n_s=shot_impulse,
                               recoiling_mass_kg=slide_mass_kg,
                               stroke_m=slide_stroke_m,
                               piston_bore_m=slide_piston_bore_m,
                               peak_force_n=recoil_peak_force_n)
    # THE CRADLE IS WHAT THE COLLAR SLIDES THROUGH: the trunnion carries
    # the cradle, and the tube and its collar run in the cradle's ways.
    g.assembly = "recoil-gear"
    g.edge("turret.collar_guide", "turret.collar", "turret.cradle",
           "single-axis-slider", radius=0.030, palette="rollbar-silver",
           slide_axis=(0.0, 0.0, 1.0),
           load_path="collar-rides-the-cradle-ways-and-takes-the-side-loads")

    # THE SLIDE IS BETWEEN THE RECOILING BODY AND THE CRADLE IT RUNS IN.
    # It ran from the fine platform's hub to `turret.weapon` -- the
    # BARREL'S CENTROID, 1932 mm away in the middle of a 2.8 m tube --
    # which is not where a recoil slide is and not what it connects. The
    # breech is the back of the recoiling assembly; the cradle is what it
    # slides through. Those two, and nothing else.
    # ============ THREE PIECES AROUND ONE JOURNAL ============
    # What was here: three 300 mm carriages on the tank's bottom face,
    # and an absorber in its own 218 mm casing slung underneath. Two
    # bodies, six discrete bearing patches, and a trunnion 1.6 m behind
    # the nearest end of the rail.
    #
    # What is here now is one stack, and the order of the pieces is the
    # whole design:
    #
    #   THE MIDDLE PIECE IS THE TANK, and it recoils. Its entire 305 mm
    #   square outside is bearing surface -- not three patches on one
    #   face, the whole section, all four faces, the full length. That
    #   is what buys the layout: see below.
    #
    #   THE UPPER AND LOWER SHOES DO NOT RECOIL. They are journal pads
    #   closed around the tank, and the tank slides through them.
    #
    #   A STRUT TIES THEM THROUGH THE TANK. It passes up a 90 mm channel
    #   bored the length of the section, on a slot long enough for the
    #   whole stroke, so the two shoes are one closed body around a part
    #   that is moving through them. Without it the lower shoe is a hook
    #   and the upper shoe is a separate hook, and the pair can only
    #   pinch as hard as their own bending allows.
    #
    #   THE TRUNNION HANGS OFF THE LOWER SHOE. Which puts the pivot 440
    #   mm below the bore and makes full recoil a 48 kN.m couple -- six
    #   times what a pivot at the collar had. THAT IS THE COST THE
    #   JOURNAL PAYS FOR: 48 kN.m over a two-metre engagement is a 24 kN
    #   pair at 0.26 MPa of bearing pressure. Distributing the bearing
    #   is not a refinement here, it is the thing that makes hanging the
    #   trunnion underneath legal at all.
    RECOIL_STROKE_M = 0.844
    JOURNAL_LENGTH_M = 2.00
    g.assembly = "recoil-journal"
    g.motion_group = "elevation"          # the shoes do NOT recoil
    _tank_half = spine_side_m / 2.0
    _journal_z = TRUNNION_Z               # the stack is AT the pivot
    _shoe_t = 0.075
    _pad_mass = round(spine_side_m * JOURNAL_LENGTH_M * _shoe_t * 7850.0, 1)
    for tag, sign in (("lower", -1.0), ("upper", +1.0)):
        g.node(f"turret.journal.{tag}",
               [0.0, spine_y + sign * (_tank_half + _shoe_t / 2.0), _journal_z],
               "load-bearing-structure", material="hardened-steel",
               mass_in_total=False, mass_kg=_pad_mass,
               part_role=f"recoil-journal-{tag}-shoe",
               recoils=False,
               bearing_length_m=JOURNAL_LENGTH_M,
               bearing_width_m=spine_side_m,
               bearing_area_m2=round(JOURNAL_LENGTH_M * spine_side_m, 4),
               rides_on="turret.spine",
               half_extent_m=(_tank_half, _shoe_t / 2.0,
                              JOURNAL_LENGTH_M / 2.0))
        # THE BEARING ITSELF: free along the bore, rigid in everything
        # else, declared through the joint registry like any other slide.
        for k, frac in enumerate((-0.38, 0.0, 0.38)):
            _z = _journal_z + JOURNAL_LENGTH_M * frac
            _mod = min(max(int(round((_z - spine_z0) / SPINE_SECTION_M)), 0),
                       spine_modules)
            g.edge(f"turret.journal.{tag}.bear.{k}", f"turret.journal.{tag}",
                   f"turret.spine.{_mod}", "single-axis-slider",
                   radius=_tank_half * 0.7, palette="chassis-grey",
                   free_axis="bore", slide_axis=(0.0, 0.0, 1.0),
                   travel_m=RECOIL_STROKE_M,
                   bearing_pressure_pa=round(
                       48_000.0 / max(JOURNAL_LENGTH_M, 0.1)
                       / (JOURNAL_LENGTH_M * spine_side_m), 0),
                   load_path="the-tank-running-in-its-journal")
    # THE SIDE FACES BEAR TOO, which is what stops the stack rolling on
    # the section it is wrapped around.
    for k, sx in enumerate((-1.0, 1.0)):
        g.node(f"turret.journal.side.{k}",
               [sx * (_tank_half + _shoe_t / 2.0), spine_y, _journal_z],
               "load-bearing-structure", material="hardened-steel",
               mass_in_total=False, mass_kg=_pad_mass,
               part_role="recoil-journal-side-shoe", recoils=False,
               bearing_length_m=JOURNAL_LENGTH_M,
               half_extent_m=(_shoe_t / 2.0, _tank_half,
                              JOURNAL_LENGTH_M / 2.0))
        for tag in ("lower", "upper"):
            g.edge(f"turret.journal.side_tie.{k}.{tag}",
                   f"turret.journal.side.{k}", f"turret.journal.{tag}",
                   "rigid-distance", radius=0.032, palette="rollbar-silver",
                   load_path="the-four-pads-closed-into-one-shoe")
        _mod = min(max(int(round((_journal_z - spine_z0) / SPINE_SECTION_M)),
                       0), spine_modules)
        g.edge(f"turret.journal.side.bear.{k}", f"turret.journal.side.{k}",
               f"turret.spine.{_mod}", "single-axis-slider",
               radius=_tank_half * 0.5, palette="chassis-grey",
               free_axis="bore", slide_axis=(0.0, 0.0, 1.0),
               travel_m=RECOIL_STROKE_M,
               load_path="side-journal-stopping-the-stack-rolling")
    # THE STRUT UP THE CHANNEL. Bottom piece to top piece, through the
    # middle piece, on a slot as long as the stroke.
    g.node("turret.journal.strut", [0.0, spine_y, _journal_z],
           "load-bearing-structure", material="4340qt",
           mass_in_total=False,
           mass_kg=round(math.pi * (TANK_CHANNEL_M / 2.0) ** 2
                         * (spine_side_m + _shoe_t) * 7850.0, 1),
           part_role="journal-clamp-strut", recoils=False,
           diameter_m=TANK_CHANNEL_M,
           slot_length_m=RECOIL_STROKE_M,
           note="it passes THROUGH the recoiling tank; the slot is what "
                "lets the tank move while the strut does not",
           half_extent_m=(TANK_CHANNEL_M / 2.0, _tank_half + _shoe_t,
                          TANK_CHANNEL_M / 2.0))
    for tag in ("lower", "upper"):
        g.edge(f"turret.journal.strut_{tag}", "turret.journal.strut",
               f"turret.journal.{tag}", "rigid-distance",
               radius=TANK_CHANNEL_M / 2.0, alloy="4340qt",
               palette="rollbar-silver",
               load_path="strut-closing-the-shoe-around-the-tank")
    g.edge("turret.journal.strut_slot", "turret.journal.strut",
           f"turret.spine.{min(max(int(round((_journal_z - spine_z0) / SPINE_SECTION_M)), 0), spine_modules)}",
           "single-axis-slider", radius=TANK_CHANNEL_M / 2.0,
           palette="chassis-grey", free_axis="bore",
           slide_axis=(0.0, 0.0, 1.0), travel_m=RECOIL_STROKE_M,
           load_path="the-slot-the-strut-rides-in")

    # TWO REAR STOPS, symmetric about the bore.  The moving strikers are
    # part of the square tank and the buffers are carried by the closed
    # journal shoe.  Their 844 mm open gap is the allowed recoil travel;
    # no force exists before it closes.  At the end of travel the pair
    # shares the reaction without putting a roll moment into the slide.
    _stop_z = _journal_z - RECOIL_STROKE_M / 2.0
    _striker_z = _journal_z + RECOIL_STROKE_M / 2.0
    for side, sx in (("left", -0.62), ("right", +0.62)):
        x = sx * _tank_half
        fixed = f"turret.bump_stop.{side}.buffer"
        moving = f"turret.bump_stop.{side}.striker"
        g.assembly = "recoil-gear"
        g.motion_group = "elevation"
        g.node(fixed, [x, spine_y, _stop_z], "recoil-bump-stop-buffer",
               material="polyurethane", mass_in_total=False, mass_kg=2.4,
               part_role="rear-recoil-bump-stop", recoils=False,
               half_extent_m=(0.042, 0.042, 0.030))
        g.edge(f"{fixed}.seat", fixed, "turret.journal.lower",
               "rigid-distance", radius=0.024, alloy="4340qt",
               palette="chassis-grey",
               load_path="rear-stop-buffer-into-the-closed-journal-shoe")
        g.edge(f"{fixed}.brace", fixed, "turret.journal.upper",
               "rigid-distance", radius=0.018, alloy="4340qt",
               palette="chassis-grey",
               load_path="rear-stop-buffer-braced-across-the-closed-journal-shoe")
        g.motion_group = "fine"
        g.node(moving, [x, spine_y, _striker_z], "recoil-stop-striker",
               material="4340qt", mass_in_total=False, mass_kg=1.8,
               part_role="rear-recoil-stop-striker", recoils=True,
               half_extent_m=(0.046, 0.046, 0.024))
        g.edge(f"{moving}.seat", moving,
               f"turret.spine.{min(max(int(round((_striker_z - spine_z0) / SPINE_SECTION_M)), 0), spine_modules)}",
               "rigid-distance", radius=0.026, alloy="4340qt",
               palette="rollbar-silver",
               load_path="rear-stop-striker-is-part-of-the-recoiling-tank")
        g.edge(f"{moving}.brace", moving,
               f"turret.spine.{min(max(int(round((_striker_z - spine_z0) / SPINE_SECTION_M)) + 1, 0), spine_modules)}",
               "rigid-distance", radius=0.018, alloy="4340qt",
               palette="rollbar-silver",
               load_path="rear-stop-striker-braced-into-the-recoiling-tank")
        g.edge(f"turret.bump_stop.{side}.contact", fixed, moving,
               "bump-stop-contact", radius=0.040,
               palette="actuator-yellow",
               kind=("bump-stop" if side == "left"
                     else "equalized-bump-stop-face"),
               slide_axis=(0.0, 0.0, 1.0),
               coupled_slide_edges=(
                   ("turret.bump_stop.left.contact",
                    "turret.bump_stop.right.contact")
                   if side == "left" else ()),
               slide_load_share=((0.5, 0.5) if side == "left" else ()),
               clearance_m=RECOIL_STROKE_M,
               maximum_compression_m=0.030,
               linear_stiffness_n_per_m=12.0e6,
               cubic_stiffness_n_per_m3=1.2e10,
               compression_damping_n_s_per_m=90_000.0,
               load_path="symmetric-rear-stop-at-the-end-of-slide-travel")

    # ---- THE TRUNNION, ON THE LOWER SHOE ----
    g.assembly = "trunnion-thrust-seat"
    g.node("turret.trunnion.block",
           [0.0, spine_y - _tank_half - _shoe_t - 0.06, _journal_z],
           "load-bearing-structure", material="4340qt",
           mass_in_total=False, mass_kg=96.0,
           part_role="trunnion-block", recoils=False,
           bore_to_pivot_m=round(gun_y - (spine_y - _tank_half - _shoe_t
                                          - 0.06), 4),
           half_extent_m=(_tank_half, 0.060, 0.180))
    # These faces touch at the block top/shoe underside.  Three duplicate
    # centreline beams used to stand in for the joint, tripling stiffness at
    # one point without creating a bolt spacing or weld lever arm.  Put six
    # real filler-metal elements along the two outside fillets instead.
    from fabrication import weld_touching
    from welding import shop_weld
    _weld_throat = 0.008
    _interface_y = spine_y - _tank_half - _shoe_t
    _toe_pairs = []
    for sx in (-1.0, 1.0):
        for dz in (-0.12, 0.0, 0.12):
            _toe_pairs.append((
                (sx * _tank_half, _interface_y - _weld_throat, _journal_z + dz),
                (sx * (_tank_half - _weld_throat), _interface_y,
                 _journal_z + dz)))
    weld_touching(
        g, "turret.trunnion.block_to_lower_shoe",
        "turret.trunnion.block", "turret.journal.lower", _toe_pairs,
        throat_m=_weld_throat, quality=shop_weld("4340qt"),
        restraint=0.85, stress_relieved=True)
    g.edge("turret.trunnion.pin", "turret.trunnion.block", "turret.pitch",
           "pinned-trunnion-mount", radius=0.070, alloy="4340qt",
           palette="rollbar-silver", frame_mount=True,
           part_role="trunnion-pin",
           load_path="the-whole-elevating-mass-through-one-pin-pair")

    # ---- THE ABSORBER, NOW INSIDE THE TANK ----
    # Three elements, three bores, one body, and the reaction goes
    # straight into the shoe that is already wrapped around them.
    g.assembly = "recoil-absorber"
    _recuperator_total_rate = 2.0 * 30934.0 / RECOIL_STROKE_M ** 2
    _absorber = (
        ("spring", "spring-damper", dict(
            spring_rate_n_per_m=_recuperator_total_rate / 3.0,
            force_components=("spring-recuperator",),
            peak_factor=2.0, alone_takes_full_charge=True,
            note="one of three equal recuperator springs around the pistons")),
        ("orifice", "oleo-recoil-slide", dict(
            orifice_area_m2=6.2e-4, peak_factor=3.0,
            force_components=("oil-orifice",),
            alone_takes_full_charge=True,
            note="force with the square of velocity: biggest at the "
                 "start, and it never quite stops the mass on its own")),
        ("magnetorheological", "linear-hydraulic-actuator", dict(
            yield_min_pa=1.0e3, yield_max_pa=6.0e4, gap_m=0.006,
            peak_factor=1.0, controllable=True, coil_watts=22.0,
            kind="magnetorheological", current_frac=1.0,
            force_components=("mr-yield",),
            alone_takes_full_charge=True,
            note="the only one that can be trimmed DOWN, so it is what "
                 "flattens the sum of the other two")),
    )
    _elem_area = math.pi * (ABSORBER_BORE_M / 2.0) ** 2
    for i, (name, constraint, attrs) in enumerate(_absorber):
        ang = math.radians(90.0 + 120.0 * i)
        g.node(f"turret.absorber.piston.{name}",
               [ABSORBER_PITCH_M * math.cos(ang),
                spine_y + ABSORBER_PITCH_M * math.sin(ang),
                spine_z0 + _TANK_BORED_LENGTH - RECOIL_STROKE_M],
               "load-bearing-structure", material="hardened-steel",
               mass_in_total=False, mass_kg=12.0,
               part_role="recoil-absorber-piston", recoils=False,
               element=name, piston_area_m2=round(_elem_area, 6),
               half_extent_m=(ABSORBER_BORE_M / 2.0, ABSORBER_BORE_M / 2.0,
                              0.055))
        # the piston is held by the SHOE, because the shoe is the part
        # that does not recoil -- the tank slides over the pistons
        g.edge(f"turret.absorber.rod.{name}",
               f"turret.absorber.piston.{name}", "turret.journal.lower",
               "rigid-distance", radius=ABSORBER_BORE_M * 0.33,
               alloy="300m", palette="rollbar-silver",
               load_path="piston-rod-anchored-to-the-shoe")
        # THE PISTON'S OWN LAND, which is the third real thing about it
        # and not a third connection found to satisfy a count: a piston
        # bears sideways on the bore it runs in, and that bore is in the
        # tank, which is moving past it.
        g.edge(f"turret.absorber.land.{name}",
               f"turret.absorber.piston.{name}",
               f"turret.spine.{min(1, spine_modules)}", "single-axis-slider",
               radius=ABSORBER_BORE_M / 2.0, palette="chassis-grey",
               free_axis="bore", slide_axis=(0.0, 0.0, 1.0),
               travel_m=RECOIL_STROKE_M,
               load_path="piston-bearing-in-the-bore-it-runs-in")
        # AND THE ELEMENT: the column of fluid between that piston and
        # the closed end of its own bore in the tank.
        g.edge(f"turret.absorber.{name}", f"turret.absorber.piston.{name}",
               "turret.spine.0", constraint,
               radius=ABSORBER_BORE_M / 2.0, palette="actuator-yellow",
               part_role="recoil-absorber-element",
               recoil_stroke_m=RECOIL_STROKE_M,
               slide_axis=(0.0, 0.0, 1.0),
               piston_area_m2=round(_elem_area, 6),
               working_pressure_pa=2.0e7,
               parallel_with=[f"turret.absorber.{n}" for n, _, _ in _absorber
                              if n != name],
               **attrs)
        if name != "spring":
            # One third of the recuperator surrounds each piston. The three
            # rates sum to the original stored-energy design while their
            # symmetric placement removes the off-axis spring couple.
            g.edge(f"turret.absorber.recuperator.{name}",
                   f"turret.absorber.piston.{name}", "turret.spine.0",
                   "spring-damper", radius=ABSORBER_BORE_M * 0.42,
                   palette="actuator-yellow",
                   part_role="recoil-absorber-recuperator-spring",
                   recoil_stroke_m=RECOIL_STROKE_M,
                   slide_axis=(0.0, 0.0, 1.0),
                   spring_rate_n_per_m=_recuperator_total_rate / 3.0,
                   force_components=("spring-recuperator",),
                   parallel_with=["turret.absorber.spring"],
                   note="equal spring share around each of the three pistons")

    g.assembly = "gun-tube"

    g.edge("turret.recoil_slide", "turret.breech", "turret.cradle", "single-axis-slider",
           radius=max(0.020, barrel_radius_m * 0.6), palette="actuator-yellow",
           slide_axis=(0.0, 0.0, 1.0), recoiling_mass_kg=slide_mass_kg,
           recoil_stroke_m=slide_stroke_m,
           piston_bore_m=slide_piston_bore_m,
           metering_pin=True,
           carries_firing_load=True,
           recoil_velocity_m_s=round(slide["recoil_velocity_m_s"], 3),
           free_recoil_energy_j=round(slide["free_recoil_energy_j"], 1),
           design_peak_force_n=round(slide["design_peak_force_n"], 1),
           load_path="tube-rides-in-the-fine-platform-ways")
    g.edge("turret.recoil", "turret.breech", "turret.cradle",
           "point-impulse-wrench-coupling",
           radius=0.020, load_path="individual-shot-recoil-r-cross-impulse-into-the-slide",
           impulse_n_s=round(shot_impulse, 1))

    # ---- THE FIRING LOCK (optional, and now a belt to the seat's braces) ----
    # THE FINE STAGE IS NOT A FIRING LOAD PATH, and measuring it said so
    # loudly: resolving one shot onto six nearly-flat legs put 93 kN
    # through a 20 mm cylinder at 40 mm bore -- 296 MPa of oil in a
    # 25 MPa actuator -- and 375 kN at 76 mm. That is not a leg that
    # needs to be bigger; it is a load that must not go through the legs
    # at all. With the shot bypassing them the same legs carry 0.7 to
    # 7.4 MPa, so they were right-sized from the start and merely being
    # asked to do somebody else's job.
    #
    # So: position with the fine platform, CLAMP, then fire. The clamp
    # ties the cradle straight to the trunnion and the legs sit in
    # parallel with it carrying nothing.
    #
    # IT IS A FRICTION CLAMP, AND THAT IS NOT AN IMPLEMENTATION DETAIL.
    # Anything form-fitting -- a cone, a taper, a detent, a tooth --
    # centres itself as it closes, which means it drags the gun back to
    # nominal and throws away the very correction the fine stage just
    # made. A clamp that holds wherever it is shut is the only kind that
    # can be closed after aiming. Real hardware: hydraulic rail brakes
    # and shrink-disc shaft clamps do exactly this.
    #
    # SIZED FROM THE STANDARD BREECH, NOT THE FITTED BARREL. The mount
    # is built once for the largest round it will ever take, and every
    # smaller tube then has margin instead of its own structure. That is
    # the whole return on a standard breech: one clamp, one trunnion,
    # one set of braces, and the barrel becomes the only variable.
    breech_cal = _cannon_calibre(standard_breech_bore_mm)
    breech_charge_kg = breech_cal.mass_kg * 0.36
    breech_impulse = (breech_cal.mass_kg * breech_cal.muzzle_m_s
                      + breech_charge_kg * breech_cal.muzzle_m_s * 1.5)
    breech_slide_mass = _mass_of("gun-steel",
                                 math.pi * (standard_breech_bore_mm / 1000.0 * 0.72) ** 2
                                 * max(0.4, breech_cal.length_m * 14.0),
                                 hollow_fraction=0.18) + 25.0
    breech_stroke = max(0.28, standard_breech_bore_mm / 1000.0 * 8.0)
    worst_shot_n = _size_recoil_slide(
        impulse_n_s=breech_impulse, recoiling_mass_kg=breech_slide_mass,
        stroke_m=breech_stroke,
        piston_bore_m=max(0.060, standard_breech_bore_mm / 1000.0 * 2.4),
        peak_force_n=recoil_peak_force_n)["design_peak_force_n"]
    # a friction joint must not slip, so it is sized with real margin
    clamp_hold_n = worst_shot_n * firing_lock_margin
    clamp_area_m2 = clamp_hold_n / (firing_lock_friction * firing_lock_pressure_pa)
    clamp_face_m = 0.042
    clamp_diameter_m = clamp_area_m2 / (math.pi * clamp_face_m)
    # ---- THE THRUST SEAT, AND WHAT PRELOADS IT ----
    # RECOIL IS AXIAL; AIMING IS ANGULAR. They are orthogonal, so there
    # is no need to switch between carrying the shot and being able to
    # aim. The spherical seat is between the NON-RECOILING closed journal
    # shoe and the pitch body: the slide and its three absorbers act first,
    # then their reaction crosses this seat into the trunnion. Putting the
    # seat on the moving collar would bypass and lock the recoil slide.
    #
    # The arithmetic is one-sided to the point of being funny. A 20 mm
    # leg's stiffness is its oil column, beta*A/L, which comes to
    # 2.8 MN/m -- so soft that one 40 mm shot would squeeze it 28 mm and
    # a 76 mm shot 113 mm, through an actuator with 25 mm of travel. Put
    # a steel seat in parallel at 59 000 MN/m and the six legs take
    # 0.03% of the shot: ninety grams' worth of force. The leg being
    # soft is not the problem, it is the whole mechanism -- load divides
    # by stiffness, and there is no contest.
    #
    # A SERIES SPRING INSIDE THE LEG WOULD DO ALMOST NOTHING, which is
    # worth writing down because it is the intuitive place to put one.
    # The oil is already the soft element; a 10 MN/m stack in series
    # with 2.8 MN/m of oil gives 2.2 MN/m, and moves the load share from
    # 0.03% to 0.023%. The spring has to act in PARALLEL, holding the
    # gun into the seat, to be worth anything.
    #
    # AND THAT IS WHAT THE PRELOAD IS FOR. The seat only works while it
    # is in contact. Recoil pushes the gun INTO it, so the shot is never
    # the problem -- the recuperator is, shoving the tube back to
    # battery hard enough to lift the gun off its seat. The stacks are
    # sized to beat that with margin, and they also take the lash out of
    # every joint they cross, which is pointing error the coarse drives
    # cannot control away.
    seat_r = 0.075
    # The journal shoe is already tied to the trunnion block. This parallel
    # seat carries translation while allowing the pitch rotations.
    g.assembly = "trunnion-thrust-seat"
    g.edge("turret.thrust_seat", "turret.journal.lower", "turret.pitch",
           "spherical-thrust-seat",
           radius=0.055, palette="rollbar-silver",
           seat_radius_m=seat_r, free_rotation=True, carries_firing_load=True,
           axial_stiffness_n_per_m=200e9 * math.pi * seat_r ** 2 / 0.06,
           bearing_stress_pa=round(worst_shot_n / (math.pi * seat_r ** 2), 1),
           load_path="shot-goes-straight-into-the-trunnion-around-the-fine-stage")
    # The recuperator's maximum return force is its declared linear rate at
    # full stroke. Size the seat stacks against that actual element instead
    # of the removed, duplicate gas-over-oil slide law.
    recup_n = _recuperator_total_rate * RECOIL_STROKE_M
    preload_total_n = (recup_n + slide_mass_kg * 9.81) * preload_margin
    for k, ang in enumerate((0.0, 2.0943951, 4.1887902)):
        g.edge(f"turret.seat_preload.{k}", "turret.journal.lower", "turret.pitch",
               "belleville-preload-stack", radius=0.038, palette="actuator-yellow",
               preload_n=round(preload_total_n / 3.0, 1),
               stack_rate_n_per_m=1.0e7,
               holds="spherical-thrust-seat-in-contact-against-the-recuperator",
               also="takes the lash out of the fine stage",
               carries_firing_load=False,
               load_path="constant-preload-across-the-seat")

    # The clamp is no longer what makes firing possible -- the seat is,
    # and it costs no cycle time at all where the clamp cost 70 ms a
    # round. It is kept as a declared option because holding the mount
    # rigid is still worth having for travel, for a gun left laid on a
    # bearing, and as a second path if a preload stack is lost.
    for side, sx in (("l", -1.0), ("r", 1.0)) if firing_lock else ():
        g.edge(f"turret.firing_lock.{side}", "turret.cradle", "turret.pitch",
               "firing-lock-clamp", radius=0.048, palette="rollbar-silver",
               engaged=True, carries_firing_load=True,
               friction_coefficient=firing_lock_friction,
               clamp_pressure_pa=firing_lock_pressure_pa,
               clamp_area_m2=round(clamp_area_m2 / 2.0, 5),
               clamp_diameter_m=round(clamp_diameter_m, 4),
               clamp_face_width_m=clamp_face_m,
               holding_force_n=round(clamp_hold_n / 2.0, 1),
               sized_for_bore_mm=standard_breech_bore_mm,
               sized_for_shot_n=round(worst_shot_n, 1),
               engage_time_s=0.035, release_time_s=0.035,
               self_centring=False,
               load_path="cradle-clamped-to-trunnion-so-the-shot-bypasses-the-fine-stage")

    # ---- THE MACHINE GUN: its own hemisphere, its own structure ----
    # ---- THE TWO SWEPT CIRCLES MUST NOT OVERLAP ----
    # The machine gun has to be able to go fully vertical and then drop
    # straight down into the vehicle, all the way in, for service and
    # for keeping it out of the weather and out of sight. That is only
    # possible if nothing of the main gun's fine-aim rig ever passes
    # through the column the mount retracts down. So the two swept
    # circles are computed and CHECKED, not eyeballed: the fine rig's
    # is the radius its basket and legs sweep about the bore axis, the
    # gun's is the barrel's own swept circle when it stands vertical.
    #
    # The wider ring pair is what makes this fit. At the old 0.52 m
    # ring there was no annulus left to stand a retracting mount in
    # once the basket had taken its share.
    mg_clear_r = 0.26                       # the mount's own retracting column
    fine_sweep_r = base_half_w + 0.075      # basket corner plus leg body
    mg_base_r = fine_sweep_r + mg_clear_r + 0.040
    assert mg_base_r <= ring_radius * 0.98, (
        f"the machine gun's retraction column ({mg_clear_r * 1000:.0f} mm "
        f"radius) does not fit between the fine rig's sweep "
        f"({fine_sweep_r * 1000:.0f} mm) and the ring "
        f"({ring_radius * 1000:.0f} mm) -- widen the ring pair")
    mg_base_x, mg_base_z = mg_base_r, -TRUNNION_Z * 0.4
    mg_y = carriage_y + 0.30
    # the machine gun's pedestal traverses with the mount
    g.motion_group = "traverse"
    # THE SECONDARY MACHINE GUN IS GONE FOR NOW. It was a 1.66 m tube
    # at bore height offset to one side, reaching past the deck edge
    # and braced by a 3.91 m diagonal to the opposite corner -- a stub
    # with no withdrawal travel, no vertical stow, and nothing
    # enforcing the rule that its sweep must not overlap the fine-aim
    # armature. An afterthought to the cannon, and the conspicuous
    # bar across every render of it.
    #
    # The clearance check above still runs: mg_base_r is the radius a
    # secondary WOULD need and it still has to clear the fine sweep
    # and the ring, so putting one back later is a question already
    # answered rather than one to rediscover.

    # ---- THE JACKS ARE SIZED TO WHAT THEY ACTUALLY HAVE TO LIFT ----
    # `design_lift_mass_kg` was a number typed into the base builder
    # before the gun on top of it existed, which is the same mistake as
    # declaring a mass next to a geometry: the answer is already in the
    # graph. Every body's mass is summed here and the four legs are
    # sized off it, with the margin stated rather than buried.
    _lifted = sum(float(n.get("mass_kg", 0.0)) for n in g.nodes)
    _per_leg = _lifted * 9.81 / 4.0 * 1.6      # 1.6 on the static share
    for _n in g.nodes:
        if _n.get("lifts_machine"):
            _n["lifted_mass_kg"] = round(_lifted, 1)
            _n["capacity_n"] = round(_per_leg, 1)
            _n["margin_over_static"] = 1.6
    for _e in g.edges:
        if _e.get("kind") == "outrigger-lift-jack":
            _e["holding_force_n"] = round(_per_leg, 1)
            _e["relief_force_n"] = round(_per_leg * 1.4, 1)
            # and the cylinder has to be able to make that force
            _e["required_pressure_pa"] = round(
                _per_leg / (math.pi * (_e["bore_m"] / 2.0) ** 2), 0)
    # AND THE STROKE HAS TO BUY REAL CLEARANCE. A jack that extends
    # 1.3 m under a machine whose lowest box hangs 0.3 m down has not
    # made 1.3 m of room; it has made a metre. The number that matters
    # is measured to the bottom of the lowest BODY, and it is checked
    # here against what has to fit under it.
    _raise = raise_on_hydraulics(g)
    assert not _raise["log"]["stalled"], (
        "the power pack cannot lift this machine: "
        f"{_raise['log']['required_pressure_pa'] / 1e5:.0f} bar needed")
    _clear = deployed_clearance(g, lift_m=_raise["lift_m"])["clearance_m"]
    g.deploy_seconds = round(_raise["log"]["elapsed_s"], 1)
    assert _clear >= 1.05, (
        f"deployed, this machine has only {_clear * 1000:.0f} mm under it "
        f"-- not enough to run a trailer or an armoured chassis in")
    g.deployed_clearance_m = round(_clear, 3)
    assert all(e.get("required_pressure_pa", 0.0) <= 35.0e6
               for e in g.edges if e.get("kind") == "outrigger-lift-jack"), (
        "the outrigger jacks would need more than 350 bar to pick this "
        "machine up -- give them a bigger bore")
    return g


def command_hoist(graph: ProductionGraph, rest_length_m: float) -> None:
    """Hand the solver the one number the engine sim owns.

    This is the whole interface between the interior sim and the world:
    engine_toy works out what the fluid is doing -- what the pack can
    hold, how much volume has gone into the ram -- and publishes a
    commanded rest length. It does NOT move anything."""
    for e in graph.edges:
        if e["constraint"] == "linear-hydraulic-actuator":
            e["commanded_rest_length_m"] = max(
                float(e.get("minimum_rest_length_m", 0.0)),
                min(float(e.get("maximum_rest_length_m", 1e9)), float(rest_length_m)))


# =====================================================================
#  THE INTERIOR SIM'S SIDE OF THE INTERFACE
# =====================================================================
def hoist_rest_length_from_fluid(pack, dt: float, commanded: float,
                                 bore_m: float = 0.125,
                                 stroke_m: float = 0.850,
                                 current_rest_m: float = 0.0) -> dict:
    """What engine_toy owns, and the only thing it hands over.

    The interior sim knows the fluid: what the pack and its accumulator
    can deliver this tick, and what pressure they can hold while doing
    it. That converts to a VOLUME, and a volume in a known bore is a
    LENGTH. The commanded rest length and the force the member is
    allowed to make are published; where the structure actually ends up
    is the solver's business, and it may well be somewhere else --
    something heavy on the hatch, a jammed guide, a holed line.

    Note what is NOT published: a position. A position would be this
    sim asserting an outcome it has no right to assert."""
    area = math.pi * bore_m * bore_m / 4.0
    out = pack.step(dt, abs(commanded) * area * 0.42 * 60_000.0,
                    float(getattr(pack, "relief_pressure_pa", 25e6)))
    volume_m3 = out["delivered_l_min"] / 60_000.0 * dt * (1.0 if commanded >= 0 else -1.0)
    advance = volume_m3 / max(area, 1e-9)
    rest = max(0.0, min(stroke_m, current_rest_m + advance))
    return {
        "commanded_rest_length_m": rest,
        # the member may not pull harder than the fluid behind it can
        "available_axial_force_n": out["pressure_pa"] * area,
        "supply_pressure_pa": out["pressure_pa"],
        "delivered_l_min": out["delivered_l_min"],
        "starved": bool(out.get("starved", False)),
    }
