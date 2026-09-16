"""What a node IS: the joint where members meet, and what it transmits.

A node in this project has been a point with a position and a mass. That
is enough to hang a mesh on and not nearly enough to carry structure,
because the single most consequential fact about a junction is not where
it is -- it is WHAT IT TRANSMITS. A pinned clevis and a welded gusset at
the same coordinates are different structures: the first lets members
rotate freely against each other and therefore develops no bending in
them at all, the second carries moment and therefore does.

That is why joints come before beam kinematics rather than after. The
joint type IS the boundary condition the beam solve runs against; asking
what bending strain a member carries without knowing whether its ends
are pinned or welded is asking an unanswerable question.

THE DOF TRANSFORM IS DERIVED FROM NOMINAL GEOMETRY wherever the geometry
determines it, and declared parametrically only where it genuinely does
not. A fillet weld's moment capacity follows from its throat and the
perimeter it runs around; a pin's from its diameter in double shear; a
socket's from how far the tube is engaged. None of those are numbers
anyone should be typing in.

WHAT A JOINT DOES WITH EACH DEGREE OF FREEDOM is one of three things,
and the distinction is the whole model:

    RIGID      transmitted with the parent section's own stiffness
    FREE       not transmitted at all; the members articulate here
    COMPLIANT  transmitted through a real stiffness, with real damping
               and a real travel before yield -- the existing six-axis
               Kelvin-Voigt bushing, generalised off the edge and onto
               the node where it belongs

BRAZED IS NOT WELDED, and it is worth its own type rather than a flag.
A silver-brazed joint carries perhaps 170 MPa across its fillet where
the 4130 tube it joins yields at 460. The joint is weaker than the parts
it joins, which is exactly backwards from a welded node and is why
aircraft tube frames are welded rather than brazed wherever the load
path is serious.
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass, field

#: The six degrees of freedom, in the joint's own frame. x is along the
#: principal member; y and z are transverse; rx is torsion about it.
DOF_NAMES = ("x", "y", "z", "rx", "ry", "rz")

RIGID, FREE, COMPLIANT = "rigid", "free", "compliant"


@dataclass(frozen=True)
class BondMaterial:
    """What actually holds the joint together, which is usually not what
    the members are made of."""
    key: str
    label: str
    shear_strength_pa: float
    tensile_strength_pa: float
    note: str = ""


BONDS = {
    "e70-fillet": BondMaterial(
        "e70-fillet", "E70 fillet weld", 290e6, 483e6,
        note="ordinary steel structural weld: the throat is as strong as "
             "the parent in shear, so a properly sized weld is not the "
             "weak link"),
    "tig-4130": BondMaterial(
        "tig-4130", "TIG weld, 4130 tube", 320e6, 560e6,
        note="normalised after welding or it is brittle at the heat-"
             "affected zone, which is where a cage actually cracks"),
    "bag-silver": BondMaterial(
        "bag-silver", "BAg silver braze", 170e6, 240e6,
        note="a THIRD of the tube's strength: the fillet is the weak "
             "element by design, and joint gap decides whether you get "
             "even that"),
    "rbcuzn-bronze": BondMaterial(
        "rbcuzn-bronze", "RBCuZn bronze braze", 210e6, 300e6,
        note="stronger than silver and hotter; still well below the parent"),
    "bolted-preload": BondMaterial(
        "bolted-preload", "preloaded bolted flange", 620e6, 830e6,
        note="grade 8 in the bolt, but the joint slips at friction long "
             "before the bolt sees that"),
    # ---- what holds sheet metal and light frames together ----------
    "sheet-metal-screw": BondMaterial(
        "sheet-metal-screw", "self-tapping sheet metal screw", 480e6, 620e6,
        note="case-hardened, so the screw itself is never the weak part: "
             "the SHEET is, and it fails by the thread stripping or the "
             "hole tearing, which is why a seam's capacity is per screw "
             "in a given gauge and not per screw"),
    "spot-weld": BondMaterial(
        "spot-weld", "resistance spot weld", 300e6, 420e6,
        note="a nugget the size of the electrode tip, every few "
             "centimetres: a unibody is a few thousand of them"),
    "tack-weld": BondMaterial(
        "tack-weld", "tack weld", 290e6, 483e6,
        note="an E70 fillet a centimetre long: the way a bracket is put "
             "on a sheet casing, because a full seam would warp the sheet. "
             "Each tack is a short fillet with a real throat and the "
             "bracket's capacity is the tacks' sum"),
    "snap-fit": BondMaterial(
        "snap-fit", "moulded cantilever snap", 40e6, 60e6,
        note="held by the retention of a plastic hook, which is a "
             "force, not a strength: it lets go at that force and is "
             "undamaged by having done so, which is what it is for"),
}


@dataclass
class JointType:
    """A kind of junction, and what it does with each degree of freedom."""
    key: str
    label: str
    transform: tuple            # one of RIGID/FREE/COMPLIANT per DOF
    bond: str = "tig-4130"
    note: str = ""

    def freedoms(self) -> tuple:
        return tuple(n for n, t in zip(DOF_NAMES, self.transform) if t == FREE)

    def describe(self) -> str:
        free = ", ".join(self.freedoms()) or "nothing"
        return f"{self.label}: free in {free}"


JOINT_TYPES = {
    # --- everything transmitted -------------------------------------
    "solid-welded": JointType(
        "solid-welded", "solid welded node", (RIGID,) * 6, "tig-4130",
        note="members burned directly into one another, or into a gusset. "
             "Full moment transfer, so this is where beam bending "
             "actually develops"),
    "jacketed-welded": JointType(
        "jacketed-welded", "jacketed and welded node", (RIGID,) * 6, "tig-4130",
        note="a sleeve over the junction before welding: the same six "
             "rigid freedoms, but far more moment capacity because the "
             "jacket's own section carries it"),
    "brazed": JointType(
        "brazed", "brazed node", (RIGID,) * 6, "bag-silver",
        note="rigid until the fillet lets go, and the fillet is a third "
             "the strength of the tube"),
    "hollow-socket": JointType(
        "hollow-socket", "hollow socket joining", (RIGID,) * 6, "tig-4130",
        note="one tube inside another. Axial and shear are as good as "
             "welded; MOMENT is limited by the engagement length, because "
             "the couple has only that lever to work with"),
    # --- articulating ------------------------------------------------
    "pinned-clevis": JointType(
        "pinned-clevis", "pinned clevis", (RIGID, RIGID, RIGID, RIGID, FREE, RIGID),
        "bolted-preload",
        note="THE PIN IS PERPENDICULAR TO THE MEMBER, so what it frees is "
             "a BENDING axis (ry) and what it resists is torsion about "
             "the member (rx) -- the cheeks bear on the tongue. Written "
             "the other way round first, which would have been a pin "
             "doing the exact opposite of its job: transmitting bending "
             "and letting the member spin"),
    "spherical-seat": JointType(
        "spherical-seat", "spherical seat", (RIGID, RIGID, RIGID, FREE, FREE, FREE),
        "bolted-preload",
        note="three translations transmitted, all three rotations free: "
             "a thrust bearing that aims. The turret's own gun mount"),
    "universal": JointType(
        "universal", "universal joint", (RIGID, RIGID, RIGID, RIGID, FREE, FREE),
        "bolted-preload",
        note="torque through, bending free in both transverse axes"),
    # --- compliant ---------------------------------------------------
    "bushed": JointType(
        "bushed", "elastomer-bushed node", (COMPLIANT,) * 6, "bolted-preload",
        note="everything transmitted through a real stiffness with real "
             "damping: the six-axis Kelvin-Voigt junction, which is what "
             "isolates a structure instead of merely holding it"),
    "bolted-flange": JointType(
        "bolted-flange", "preloaded bolted flange", (RIGID,) * 6, "bolted-preload",
        note="rigid while the preload holds and a slip plane once it does "
             "not, which is a failure mode rather than a stiffness"),
    # ---- sheet and light frame ------------------------------------
    "screwed-seam": JointType(
        "screwed-seam", "sheet metal screwed seam", (COMPLIANT,) * 6,
        "sheet-metal-screw",
        note="a lap of sheet with a screw every so many inches. Not "
             "rigid: each screw is a stiffness in bearing against its "
             "hole, and the seam's stiffness is that times the count. "
             "Not bushed: it has no damping worth the name. See Seam"),
    "snap-fastened": JointType(
        "snap-fastened", "snap-fitted panel", (COMPLIANT,) * 6, "snap-fit",
        note="a panel held by hooks: stiff enough to locate it, and it "
             "releases at the retention force rather than breaking -- so "
             "its capacity is small and its failure is reversible"),
    "tack-welded": JointType(
        "tack-welded", "tack-welded bracket", (RIGID,) * 6, "tack-weld",
        note="a bracket, a stiffener or a stand-off on sheet casing, held "
             "by tacks at a pitch. Rigid while it holds; the capacity is "
             "the tacks', which is far less than the bracket's, so a "
             "tacked bracket fails at the sheet and not in the bracket"),
    "piano-hinge": JointType(
        "piano-hinge", "piano hinge", (RIGID, RIGID, RIGID, RIGID, FREE, RIGID),
        "bolted-preload",
        note="a continuous pin the length of the door edge: free to turn "
             "about its own line (ry, the line being vertical) and rigid "
             "in everything else, because a pin that long cannot rack. "
             "It carries the door OPEN; shut, the dogs carry the pressure "
             "and the hinge is merely along for it"),
    "unibody-spot-welded": JointType(
        "unibody-spot-welded", "spot-welded unibody seam", (RIGID,) * 6,
        "spot-weld",
        note="sheet folded into a box and spot welded along its flanges: "
             "the case IS the frame. Rigid, because a spot weld does not "
             "slip, and light, because there is nothing but the skin"),
}


@dataclass
class Joint:
    """One node's junction, sized from its own nominal geometry."""
    identity: str
    kind: str = "solid-welded"
    #: the principal tube meeting here: what sets the joint's scale
    tube_outer_diameter_m: float = 0.048
    tube_wall_m: float = 0.004
    #: fillet leg for a weld or braze; engagement for a socket; pin
    #: diameter for a clevis; jacket wall for a jacketed node
    fillet_leg_m: float = 0.005
    engagement_m: float = 0.0
    pin_diameter_m: float = 0.016
    jacket_wall_m: float = 0.0
    bushing: dict = None
    members: tuple = ()
    #: HOW WELL IT WAS ACTUALLY MADE. This does NOT scale the capacities
    #: below -- it decides what `welding.weld_edge` puts in the graph.
    #: The capacities here are the joint's nominal geometry; the weld is
    #: a member of its own, and the solve is where the two meet.
    weld: object = None

    @property
    def type(self) -> JointType:
        return JOINT_TYPES[self.kind]

    @property
    def bond(self) -> BondMaterial:
        return BONDS[self.type.bond]

    # ------------------------------------------------------------------
    @property
    def weld_throat_m(self) -> float:
        """A fillet's throat is its leg over root two: the shortest path
        through the metal, which is where it parts."""
        return self.fillet_leg_m * math.sqrt(0.5)

    @property
    def bond_perimeter_m(self) -> float:
        return math.pi * self.tube_outer_diameter_m

    @property
    def parent_axial_capacity_n(self) -> float:
        """What the TUBE itself carries. A joint cannot be stronger than
        the thing it joins, and a socket sized on bond area alone happily
        reported eleven times the tube's own yield -- a junction that
        would tear the member out rather than fail itself."""
        from milspec import TubeSection
        ro = self.tube_outer_diameter_m / 2.0
        ri = max(0.0, ro - self.tube_wall_m)
        area = math.pi * (ro * ro - ri * ri)
        return area * 460e6          # 4130 normalised, the frame default

    def axial_capacity_n(self) -> float:
        """What the junction carries along the member, never more than
        the member itself, and never more than the weld achieved."""
        return min(self._bond_axial_n(), self.parent_axial_capacity_n)

    def _bond_axial_n(self) -> float:
        """What the bond alone would carry, before the parent caps it."""
        t = self.type
        if t.key == "pinned-clevis":
            # a pin in DOUBLE shear: two planes carry it
            return 2.0 * math.pi * self.pin_diameter_m ** 2 / 4.0 * self.bond.shear_strength_pa
        if t.key == "hollow-socket":
            # the socket carries axially on its bond over the engaged area
            area = math.pi * self.tube_outer_diameter_m * max(self.engagement_m, 1e-6)
            return area * self.bond.shear_strength_pa
        area = self.weld_throat_m * self.bond_perimeter_m
        if t.key == "jacketed-welded":
            # the jacket adds its own section in parallel
            area += math.pi * (self.tube_outer_diameter_m + self.jacket_wall_m) \
                * self.jacket_wall_m
        return area * self.bond.shear_strength_pa

    @property
    def parent_moment_capacity_nm(self) -> float:
        """The tube's own plastic moment: the other ceiling."""
        ro = self.tube_outer_diameter_m / 2.0
        ri = max(0.0, ro - self.tube_wall_m)
        modulus = math.pi * (ro ** 4 - ri ** 4) / (4.0 * ro)
        return modulus * 460e6

    def moment_capacity_nm(self) -> float:
        """What the junction carries as BENDING -- the number that
        decides whether beam theory has anything to work with here.

        A free rotation carries none by definition, which is not a
        weakness but the joint doing its job."""
        t = self.type
        if FREE in (t.transform[4], t.transform[5]):
            return 0.0
        d = self.tube_outer_diameter_m
        cap = self.parent_moment_capacity_nm
        if t.key == "pinned-clevis":
            # rigid about rz through the clevis cheeks, over their gap
            return min(self.axial_capacity_n() * self.pin_diameter_m * 0.5, cap)
        if t.key == "hollow-socket":
            # THE ENGAGEMENT LENGTH IS THE LEVER. A socket resists moment
            # as a couple of bearing forces at either end of the overlap,
            # so a short engagement is a hinge with extra steps.
            bearing = math.pi * d * max(self.engagement_m, 1e-6) * 0.5 \
                * self.bond.shear_strength_pa
            return min(bearing * max(self.engagement_m, 1e-6) * 0.5, cap)
        # a fillet around a tube: section modulus of an annular weld group
        throat = self.weld_throat_m
        modulus = math.pi * d * d * throat / 4.0
        if t.key == "jacketed-welded":
            modulus += math.pi * d * d * self.jacket_wall_m / 2.0
        return min(modulus * self.bond.shear_strength_pa, cap)

    def torsion_capacity_nm(self) -> float:
        if self.type.transform[3] == FREE:
            return 0.0
        d = self.tube_outer_diameter_m
        nominal = math.pi * d * d * self.weld_throat_m / 2.0 * self.bond.shear_strength_pa
        return nominal

    # ------------------------------------------------------------------
    def dof_transform(self) -> dict:
        """What this joint does with each degree of freedom, with the
        capacity behind it. THIS is what a beam assembly needs from a
        node: not a position, but a statement of what passes through."""
        t = self.type
        axial, moment, torsion = (self.axial_capacity_n(),
                                  self.moment_capacity_nm(),
                                  self.torsion_capacity_nm())
        out = {}
        for name, mode in zip(DOF_NAMES, t.transform):
            capacity = (axial if name in ("x", "y", "z")
                        else torsion if name == "rx" else moment)
            entry = {"mode": mode, "capacity": capacity}
            if mode == COMPLIANT and self.bushing:
                entry["stiffness"] = (self.bushing["linear_stiffness_n_per_m"]
                                      if name in ("x", "y", "z")
                                      else self.bushing["angular_stiffness_nm_per_rad"])
                entry["damping"] = (self.bushing["linear_damping_n_s_per_m"]
                                    if name in ("x", "y", "z")
                                    else self.bushing["angular_damping_nm_s_per_rad"])
            out[name] = entry
        return out

    def transmits_moment(self) -> bool:
        """Whether members meeting here develop bending at all."""
        return self.moment_capacity_nm() > 0.0

    def describe(self) -> list:
        t = self.type
        return [
            f"  {self.identity}: {t.label} ({self.bond.label})",
            f"    free in {', '.join(t.freedoms()) or 'nothing'}; "
            f"axial {self.axial_capacity_n() / 1000:8.1f} kN  "
            f"moment {self.moment_capacity_nm():8.1f} N.m  "
            f"torsion {self.torsion_capacity_nm():8.1f} N.m",
        ]


def bushing_pack(frame_mount: bool = False) -> dict:
    """The existing six-axis Kelvin-Voigt junction, unchanged.

    Kept as the compliant joint's parameter pack rather than rewritten:
    it already derives its stiffness from a real polyurethane annulus and
    its damping from the critical damping of that stiffness, which is
    exactly right. What was wrong was only WHERE it lived -- bolted to
    edges, in two hardcoded flavours -- not what it computed."""
    from turret_production import _bushing
    return _bushing(frame_mount)


# =====================================================================
#  THE OTHER HALF: WHAT A MEMBER DOES, NOT WHAT A NODE DOES
# =====================================================================
# Everything above describes a JUNCTION -- how members meeting at a node
# are bonded to one another. It says nothing about the member itself,
# and members in this project carry a second, entirely separate joint
# concept: a `constraint` spelling on the edge. "single-axis-slider".
# "oleo-recoil-slide". "coolant-line".
#
# THAT SPELLING WAS AN IDENTITY MADE OF A STRING, and every consumer
# that needed to know anything about it kept its own set of strings to
# compare against. There were five in the turret builder alone --
# ZERO_LENGTH_CONSTRAINTS, ROUTED_LINE_CONSTRAINTS, BUSHED_CONSTRAINTS,
# _MEMBER_ALLOY, _MEMBER_MATERIAL -- plus an inline tuple deciding what
# counted as spring-like and another in the beam solver deciding what
# was a fluid line. Seven places that all had to agree about one word,
# none of which could be checked, and a misspelling in any of them
# produced a member with the wrong steel, no bushing and no damage law,
# silently.
#
# The frame solver had none of those sets, so it did the thing that
# costs most: it assembled EVERY member as a fully welded twelve-freedom
# beam. Thirty-nine sliders solved as bars. Eighty-four coolant lines
# carrying structural load around the barrel. A recoil mechanism whose
# entire purpose is to be free along one axis, welded shut.
#
# So a member constraint is an object, in the SAME per-freedom
# vocabulary the junctions above already use: RIGID, FREE, COMPLIANT,
# one per degree of freedom, x along the member.

#: Released at one end only (a slider, a ram) or at both (a pin-ended
#: strut, which then transmits no moment anywhere along it).
ONE_END, BOTH_ENDS = "b", "both"

#: Free to slide along itself, rigid in the other five. Whatever force
#: such a member carries along its axis comes from its own law, never
#: from its section.
SLIDING = (FREE, RIGID, RIGID, RIGID, RIGID, RIGID)
#: A hinge: one rotation free, about the pin. ry, because a door's hinge
#: line stands up.
HINGED = (RIGID, RIGID, RIGID, RIGID, FREE, RIGID)
#: Free in all three rotations: the honest idealisation of a bearing
#: whose axis the member does not declare. It is under-stiff in torsion
#: rather than infinitely over-stiff in bending, and of those two errors
#: only one invents structure that is not there.
PINNED = (RIGID, RIGID, RIGID, FREE, FREE, FREE)
WELDED = (RIGID,) * 6
# A screw closes a split clamp axially.  Its authored spring/preload owns x;
# the bolt shank locates the two lugs transversely but does not weld their
# rotations together.
# The screw's axial extension is owned by its torque/preload spring.  Below
# slip, however, the clamped lug faces and bolt bearing transmit both shear
# and relative rotation.  Releasing all three rotations left every two-piece
# strap with a closure mechanism that a thin strap happened to regularise.
TENSIONER = (FREE, RIGID, RIGID, RIGID, RIGID, RIGID)


@dataclass(frozen=True)
class MemberConstraint:
    """What a member transmits, what it is made of, how it behaves.

    One object per constraint spelling, replacing seven scattered sets
    of strings that all had to agree and never could be checked."""
    key: str
    transform: tuple = WELDED
    release_ends: str = BOTH_ENDS
    #: carries fluid or signal, not load. Not a member of the frame at
    #: all -- no damage law, no bushing, no stiffness.
    routed: bool = False
    #: its length is COMMANDED, so a change in it is not strain
    travels: bool = False
    #: it develops its own force from its own law
    spring_like: bool = False
    #: its two ends are the same point: two surfaces in contact
    zero_length: bool = False
    #: a joint rather than a column, so it gets the compliant bushing
    bushed: bool = False
    alloy: str = "4130n"
    material: str = "steel-plate"
    note: str = ""
    #: assigned at registration; the identity everything hot compares
    token: int = -1

    def freedoms(self) -> tuple:
        return tuple(n for n, t in zip(DOF_NAMES, self.transform) if t == FREE)

    def released_indices(self, end: str) -> tuple:
        """Which local freedoms carry nothing, at the named end.

        A SLIDER RELEASES AT ONE END, NOT BOTH. Releasing the same
        freedom at both ends of a two-node element does not make it
        twice as free -- it disconnects it, and the stiffness matrix
        goes singular. One end is what makes the member transmit
        nothing along that axis while still holding the two bodies in
        line, which is what a slide actually does."""
        free = tuple(i for i, t in enumerate(self.transform) if t == FREE)
        if not free:
            return ()
        if self.release_ends == BOTH_ENDS or end == "b":
            return free
        return ()

    @property
    def welded(self) -> bool:
        return FREE not in self.transform


#: THE SPELLING IS NOT THE IDENTITY. A constraint is a TOKEN -- a small
#: integer, assigned once at declaration -- and the string is a label
#: for people to read.
#:
#: Why this matters and is not tidiness: every consumer of this graph
#: was comparing strings. `e["constraint"] == "greased-thrust-track"` is
#: a hash and a character-by-character compare, run once per member per
#: pass, in a frame solve that touches sixteen hundred members and a
#: beam survey that touches them again. It is slow in the inner loop and
#: it is worse than slow at the edges: a misspelling compares FALSE
#: against everything and silently means "not that kind of joint", which
#: is how thirty-nine sliders came to be welded and eighty-four coolant
#: lines came to be structural steel. Nothing raises. Nothing can.
#:
#: A token cannot be misspelled, because there is no spelling to get
#: wrong -- `joints.T.single_axis_slider` is resolved at import and an
#: unknown name is an AttributeError at the line that wrote it. And the
#: comparison is an integer.
#:
#: Edges carry BOTH: `constraint` stays the readable string so every
#: document already written still loads and every report still reads,
#: and `constraint_token` is what the hot paths compare.
_TOKENS: list = []


class _TokenNamespace:
    """`joints.T.<name>` -> the token for that constraint.

    Attribute access rather than a dict lookup, so a name that does not
    exist fails at the point of use instead of returning None and being
    compared falsely against everything."""

    def __getattr__(self, name: str) -> int:
        key = name.replace("_", "-")
        mc = MEMBER_CONSTRAINTS.get(key)
        if mc is None:
            raise AttributeError(
                f"no declared member constraint {key!r} "
                f"(from joints.T.{name})")
        return mc.token

    def __dir__(self):
        return [k.replace("-", "_") for k in MEMBER_CONSTRAINTS]


def token_of(key: str) -> int:
    """The token for a spelling. Raises on an undeclared one."""
    return member_constraint(key).token


def constraint_of(token: int) -> "MemberConstraint":
    """The constraint a token names -- an index, not a search."""
    return _TOKENS[token]


MEMBER_CONSTRAINTS: dict = {}


def register_constraint(mc: MemberConstraint) -> MemberConstraint:
    if mc.key in MEMBER_CONSTRAINTS:
        raise ValueError(f"member constraint {mc.key!r} registered twice")
    if len(mc.transform) != len(DOF_NAMES):
        raise ValueError(f"{mc.key}: a transform is one entry per freedom")
    # THE TOKEN IS THE POSITION. Assigned in declaration order, so
    # `constraint_of(token)` is a list index and never a lookup.
    mc = dataclasses.replace(mc, token=len(_TOKENS))
    _TOKENS.append(mc)
    MEMBER_CONSTRAINTS[mc.key] = mc
    return mc


def member_constraint(key: str) -> MemberConstraint:
    """The constraint, or a loud failure.

    A spelling nobody has declared is not a default. The default used to
    be a weld in 4130 with no bushing and no damage law, applied in
    silence, which is how a slider came to be solved as a bar."""
    try:
        return MEMBER_CONSTRAINTS[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a declared member constraint. Declare it in "
            f"joints.py with what it transmits -- do not let it default, "
            f"because the default is a weld.") from None


def _mc(key, transform=WELDED, **kw):
    return register_constraint(MemberConstraint(key, transform, **kw))


# ---- structure that does not move -----------------------------------
_mc("rigid-distance", WELDED, alloy="4130n",
    note="the space frame: the only thing here that is genuinely a beam "
         "welded at both ends")
_mc("rigid-offset", WELDED, zero_length=True,
    note="two surfaces of ONE piece of metal described twice -- a seat, "
         "a boss, a machined face")
_mc("welded-joint-element", WELDED, alloy="4130n",
    note="explicit filler-metal element between two weld toes; unlike a "
         "rigid-distance member, its throat, filler and damage are authored")
_mc("strap-clamped-contact", WELDED, alloy="4130n",
    note="preloaded shaped-strap contact, locked only below its declared "
         "friction capacities; a later contact solve may release it on slip")
_mc("bolted-flange-mount", WELDED, zero_length=True)
_mc("point-impulse-wrench-coupling", WELDED, zero_length=True, bushed=True,
    alloy="300m", material="gun-steel",
    note="where the shot's own impulse enters the structure")
# AN ENGINE MOUNT IS COMPLIANT IN EVERY DIRECTION AND THAT IS ITS JOB.
# A prime mover is an unbalanced reciprocating mass bolted to a frame
# somebody has to stand next to; the mount exists to be softer than
# both, in all six freedoms, so the block's own shaking does not become
# the structure's. Declaring it WELDED -- which is what letting it
# default would do -- makes the bay carry the firing impulses of a
# seven-litre multifuel directly, and the beam solve would faithfully
# report a frame being hammered apart by a fault that was in the
# declaration rather than in the machine.
COMPLIANT_ALL = (COMPLIANT,) * 6
_mc("engine-mount-isolator", COMPLIANT_ALL, bushed=True, spring_like=True,
    alloy="4130n", material="elastomer",
    note="the elastomer mount a prime mover sits on: soft in all six, "
         "so the block's own imbalance stays in the block")
_mc("tension-limit-strap", PINNED, alloy="4130n",
    note="carries tension and nothing else, so it is pinned: a strap "
         "that transmitted bending would be a bar")
_mc("strap-tensioner", TENSIONER, release_ends=BOTH_ENDS,
    spring_like=True, alloy="4340qt", material="hardened-steel",
    note="torqued side closure: axial preload is its constitutive load; "
         "two closures make the shaped clamp removable and adjustable")

# ---- things that turn -----------------------------------------------
for _k, _alloy, _zero, _bush in (
        ("gimbal-yaw-bearing", "4340qt", False, True),
        ("gimbal-pitch-bearing", "4340qt", False, True),
        ("pinned-trunnion-mount", "4340qt", True, False),
        ("spherical-thrust-seat", "4340qt", True, False),
        ("pinion-carrier-bearing", "4340qt", False, False),
        ("captive-pinion-mesh", "4340qt", False, False),
        ("socket-joining", "4130n", True, False),
        ("universal-joint", "4130n", True, False),
        ("bushing-mount", "4130n", True, True)):
    _mc(_k, PINNED, alloy=_alloy, material="hardened-steel",
        zero_length=_zero, bushed=_bush)
_mc("actuated-damped-clutch-gimbal-base", PINNED, bushed=True, alloy="hy80",
    material="hardened-steel",
    note="a bearing that can be clamped. Declared released, because "
         "released is the case the structure has to survive")
_mc("firing-lock-clamp", PINNED, alloy="4340qt", material="hardened-steel")

# ---- things that slide or push ---------------------------------------
_mc("single-axis-slider", SLIDING, release_ends=ONE_END, travels=True,
    alloy="4340qt", material="hardened-steel",
    note="the carriage on its rail: rigid in five freedoms, free in the "
         "sixth, and being free in the sixth is the entire point of it")
_mc("prismatic-guide-strut", SLIDING, release_ends=ONE_END, travels=True,
    alloy="4340qt", material="hardened-steel")
_mc("oleo-recoil-slide", SLIDING, release_ends=ONE_END, travels=True,
    spring_like=True, alloy="300m", material="hardened-steel",
    note="landing-gear steel, and along its axis it transmits only what "
         "its own fluid law says it does")
_mc("spring-damper", SLIDING, release_ends=ONE_END, travels=True,
    spring_like=True)
for _k in ("linear-hydraulic-actuator", "commanded-rest-length-elevation-ram",
           "outrigger-lift-jack", "chain-winch-hoist"):
    _mc(_k, SLIDING, release_ends=ONE_END, travels=True, spring_like=True,
        bushed=True, alloy="4340qt", material="hardened-steel")
for _k in ("fine-aim-twitch-actuator", "belleville-preload-stack"):
    _mc(_k, SLIDING, release_ends=ONE_END, travels=True, spring_like=True,
        alloy="4340qt", material="hardened-steel")
_mc("preloaded-captive-body-retainer-spring", SLIDING, release_ends=ONE_END,
    travels=True, spring_like=True)
_mc("bump-stop-contact", SLIDING, release_ends=ONE_END, travels=True,
    alloy="4340qt", material="polyurethane",
    note="normally open unilateral end-stop: it transmits compression only "
         "after the declared slide clearance is exhausted")

# ---- a turntable running on its own face ------------------------------
#: A FLAT GREASED TRACK IS NOT A BEARING IN A HOUSING. It carries
#: compression and shear through a film, and it is free in exactly one
#: rotation: about its own axis. In a member's local frame the member
#: IS that axis, so what a vertical thrust pad releases is rx -- the
#: same freedom a pinned clevis deliberately does NOT release. Writing
#: it the other way round would be a turntable that could not turn and
#: a pad that could not carry the machine.
THRUST_TRACK = (RIGID, RIGID, RIGID, FREE, RIGID, RIGID)

_mc("greased-thrust-track", THRUST_TRACK, alloy="4340qt",
    material="hardened-steel",
    note="a floor resting on the face of the ring below it, on grease: "
         "the whole annulus is the bearing surface")
_mc("slew-thrust-roller", THRUST_TRACK, alloy="4340qt",
    material="hardened-steel",
    note="a heavy roller in that track, taking its share of the load "
         "where the track alone would be asked for too much film")
_mc("slew-holddown-roller", THRUST_TRACK, alloy="4340qt",
    material="hardened-steel",
    note="the same roller under the lip, for the side of the ring the "
         "overhang is trying to LIFT -- a flat track carries compression "
         "and nothing else, so without these the turntable hinges")

# ---- a face that clamps to direct drive ---------------------------
#: DIRECT DRIVE ON LOCK, and the difference from teeth is the whole
#: reason to prefer it. A dog is positive: once home it cannot slip at
#: all, so anything arriving from the far side goes straight through it
#: into whatever is behind. A lock-up face is clamped hard enough that
#: it does not slip in service -- direct drive for every purpose that
#: matters -- and still lets go above its slip torque, which is an
#: overload path a tooth does not have.
#:
#: It also engages at any speed. There is no sync window, because a
#: face that can slip does its own matching on the way in.
_mc("direct-drive-lockup", WELDED, travels=True, spring_like=False,
    alloy="4340qt", material="hardened-steel",
    note="clamped to direct drive; slips above its rated torque instead "
         "of passing a shock on")

# ---- things that carry fluid or signal, not load ----------------------
#: The beam solver already knew these were not members. The frame solver
#: did not, so eighty-four coolant lines were being assembled as
#: structural steel around the barrel -- a water jacket stiffening the
#: gun in the model and doing nothing of the kind in metal.
for _k in ("coolant-line", "oil-line", "exhaust-flow-path", "fuel-line",
           "air-line", "pressure-rated-hydraulic-line",
           "flexible-hydraulic-hose", "pressure-rated-air-line",
           "flexible-air-line", "insulated-copper-wire", "refrigerant-line",
           "flexible-multi-circuit-conduit", "metaconduit-isolated-channel",
           "insulated-flexible-thermal-duct",
           "insulated-flexible-exhaust-duct",
           # A service shaft belongs to the runnable rigid power graph.
           # Its torque reaction reaches the beam model through the
           # mounted machine body; it is not itself a beam member.
           "shaft-service-drive",
           # ---- the autoclave's three, declared rather than borrowed ----
           # A steam line is not a hydraulic line with a different name.
           # It carries a compressible fluid whose temperature is fixed
           # by its pressure, it must be lagged or it condenses in the
           # run, and it drains downhill to a trap. Borrowing
           # "pressure-rated-air-line" would have assembled it as the
           # right kind of member with the wrong contents, which is
           # exactly the silent-misspelling failure this table exists to
           # end.
           "steam-line",
           # Condensate runs by GRAVITY and carries whatever the cycle
           # put into it -- water, and on a dewax cycle molten wax that
           # is recovered rather than discarded.
           "condensate-line",
           # A vacuum line is the only one here whose inside is at LOWER
           # pressure than its outside, so it fails by collapsing rather
           # than by bursting and is the one that needs wall support.
           "vacuum-line"):
    _mc(_k, WELDED, routed=True, material="steel-pipe")
_mc("port-face-seal", WELDED, routed=True, zero_length=True,
    material="steel-pipe")
# An electrical conduit carries wire, not fluid, and no load unless it
# says so -- a steel conduit declared `load_bearing` on the edge holds
# its junction box up by its own bending, which is what conduit does.
_mc("electrical-conduit", WELDED, routed=True, material="steel-pipe")


#: The tokens, by name: `joints.T.single_axis_slider`, `joints.T.coolant_line`.
T = _TokenNamespace()

#: Sets of tokens the hot paths test membership against, built once.
ROUTED_TOKENS = frozenset(m.token for m in MEMBER_CONSTRAINTS.values()
                          if m.routed)
SLIDING_TOKENS = frozenset(m.token for m in MEMBER_CONSTRAINTS.values()
                           if "x" in m.freedoms())
PINNED_TOKENS = frozenset(m.token for m in MEMBER_CONSTRAINTS.values()
                          if set(m.freedoms()) >= {"rx", "ry", "rz"})
SPRING_TOKENS = frozenset(m.token for m in MEMBER_CONSTRAINTS.values()
                          if m.spring_like)
ZERO_LENGTH_TOKENS = frozenset(m.token for m in MEMBER_CONSTRAINTS.values()
                               if m.zero_length)


# =====================================================================
#  A SEAM: fasteners along a length, which is how sheet is joined
# =====================================================================
# "A sheet metal screw every three inches" is the whole specification a
# tinsmith gives, and it is enough, because everything else follows:
# how many screws there are is the length over the pitch; what each
# one carries is set by the sheet it is in, not by the screw; and what
# each one is worth as a stiffness is the screw bearing on the hole it
# cut, which is the sheet's modulus times its thickness, scaled by how
# much of the hole is actually in bearing. Spot welds and snap hooks
# are the same shape of declaration with different per-fastener
# numbers, so one record covers all three.

#: Per-fastener bearing stiffness as a fraction of E.t -- the sheet's
#: modulus times its thickness -- which is the classical lap-joint
#: fastener flexibility written the other way up. About a quarter for
#: a screw in a hole it cut itself, which is a loose fit; a spot weld
#: is a fused nugget and takes nearly all of it.
FASTENER_BEARING_FRACTION = {
    "sheet-metal-screw": 0.25,
    "tack-weld": 0.80,
    "spot-weld": 0.90,
    "snap-fit": 0.05,
}

#: Where a fastener's capacity comes from. A screw in thin sheet fails
#: by the sheet tearing or the thread stripping: bearing area (d.t)
#: times the sheet's ultimate, times a knock-down for tear-out. A spot
#: weld fails by shearing its nugget. A snap hook does not fail; it
#: releases at a declared force.
INCH_M = 0.0254


@dataclass(frozen=True)
class Seam:
    """Fasteners along a joint, declared the way a shop declares them."""
    fastener: str = "sheet-metal-screw"
    #: how many per inch of seam, because that is how it is specified.
    #: A screw every three inches is 1/3.
    per_inch: float = 1.0 / 3.0
    #: the sheet the fastener is in, which is what it is worth
    sheet_thickness_m: float = 0.0015
    sheet_material: str = "a36"
    #: the screw's nominal diameter (a #10 is 4.8 mm), a spot weld's
    #: nugget diameter, a snap hook's width
    fastener_diameter_m: float = 0.0048
    #: SNAP HOOKS ONLY: the force each hook lets go at
    release_force_n: float = 0.0

    @property
    def per_m(self) -> float:
        return self.per_inch / INCH_M

    @property
    def bond(self) -> BondMaterial:
        return BONDS[self.fastener]

    def _sheet(self):
        from milspec import MATERIAL_BY_KEY
        return MATERIAL_BY_KEY[self.sheet_material]

    def stiffness_per_fastener_n_per_m(self) -> float:
        """The fastener bearing on the sheet it is in."""
        return (FASTENER_BEARING_FRACTION[self.fastener]
                * self._sheet().youngs_pa * self.sheet_thickness_m)

    def capacity_per_fastener_n(self) -> float:
        if self.fastener == "snap-fit":
            return float(self.release_force_n)
        if self.fastener == "spot-weld":
            nugget = math.pi * (self.fastener_diameter_m / 2.0) ** 2
            return nugget * self.bond.shear_strength_pa
        if self.fastener == "tack-weld":
            # a short fillet: its throat is the leg over root two and
            # its length is `fastener_diameter_m`, read as the tack's
            # length; the leg is the sheet's thickness, because a tack
            # on sheet cannot be bigger than the sheet
            throat = self.sheet_thickness_m * math.sqrt(0.5)
            return throat * self.fastener_diameter_m * self.bond.shear_strength_pa
        # a screw: the sheet bears on the shank and tears out at about
        # half its bearing ultimate, which is the ordinary allowance for
        # a hole near a sheet edge
        bearing = self.fastener_diameter_m * self.sheet_thickness_m
        return 0.5 * bearing * self._sheet().ultimate_pa

    def stiffness_n_per_m(self, length_m: float) -> float:
        return self.stiffness_per_fastener_n_per_m() * self.count(length_m)

    def capacity_n(self, length_m: float) -> float:
        return self.capacity_per_fastener_n() * self.count(length_m)

    def count(self, length_m: float) -> int:
        """Whole fasteners, and never fewer than two: one screw is a
        hinge, and a seam with one is not a seam."""
        return max(2, int(round(self.per_m * float(length_m))))

    def pack(self, length_m: float) -> dict:
        """What the edge carries, in the same field names the bushing
        pack uses for the numbers that mean the same thing -- so a walk
        that reads a bushing's linear stiffness reads a seam's the same
        way."""
        n = self.count(length_m)
        return {
            "model": "fastened-seam",
            "fastener": self.fastener,
            "per_inch": self.per_inch,
            "count": n,
            "sheet_thickness_m": self.sheet_thickness_m,
            "sheet_material": self.sheet_material,
            "linear_stiffness_n_per_m": self.stiffness_per_fastener_n_per_m() * n,
            "capacity_n": self.capacity_per_fastener_n() * n,
            "releases": self.fastener == "snap-fit",
            # a seam has no damper in it; what damping it has is the
            # sheet rubbing on itself, and that is small
            "damping_ratio": 0.03,
        }

    def describe(self, length_m: float) -> str:
        n = self.count(length_m)
        return (f"{self.fastener} every {1.0 / self.per_inch:.1f} in "
                f"({n} over {length_m:.2f} m) in {self.sheet_thickness_m * 1000:.1f} mm "
                f"{self.sheet_material}: {self.stiffness_n_per_m(length_m) / 1e6:.1f} MN/m, "
                f"{'releases' if self.fastener == 'snap-fit' else 'holds'} "
                f"{self.capacity_n(length_m) / 1000:.1f} kN")


# ---- the member spellings that carry a seam or attach to a shell -----
_mc("shell-attachment-weld", WELDED, zero_length=True, material="steel-plate",
    note="a nozzle, a wear plate, a rail foot: metal burned to the shell it "
         "sits on. Two surfaces in contact and everything transmitted, which "
         "is what a port-face-seal was being used to mean and is not")
_mc("screwed-seam", WELDED, material="steel-sheet",
    note="sheet lapped on sheet or on a frame flange, held by screws at a "
         "declared pitch; the edge carries a Seam pack with its stiffness "
         "and capacity, and the walk reads it as compliant")
_mc("spot-welded-seam", WELDED, material="steel-sheet",
    note="the unibody's seam: spot welds at a pitch along a flange")
_mc("piano-hinge", HINGED, release_ends=ONE_END, alloy="a36",
    material="steel-sheet",
    note="a door's continuous hinge: turns about its own line and nothing "
         "else. Released at one end only, like a slider, so it locates the "
         "door while letting it swing")
_mc("tack-welded-bracket", WELDED, material="steel-sheet",
    note="a bracket on sheet casing: tacks at a pitch, each a short fillet "
         "no bigger than the sheet; the edge carries a Seam pack of tacks")
_mc("snap-seam", WELDED, material="plastic-sheet",
    note="hooks at a pitch along a panel edge; releases at a force and is "
         "not damaged by releasing")
SEAM_CONSTRAINTS = frozenset(("screwed-seam", "spot-welded-seam", "snap-seam",
                              "tack-welded-bracket"))
