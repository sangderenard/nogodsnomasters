"""THE STAND: the work area, the two engine bays, and the legs.

WHAT SITS ON WHAT, bottom up:

    the GROUND          wherever it is, and the pads have to not sink
                        into it
    four OUTRIGGER LEGS the real hydraulic ones, `outriggers.
                        OutriggerSet` -- a pump, four cylinders, one
                        circuit. They lift the WHOLE site, not just the
                        turret: the work area and both engines come up
                        together and stand at outrigger height.
    the STAND           a rectangle in plan. Long across the machine,
                        short fore and aft.
        the WORK AREA   in the middle, directly under the turret, open
                        so somebody can get under it
        two ENGINE BAYS one hung off each short end, outboard of the
                        work area, on the side axis
    the TURRET PLATFORM above all of it -- the drum, the arches, the
                        two platforms and the gun

    THE LEGS CARRY THE ENGINES TOO, and that is the point of putting
    them on the stand rather than on the ground beside it. A prime
    mover sitting on the ground while the machine it feeds stands a
    metre and a half above it needs flexible line to cross that gap,
    and every one of those crossings is a joint that moves whenever the
    legs move. On the stand, the engines and the thing they power are
    one rigid body once the legs are set, and the plumbing never has to
    cross a moving gap at all.

WHY THE ENGINES ARE ON THE SHORT ENDS. The gun swings fore and aft on
its arches and the recoil stroke runs that way too; the fore-aft
direction is where the machine needs room. Across the machine is the
direction with nothing happening in it, so that is where the mass goes.
It is also the direction the outriggers are widest in, so hanging two
engines out there loads the legs that are best placed to take it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: across the machine is X, up is Y, bore/fore-aft is Z -- the same
#: convention the sled and the drum are authored in.
ACROSS = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])


@dataclass
class Stand:
    """The site: a work area with an engine bay on either side."""
    identity: str = "stand"
    #: the open square under the turret, side to side and fore to aft
    work_area_m: float = 3.40
    #: how far out along X each engine bay runs from the work area
    bay_length_m: float = 2.30
    #: the bays are only as deep as they need to be fore and aft
    bay_depth_m: float = 2.20
    #: the top of the stand, which is where the drum's floor sits
    deck_y: float = 0.000
    #: how far the legs stand the whole site off the ground
    lift_m: float = 1.600
    #: How far the outer legs lean outward from vertical. THIS IS NOT
    #: STYLING. A raked leg does two things a vertical one cannot: its
    #: pad lands further out than the frame it hangs from, which widens
    #: the footprint in the direction the gun actually fires, and it
    #: takes the horizontal part of the shot as AXIAL COMPRESSION down
    #: its own length instead of as bending across it. A vertical leg
    #: asked to brace a lateral shot is a cantilever with a pin at the
    #: bottom, and it answers in the one way a strut is weakest.
    leg_rake_deg: float = 34.0
    #: where the raked legs sit for transport. Vertical and tucked in
    #: under the frame, so the machine's road width is the frame's width
    #: and not the deployed footprint's -- which is 8.00 m across and
    #: 5.56 m fore and aft, and neither of those goes down a road.
    leg_stow_deg: float = 0.0
    #: how far down the leg the deploy ram takes hold, as a fraction of
    #: its length. Low is a long lever and a gentle force; high is a
    #: short one and a hard shove, and it has to clear the frame.
    deploy_lug_frac: float = 0.42
    #: where the deploy ram's other end is anchored, inboard of the
    #: pivot and above the deck
    deploy_anchor_in_m: float = 0.55
    deploy_anchor_up_m: float = 0.40
    frame_radius_m: float = 0.085
    deck_radius_m: float = 0.060
    material: str = "4130n"
    #: what each bay is asked to carry, for the legs to be sized against
    engine_mass_kg: float = 0.0

    @property
    def half(self) -> float:
        return self.work_area_m / 2.0

    @property
    def span_x(self) -> float:
        """The whole footprint across the machine, bays included."""
        return self.work_area_m + 2.0 * self.bay_length_m

    @property
    def reach_m(self) -> float:
        """How much further out a raked pad lands than its own leg top."""
        return self.lift_m * math.tan(math.radians(self.leg_rake_deg))

    @property
    def leg_length_m(self) -> float:
        """A raked leg is longer than the height it lifts."""
        return self.lift_m / math.cos(math.radians(self.leg_rake_deg))

    @property
    def stations(self) -> tuple:
        """EIGHT LEGS, and they are not eight of the same thing.

        Four OUTER legs at the corners of the whole footprint, raked
        outward, doing the bracing and the widening. Four INNER legs
        directly under the work area's own corners, near enough
        vertical, doing the carrying.

        The inner four are the ones that matter most and are the least
        obvious. With only the outer four, the work-area corners sit
        2.3 m inboard of the nearest support, so the entire turret --
        thirteen tonnes of drum, arch, platform and gun -- lands in the
        middle of a beam propped only at its ends. Putting a leg under
        each turret post takes that span to nothing.

        Returns (tag, x, z, rake_x, rake_z, role).
        """
        x = self.span_x / 2.0
        z = max(self.half, self.bay_depth_m / 2.0)
        h = self.half
        out = []
        for tag, sx, sz in (("a", +1, +1), ("b", +1, -1),
                            ("c", -1, +1), ("d", -1, -1)):
            # THE RAKE GOES WHERE THE FOOTPRINT IS SHORT. Across the
            # machine it is already 8 m and needs nothing; fore and aft
            # it is 3.4 m and that is the direction the gun fires in, so
            # the outward lean is spent there and only a little of it
            # sideways.
            out.append((f"outer.{tag}", sx * x, sz * z,
                        sx * 0.25, sz * 1.0, "brace"))
        for tag, sx, sz in (("fr", +1, +1), ("rr", +1, -1),
                            ("fl", -1, +1), ("rl", -1, -1)):
            out.append((f"inner.{tag}", sx * h, sz * h, 0.0, 0.0, "carry"))
        return tuple(out)

    def leg_axis(self, angle_deg: float, rx: float, rz: float) -> np.ndarray:
        """A unit vector down a leg leaning `angle_deg` off vertical.

        The same function answers for stowed and deployed, because they
        are the same leg at two angles -- which is the whole point of
        putting a ram on the pivot instead of authoring two geometries
        and hoping they stay related."""
        lean = np.array([rx, 0.0, rz], float)
        n = float(np.linalg.norm(lean))
        if n <= 0.0:
            return -UP
        a = math.radians(angle_deg)
        return -UP * math.cos(a) + (lean / n) * math.sin(a)

    def deploy_geometry(self, x: float, z: float, rx: float, rz: float):
        """Where the deploy ram's two ends are, stowed and deployed.

        Returns (anchor, lug_stowed, lug_deployed, stroke). The stroke
        is not chosen: it is the difference between the two lengths the
        ram has to reach, and if that comes out longer than a cylinder
        of this bore sensibly has, the lug is in the wrong place rather
        than the cylinder being wrong."""
        lean = np.array([rx, 0.0, rz], float)
        n = float(np.linalg.norm(lean))
        inward = -(lean / n) if n > 0.0 else -FORWARD
        top = np.array([x, self.deck_y, z], float)
        anchor = top + inward * self.deploy_anchor_in_m \
            + UP * self.deploy_anchor_up_m
        reach = self.leg_length_m * self.deploy_lug_frac
        stowed = top + self.leg_axis(self.leg_stow_deg, rx, rz) * reach
        deployed = top + self.leg_axis(self.leg_rake_deg, rx, rz) * reach
        stroke = abs(float(np.linalg.norm(deployed - anchor))
                     - float(np.linalg.norm(stowed - anchor)))
        return anchor, stowed, deployed, stroke

    @property
    def stowed_span_x(self) -> float:
        """Road width: the frame, because the legs are tucked inside it."""
        return self.span_x

    @property
    def braced_half_z(self) -> float:
        """The fore-aft half-footprint the PADS make, not the frame."""
        z = max(self.half, self.bay_depth_m / 2.0)
        return z + self.reach_m


def emit_stand(g, stand: Stand, *, motion_group: str = "frame",
               assembly: str = "stand") -> dict:
    """Write the stand, the two bays and the four legs.

    THE WORK AREA IS A HOLE AND IS MEANT TO BE. Its corners are framed
    and its edges are beamed, and nothing crosses the middle: the whole
    reason for the geometry is that somebody can stand under the turret
    while it is up on its legs. Filling it in would make a better
    structure and a worse machine."""
    g.motion_group, g.assembly = motion_group, assembly
    h, y = stand.half, stand.deck_y
    made = {"corners": {}, "bays": {}, "legs": {}}

    # ---- THE WORK AREA: four posts and the ring beam round them ----
    ring = (("fr", +h, +h), ("fl", -h, +h), ("rl", -h, -h), ("rr", +h, -h))
    for tag, x, z in ring:
        ident = f"{stand.identity}.work.{tag}"
        g.node(ident, (x, y, z), "chassis-load-node", material=stand.material,
               mass_in_total=False, mass_kg=120.0,
               part_role="work-area-corner", carries="the-turret-above",
               half_extent_m=(0.14, 0.12, 0.14))
        made["corners"][tag] = ident
    for a, b in (("fr", "fl"), ("fl", "rl"), ("rl", "rr"), ("rr", "fr")):
        g.edge(f"{stand.identity}.work_beam.{a}-{b}", made["corners"][a],
               made["corners"][b], "rigid-distance",
               radius=stand.frame_radius_m, alloy=stand.material,
               palette="chassis-grey", beam_solvable=True,
               part_role="work-area-beam",
               load_path="the-ring-beam-round-the-open-work-area")
    # braced in plan, not decked over: the middle stays open
    for a, b in (("fr", "rl"), ("fl", "rr")):
        g.edge(f"{stand.identity}.work_brace.{a}-{b}", made["corners"][a],
               made["corners"][b], "rigid-distance",
               radius=0.034, alloy=stand.material, palette="chassis-grey",
               part_role="work-area-brace", beam_solvable=True,
               load_path="the-work-area-kept-square-without-closing-it")

    # ---- THE TWO ENGINE BAYS, one on each short end ----
    bz = stand.bay_depth_m / 2.0
    for side, sx in (("port", -1.0), ("starboard", +1.0)):
        outer = sx * (h + stand.bay_length_m)
        inner = sx * h
        pts = {}
        for tag, x, z in (("of", outer, +bz), ("or", outer, -bz),
                          ("if", inner, +bz), ("ir", inner, -bz)):
            ident = f"{stand.identity}.bay.{side}.{tag}"
            g.node(ident, (x, y, z), "chassis-load-node",
                   material=stand.material, mass_in_total=False,
                   mass_kg=90.0, part_role="engine-bay-corner",
                   bay=side, half_extent_m=(0.12, 0.11, 0.12))
            pts[tag] = ident
        made["bays"][side] = pts
        for a, b in (("of", "or"), ("if", "ir"), ("of", "if"), ("or", "ir")):
            g.edge(f"{stand.identity}.bay_beam.{side}.{a}-{b}", pts[a], pts[b],
                   "rigid-distance", radius=stand.frame_radius_m,
                   alloy=stand.material, palette="chassis-grey",
                   beam_solvable=True, part_role="engine-bay-beam",
                   load_path="the-engine-bay-frame")
        for a, b in (("of", "ir"), ("or", "if")):
            g.edge(f"{stand.identity}.bay_brace.{side}.{a}-{b}", pts[a],
                   pts[b], "rigid-distance", radius=0.030,
                   alloy=stand.material, palette="chassis-grey",
                   beam_solvable=True,
                   load_path="the-bay-braced-square")
        # THE BAY HANGS OFF THE WORK AREA. Its inboard corners are not
        # the work area's corners -- the bay is shallower fore and aft
        # than the work area is -- so it is tied back to both of the
        # work area's corners on that side. A bay held at one point is a
        # cantilever with an engine on the end of it.
        for tag, near in (("if", ("fr" if sx > 0 else "fl")),
                          ("ir", ("rr" if sx > 0 else "rl"))):
            g.edge(f"{stand.identity}.bay_tie.{side}.{tag}", pts[tag],
                   made["corners"][near], "rigid-distance",
                   radius=stand.frame_radius_m, alloy=stand.material,
                   palette="chassis-grey", beam_solvable=True,
                   part_role="engine-bay-tie",
                   load_path="the-bay-carried-by-the-work-area-frame")

    # ---- THE LEGS: they lift the site, engines included ----
    g.assembly = "outriggers"
    for tag, x, z, rx, rz, role in stand.stations:
        braced = role == "brace"
        # AN INNER LEG HANGS OFF THE WORK CORNER ITSELF. Writing a
        # second node at the same place gave a member of zero length
        # between two coincident nodes, which `check()` says is not a
        # member and is right to. The corner is the leg top.
        if not braced:
            top = made["corners"][tag.split(".")[-1]]
        else:
            top = f"{stand.identity}.leg.{tag}.top"
            g.node(top, (x, y, z), "chassis-load-node",
                   material=stand.material, mass_in_total=False,
                   mass_kg=70.0, part_role="outrigger-pivot", leg_role=role,
                   swings_out=True, half_extent_m=(0.14, 0.13, 0.14))
            for k, near in enumerate(_nearest_frame(g, stand, made, x, z, 2)):
                g.edge(f"{stand.identity}.leg_seat.{tag}.{k}", top, near,
                       "rigid-distance", radius=stand.frame_radius_m,
                       alloy=stand.material, palette="chassis-grey",
                       beam_solvable=True,
                       load_path="the-pivot-the-leg-swings-on")
        axis = stand.leg_axis(stand.leg_rake_deg if braced else 0.0, rx, rz)
        length = stand.leg_length_m if braced else stand.lift_m
        foot = np.array([x, y, z], float) + axis * length
        pad = f"{stand.identity}.leg.{tag}.pad"
        g.node(pad, tuple(float(v) for v in foot), "load-bearing-structure",
               material="steel-plate", mass_in_total=False, mass_kg=95.0,
               part_role="outrigger-pad", fixed_to="world",
               stands_on="the-ground", leg_role=role,
               half_extent_m=(0.20, 0.06, 0.20))
        g.edge(f"{stand.identity}.leg.{tag}", top, pad,
               "linear-hydraulic-actuator", radius=0.070,
               alloy="4340qt", palette="actuator-yellow",
               part_role="outrigger-leg", leg_role=role,
               bore_m=0.100, rod_m=0.060,
               stroke_m=round(length, 4), travel_m=round(length, 4),
               rake_deg=(stand.leg_rake_deg if braced else 0.0),
               piston_area_m2=math.pi * 0.100 ** 2 / 4.0,
               lifts="the-whole-site-work-area-and-both-engines",
               load_path=("the-leg-bracing-the-shot-into-the-ground"
                          if braced else
                          "the-leg-under-the-turret-carrying-it"))
        made["legs"][tag] = (top, pad)

        # ---- THE DEPLOY RAM: what makes the leg leave its stow ----
        # TWO CYLINDERS PER CORNER, AND THEY DO DIFFERENT JOBS. The leg
        # is one: it extends downward and stands the machine up. This
        # one swings the leg from tucked under the frame out to its
        # bracing angle, and it is what turns a road-legal package into
        # a 5.56 m footprint without anybody carrying anything.
        #
        # The two run TOGETHER, not in sequence: the leg reaches for the
        # ground while the ram pushes it outward, so the pad travels
        # down and out along a curve and never sweeps sideways across
        # ground it is already touching. A leg deployed first and
        # extended second would drag its pad through whatever it is
        # standing on.
        if not braced:
            continue
        anchor, stowed, deployed, stroke = stand.deploy_geometry(x, z, rx, rz)
        lug = f"{stand.identity}.leg.{tag}.lug"
        anc = f"{stand.identity}.leg.{tag}.anchor"
        g.node(lug, tuple(float(v) for v in deployed), "chassis-load-node",
               material="4340qt", mass_in_total=False, mass_kg=16.0,
               part_role="outrigger-deploy-lug",
               at_fraction=stand.deploy_lug_frac,
               half_extent_m=(0.07, 0.07, 0.07))
        g.node(anc, tuple(float(v) for v in anchor), "chassis-load-node",
               material=stand.material, mass_in_total=False, mass_kg=22.0,
               part_role="outrigger-deploy-anchor",
               half_extent_m=(0.08, 0.08, 0.08))
        g.edge(f"{stand.identity}.deploy_lug.{tag}", lug, top,
               "rigid-distance", radius=0.034, rigid=True,
               beam_solvable=False, alloy="4340qt",
               palette="rollbar-silver",
               load_path="the-lug-is-part-of-the-leg")
        g.edge(f"{stand.identity}.deploy_lug_foot.{tag}", lug, pad,
               "rigid-distance", radius=0.034, rigid=True,
               beam_solvable=False, alloy="4340qt",
               palette="rollbar-silver",
               load_path="the-lug-is-part-of-the-leg")
        g.edge(f"{stand.identity}.deploy_anchor.{tag}", anc, top,
               "rigid-distance", radius=0.044, alloy=stand.material,
               palette="chassis-grey", beam_solvable=True,
               load_path="the-bracket-the-deploy-ram-pushes-from")
        g.edge(f"{stand.identity}.deploy_ram.{tag}", anc, lug,
               "linear-hydraulic-actuator", radius=0.048,
               alloy="4340qt", palette="actuator-yellow",
               part_role="outrigger-deploy-ram", leg_role=role,
               bore_m=0.080, rod_m=0.050,
               stroke_m=round(stroke, 4), travel_m=round(stroke, 4),
               piston_area_m2=math.pi * 0.080 ** 2 / 4.0,
               swings_from_deg=stand.leg_stow_deg,
               swings_to_deg=stand.leg_rake_deg,
               load_path="the-ram-that-swings-the-leg-out-as-it-extends")
    g.assembly = assembly
    return made


def _nearest_frame(g, stand: Stand, made: dict, x: float, z: float,
                   count: int = 1):
    """Which frame node a leg stands under.

    The legs are at the extremes of the whole footprint, which is the
    BAYS' outer corners, not the work area's -- so a leg finds the
    nearest structural node rather than assuming it belongs to the work
    area. On a site whose bays are shorter than its work area the same
    code puts the legs on the work area instead, which is the right
    answer for that shape and not a special case."""
    pos = {n["identity"]: np.asarray(n["reference_position"], float)
           for n in g.nodes}
    here = np.array([x, stand.deck_y, z])
    candidates = list(made["corners"].values())
    for pts in made["bays"].values():
        candidates.extend(pts.values())
    ordered = sorted(candidates,
                     key=lambda k: float(np.linalg.norm(pos[k] - here)))
    return ordered[:count] if count > 1 else ordered[0]


def outrigger_set(stand: Stand, carried_kg: float):
    """The real hydraulic set, sized against what it is asked to lift.

    `outriggers.OutriggerSet` is the existing one: a pump, four
    cylinders, one circuit, and a pad that sinks if it is pushing
    harder than the ground can carry. What it gets told here is the
    whole site's mass -- the turret, the stand, AND both engines --
    because that is what the legs actually stand up."""
    from outriggers import OutriggerSet
    return OutriggerSet(machine_mass_kg=float(carried_kg),
                        stroke_m=stand.lift_m, stages=1)


def describe(stand: Stand, carried_kg: float) -> list:
    legs = len(stand.stations)
    per_leg = carried_kg / legs
    pad = 0.36 * 0.36
    return [
        f"THE STAND -- {stand.span_x:.2f} m across, "
        f"{max(stand.work_area_m, stand.bay_depth_m):.2f} m fore and aft",
        f"  work area       {stand.work_area_m:.2f} m square, open,"
        f" directly under the turret",
        f"  engine bays     2, one each side, {stand.bay_length_m:.2f} m"
        f" out by {stand.bay_depth_m:.2f} m deep",
        f"  lift            {stand.lift_m * 1000:.0f} mm -- the work area"
        f" and both engines stand at that height",
        f"  legs            {legs}: 4 raked"
        f" {stand.leg_rake_deg:.0f} deg outward at the corners,"
        f" 4 upright under the work area",
        f"  deploy rams     4, one a raked leg: they swing it from"
        f" {stand.leg_stow_deg:.0f} deg stowed to"
        f" {stand.leg_rake_deg:.0f} deg braced as the leg extends",
        f"  rake buys       {stand.reach_m * 1000:.0f} mm of reach a pad,"
        f" so the fore-aft footprint is"
        f" +-{stand.braced_half_z:.2f} m at the ground"
        f" against +-{max(stand.half, stand.bay_depth_m / 2.0):.2f} m"
        f" at the frame",
        f"  carrying        {carried_kg:.0f} kg  ->  {per_leg:.0f} kg a leg,"
        f" {per_leg * 9.81 / pad / 1000:.0f} kPa under a"
        f" {math.sqrt(pad) * 1000:.0f} mm pad",
    ]
