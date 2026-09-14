"""THE TWO-STAGE PLATFORM UNDER THE ARCHES.

TWO ARCHES, one each side, feet on the rotating platform's deck,
straddling the bulk of it. The arch is THE ONLY RIGID THING here;
everything hanging under it articulates.

Each arch carries TWO PINS, fore and aft. On those four pins -- two
arches, two pins each -- hangs the DANGLING PLATFORM: a rectangle on
four equal links, hanging straight down at rest. Arch, two links and
the platform close a four-bar on each side, so the rectangle
translates along its arc and never pitches.

The dangling platform carries four more pins of its own, INBOARD of
where its own links landed, and on those stand four equal links going
UP. On top of them sits the GUN PLATFORM. At rest those links are laid
forward, so the gun platform lies forward and low, resting on top of
the dangling platform. The gun is mounted to it.

    INBOARD AND OUTBOARD IS NOT A DETAIL. The dangling platform's
    links take the pin outboard; the gun platform's links stand
    inboard of them. That is the only reason the two stages can fold
    through each other instead of colliding.

WHAT THE STROKE DOES. Recoil drives the gun aft.

    the gun platform    its links rotate from laid-forward toward
                        upright, so it goes BACK AND UP, climbing
                        hardest at the end
    the dangling one    swings aft out of the bottom of its arc, so it
                        also goes BACK AND UP

Both stages climb, and because the gun platform's pins are carried by
the dangling platform the two displacements ADD.

Gravity is the recuperator -- it stores the stroke and returns the
assembly to battery with nothing to fatigue and nothing to leak. It is
not the damper: its resistance is nearly zero at bottom dead centre,
exactly where recoil force is largest, which is what the absorber and
the managed joints are for.

EVERY JOINT IS MANAGED. Each pin carries a spring, a controllable
damper and an actuator in parallel, declared on the joint rather than
assumed. That is what lets the assembly be PRELOADED forward against
its own stops before firing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: bore/fore-aft is +Z, up is +Y, across the machine is X.
FORWARD = np.array([0.0, 0.0, 1.0])
UP = np.array([0.0, 1.0, 0.0])
ACROSS = np.array([1.0, 0.0, 0.0])

#: how far a bay's metering valve can open and close, as a
#: fraction of the piston it meters.
MIN_APERTURE_FRAC = 0.0006
MAX_APERTURE_FRAC = 0.14


@dataclass
class PlatformStage:
    """One rectangle on four equal links, and everything managing it.

    `anchors` are four existing nodes -- the pins this stage hangs or
    stands on, given (aft-a, fwd-a, aft-b, fwd-b). The links are all
    `link_m` long and all at `rest_angle_deg`, which is what makes the
    stage a parallelogram on each side rather than four pendulums: the
    rectangle translates along its arc and its attitude never changes.

    `rises` says which way the links point off their pins:
        -1  hanging DOWN, the rectangle below the pins
        +1  standing UP, the rectangle above the pins

    `rest_angle_deg` is measured off that direction, positive forward.
    So a hanging stage at 0 deg dangles straight down, and a standing
    stage at 70 deg is laid nearly flat forward.
    """
    identity: str
    anchors: tuple
    link_m: float
    rises: float = -1.0
    rest_angle_deg: float = 0.0
    travel_deg: float = 45.0
    #: how far inboard this stage's links stand from the pins it was
    #: given. The upper stage needs this to clear the lower stage's
    #: links, which take the same pins outboard.
    inboard_m: float = 0.0
    material: str = "4340qt"
    spring_rate_n_per_m: float = 240_000.0
    damping_n_s_per_m: float = 90_000.0
    actuator_force_n: float = 60_000.0
    preload_n: float = 0.0
    carries_kg: float = 0.0
    #: each bay's damper piston. Four per stage.
    piston_area_m2: float = 0.0113
    link_radius_m: float = 0.050
    beam_radius_m: float = 0.058
    pin_radius_m: float = 0.045
    pin_length_m: float = 0.130

    def arm(self, angle_deg: float) -> np.ndarray:
        """Where the rectangle sits relative to its pins, at an angle."""
        a = math.radians(angle_deg)
        return (UP * (self.rises * self.link_m * math.cos(a))
                + FORWARD * (self.link_m * math.sin(a)))

    @property
    def swept(self) -> np.ndarray:
        """Rest to fully folded, as a displacement of the rectangle."""
        return (self.arm(self.rest_angle_deg - self.travel_deg)
                - self.arm(self.rest_angle_deg))

    @property
    def rise_m(self) -> float:
        return float(self.swept @ UP)

    @property
    def reach_m(self) -> float:
        return float(abs(self.swept @ FORWARD))

    def damper(self, identity: str = ""):
        """The adaptive damper this stage's bays are built from.

        `actuators.AdaptiveRecoilDamper` is the existing part -- the one
        that inverts the energy balance per shot to pick an aperture.
        The stage owns the geometry (how much travel a bay has), the
        damper owns the hydraulics, and neither invents the other's
        numbers. A bay's travel is the chord its diagonal sweeps, so it
        comes out of `arm()` like everything else here."""
        from actuators import AdaptiveRecoilDamper
        a0 = self.arm(self.rest_angle_deg)
        a1 = self.arm(self.rest_angle_deg - self.travel_deg)
        chord = float(np.linalg.norm(a1 - a0))
        return AdaptiveRecoilDamper(
            identity=identity or f"{self.identity}.damper",
            piston_area_m2=self.piston_area_m2,
            stroke_m=chord,
            target_stroke_m=chord * 0.80,
            spring_rate_n_per_m=self.spring_rate_n_per_m,
            spring_preload_n=self.preload_n,
            # THE VALVE HAS TO SUIT THE PISTON IT IS ON. The default
            # aperture range on `AdaptiveRecoilDamper` belongs to the
            # mount it was written for; on this piston at this recoil
            # velocity the flow is an order up, and a valve that cannot
            # open far enough does not soften the stroke -- it stops it
            # early and hard, which is the opposite of the job. So the
            # range is a fraction of the piston area, which is the only
            # thing it was ever really a fraction of.
            min_orifice_m2=self.piston_area_m2 * MIN_APERTURE_FRAC,
            max_orifice_m2=self.piston_area_m2 * MAX_APERTURE_FRAC)

    @property
    def stored_j(self) -> float:
        """What gravity takes back out of the stroke."""
        return self.carries_kg * 9.80665 * self.rise_m


def emit_platform_stage(g, stage: PlatformStage, *,
                        motion_group: str | None = None,
                        assembly: str | None = None) -> dict:
    """Four pins, four equal links, one rectangle.

    The four corners are written, then the rectangle is CLOSED -- four
    edge beams and two diagonals -- because a rectangle that is only
    four corners is four independent points, and the solver will say
    so. Closing it is what makes the four links act as two
    parallelograms instead of four pendulums.
    """
    if motion_group is not None:
        g.motion_group = motion_group
    if assembly is not None:
        g.assembly = assembly
    pos = {n["identity"]: np.asarray(n["reference_position"], float)
           for n in g.nodes}
    off = stage.arm(stage.rest_angle_deg)
    corner, pin = {}, {}
    tags = ("aft.a", "fwd.a", "aft.b", "fwd.b")
    for tag, anchor in zip(tags, stage.anchors):
        base = pos[anchor]
        # The pin this stage stands on. A stage standing INBOARD gets a
        # pin of its own, stood off the anchor on a rigid boss -- that
        # offset is what lets the two stages fold through each other.
        # A stage with no offset takes the anchor ITSELF as its pin;
        # writing a second node on top of the first would leave the
        # anchor connected to nothing and the new node floating.
        if stage.inboard_m:
            inb = -math.copysign(stage.inboard_m, float(base @ ACROSS))
            p = f"{stage.identity}.pin.{tag}"
            g.node(p, tuple(float(v) for v in base + ACROSS * inb),
                   "chassis-load-node", material=stage.material,
                   mass_in_total=False, mass_kg=7.0,
                   part_role="platform-pin", stage=stage.identity,
                   corner=tag, inboard_m=stage.inboard_m,
                   pin_level=2, shape="drum", drum_axis=(1.0, 0.0, 0.0),
                   drum_radius_m=stage.pin_radius_m,
                   drum_length_m=stage.pin_length_m,
                   half_extent_m=(stage.pin_length_m / 2.0,
                                  stage.pin_radius_m, stage.pin_radius_m))
            g.edge(f"{stage.identity}.pin_boss.{tag}", p, anchor,
                   "rigid-distance", radius=stage.pin_radius_m * 0.58,
                   rigid=True,
                   beam_solvable=False, palette="chassis-grey",
                   load_path="the-pin-stood-inboard-to-clear-the-other-stage")
        else:
            inb, p = 0.0, anchor
        c = f"{stage.identity}.corner.{tag}"
        g.node(c, tuple(float(v) for v in base + ACROSS * inb + off),
               "load-bearing-structure", material=stage.material,
               mass_in_total=False, mass_kg=64.0,
               part_role="platform-corner", stage=stage.identity, corner=tag,
               keeps_attitude=True, half_extent_m=(0.09, 0.07, 0.09))
        # THE LINK. Four of them, equal, or it is not a parallelogram.
        g.edge(f"{stage.identity}.link.{tag}", p, c,
               "pinned-trunnion-mount", radius=stage.link_radius_m,
               alloy=stage.material, palette="rollbar-silver",
               part_role="platform-link", stage=stage.identity, corner=tag,
               pin_bore_radius_m=stage.pin_radius_m,
               link_m=stage.link_m, rises=stage.rises,
               rest_angle_deg=stage.rest_angle_deg,
               travel_deg=stage.travel_deg,
               load_path="one-of-four-equal-links")
        corner[tag], pin[tag] = c, p
    # ---- CLOSE THE RECTANGLE: it is a platform, not four points ----
    for na, nb, what in (("aft.a", "fwd.a", "side-rail"),
                         ("aft.b", "fwd.b", "side-rail"),
                         ("aft.a", "aft.b", "cross-beam"),
                         ("fwd.a", "fwd.b", "cross-beam")):
        g.edge(f"{stage.identity}.deck.{na}-{nb}", corner[na], corner[nb],
               "rigid-distance", radius=stage.beam_radius_m,
               alloy=stage.material, palette="rollbar-silver",
               part_role="platform-deck", stage=stage.identity, member=what,
               beam_solvable=True, load_path=f"the-platform's-{what}")
    for na, nb in (("aft.a", "fwd.b"), ("fwd.a", "aft.b")):
        g.edge(f"{stage.identity}.brace.{na}-{nb}", corner[na], corner[nb],
               "rigid-distance", radius=0.034, alloy=stage.material,
               palette="rollbar-silver", part_role="platform-brace",
               stage=stage.identity,
               load_path="the-platform-braced-square-not-a-lozenge")
    # The rectangle carries a real sheet.  Its grid participates in the beam
    # solve and its surface is later used for contact between the two stages.
    from surfaces import Plate, emit_plate
    sheet_t = 0.012
    corner_position = {
        tag: np.asarray(next(n["reference_position"] for n in g.nodes
                             if n["identity"] == identity), float)
        for tag, identity in corner.items()}
    p00 = corner_position["aft.a"]
    sheet = emit_plate(
        g, Plate(identity=f"{stage.identity}.deck_sheet",
                 # The sheet rests on the top tangent of the perimeter pipe;
                 # it is not centred through the pipe and it does not extend
                 # beyond the four supporting centre lines in plan.
                 corner=tuple(float(v) for v in p00 + UP * (
                     stage.beam_radius_m + sheet_t / 2.0)),
                 span_u=tuple(float(v) for v in
                              (corner_position["aft.b"] - p00)),
                 span_v=tuple(float(v) for v in
                              (corner_position["fwd.a"] - p00)),
                 thickness_m=sheet_t, nu=2, nv=2,
                 material="steel-plate", alloy=stage.material,
                 attributes={"surface_role": "platform-deck-sheet",
                             "stage": stage.identity,
                             "contact_surface": True,
                             "mounting_face": "top-of-support-pipe",
                             "support_pipe_radius_m": stage.beam_radius_m,
                             "plan_boundary": "support-pipe-centrelines",
                             "overhang_m": 0.0}),
        motion_group=motion_group, assembly=assembly)
    for key, tag in (((0, 0), "aft.a"), ((2, 0), "aft.b"),
                     ((0, 2), "fwd.a"), ((2, 2), "fwd.b")):
        g.edge(f"{stage.identity}.deck_sheet.weld.{tag}",
               sheet["nodes"][key], corner[tag], "rigid-distance",
               radius=0.018, alloy=stage.material,
               palette="rollbar-silver", beam_solvable=True,
               part_role="platform-sheet-corner-weld",
               fastening="continuous-fillet-weld",
               load_path="deck-sheet-welded-to-platform-perimeter")
    # ---- EVERY BAY DAMPED, NOT EVERY SIDE ----
    # A stage has four bays: two sides, and on each side a fore and an
    # aft link. One damper per side damps the side's average and leaves
    # the pair of links free to share unevenly, which is exactly what
    # happens when the gun is off-centre or the deck is not level. So
    # each bay gets its own, and each is an `AdaptiveRecoilDamper`:
    # the aperture is set per shot from the impulse, so the stage stops
    # in the same place whether the round is 20 mm or 120 mm.
    d = stage.damper()
    damper_edges = []
    for side in ("a", "b"):
        for bay, other in (("fwd", "aft"), ("aft", "fwd")):
            damper_identity = f"{stage.identity}.damper.{bay}.{side}"
            g.edge(damper_identity,
                   pin[f"{bay}.{side}"], corner[f"{other}.{side}"],
                   "spring-damper", radius=0.030, alloy=stage.material,
                   palette="actuator-yellow", part_role="platform-damper",
                   # Clevis-ended telescoping hardware is a constitutive
                   # axial path, not a fixed-ended diagonal beam.  Its spring
                   # tangent is assembled by FrameSolver and its damping by
                   # GraphJointForces; admitting the drawn cylinder as a beam
                   # additionally invents transverse/bending restraint.
                   structural_participation=False,
                   stage=stage.identity, bay=f"{bay}.{side}",
                   spring_rate_n_per_m=d.spring_rate_n_per_m,
                   spring_preload_n=d.spring_preload_n,
                   linear_damping_n_s_per_m=stage.damping_n_s_per_m,
                   controllable_damping=True,
                   piston_area_m2=d.piston_area_m2,
                   stroke_m=d.stroke_m, target_stroke_m=d.target_stroke_m,
                   min_orifice_m2=d.min_orifice_m2,
                   max_orifice_m2=d.max_orifice_m2,
                   set_per_shot=True,
                   load_path="an-adaptive-damper-across-this-bay")
            damper_edges.append(damper_identity)
        # and one ram per side, which is what packs the stage forward
        # and holds the preload before firing
        g.edge(f"{stage.identity}.ram.{side}",
               pin[f"fwd.{side}"], corner[f"aft.{side}"],
               "linear-hydraulic-actuator", radius=0.036,
               alloy=stage.material, palette="actuator-yellow",
               part_role="platform-actuator", stage=stage.identity,
               structural_participation=False,
               bore_m=0.063, rod_m=0.036,
               holding_force_n=stage.actuator_force_n,
               commanded_rest_length_m=0.0, preload_n=stage.preload_n,
               minimum_preload_n=0.0,
               maximum_preload_n=stage.actuator_force_n,
               preload_command_frac=(stage.preload_n
                                     / max(stage.actuator_force_n, 1.0)),
               preload_energy_capacity_j=(stage.actuator_force_n
                                          * stage.reach_m),
               control="swing-preload-release-to-hang-or-pack-forward",
               load_path="the-ram-that-packs-and-unpacks-this-stage")

    # Physical end stops use the same four diagonal bay coordinates as the
    # dampers. One constitutive contact is scattered equally through all four
    # faces, so a centered stop cannot manufacture a roll couple.
    edge_by_id = {edge["identity"]: edge for edge in g.edges}
    node_position = {node["identity"]: np.asarray(
        node["reference_position"], float) for node in g.nodes}
    reference_lengths = tuple(float(edge_by_id[name]["rest_length"])
                              for name in damper_edges)
    target_off = stage.arm(stage.rest_angle_deg - stage.travel_deg)
    current_off = stage.arm(stage.rest_angle_deg)
    back_lengths = []
    for name in damper_edges:
        edge = edge_by_id[name]
        pa = node_position[edge["a"]]
        pb = node_position[edge["b"]]
        back_lengths.append(float(np.linalg.norm(
            pb - current_off + target_off - pa)))
    motion_signs = tuple(1.0 if back > reference else -1.0
                         for back, reference in zip(back_lengths,
                                                    reference_lengths))
    back_gap = float(np.mean([
        sign * (back - reference)
        for sign, back, reference in zip(motion_signs, back_lengths,
                                         reference_lengths)]))
    shares = tuple(1.0 / len(damper_edges) for _ in damper_edges)
    initial_preload_frac = (stage.preload_n
                            / max(stage.actuator_force_n, 1.0))
    release_adjustment = min(0.080, back_gap * 0.20)
    stop_common = dict(
        radius=0.034, alloy=stage.material, palette="actuator-yellow",
        structural_participation=False,
        coupled_slide_edges=tuple(damper_edges), slide_load_share=shares,
        coupled_reference_separation_m=reference_lengths,
        coupled_motion_sign=motion_signs,
        maximum_compression_m=0.025,
        linear_stiffness_n_per_m=12.0e6,
        cubic_stiffness_n_per_m3=1.2e10,
        compression_damping_n_s_per_m=90_000.0,
        equalized_across_bays=True)
    first = edge_by_id[damper_edges[0]]
    g.edge(f"{stage.identity}.stop.forward", first["a"], first["b"],
           "bump-stop-contact", kind="bump-stop",
           part_role="adjustable-forward-preload-stop", stage=stage.identity,
           contact_side="minimum-separation",
           # The normal planted pose is already the full-forward pose.  Its
           # installed ram force presses the stage against this set stop;
           # commanding zero withdraws the stop and releases the swing.
           clearance_m=0.0,
           nominal_clearance_m=0.0,
           release_adjustment_m=release_adjustment,
           actuated_setpoint=True,
           preload_command_frac=initial_preload_frac,
           installed_preload_command_frac=initial_preload_frac,
           installed_stage_preload_n=stage.preload_n * 2.0,
           maximum_stage_preload_n=stage.actuator_force_n * 2.0,
           load_path="four-face-forward-stop-reacting-packing-ram-preload",
           **stop_common)
    g.edge(f"{stage.identity}.stop.aft", first["a"], first["b"],
           "bump-stop-contact", kind="bump-stop",
           part_role="fixed-full-aft-maintenance-stop", stage=stage.identity,
           contact_side="maximum-separation", clearance_m=back_gap,
           load_path="four-face-full-aft-stop-at-parallelogram-travel-limit",
           **stop_common)
    return {"corners": corner, "pins": pin, "stage": stage,
            "deck_sheet": sheet,
            "anchors_for_next": (corner["aft.a"], corner["fwd.a"],
                                 corner["aft.b"], corner["fwd.b"])}


def emit_platform_deck_contacts(g, lower: dict,
                                upper: dict) -> tuple[str, ...]:
    """Emit unilateral contacts wherever the two real deck sheets overlap."""
    position = {node["identity"]: np.asarray(node["reference_position"], float)
                for node in g.nodes}
    lower_nodes = list(lower["deck_sheet"]["nodes"].values())
    upper_nodes = list(upper["deck_sheet"]["nodes"].values())
    lower_surface = lower_nodes[0].split(".node.", 1)[0]
    upper_surface = upper_nodes[0].split(".node.", 1)[0]
    lower_xyz = np.asarray([position[name] for name in lower_nodes])
    lo = lower_xyz[:, [0, 2]].min(axis=0)
    hi = lower_xyz[:, [0, 2]].max(axis=0)
    made = []
    for upper_node in upper_nodes:
        here = position[upper_node]
        horizontal = here[[0, 2]]
        if np.any(horizontal < lo - 1.0e-9) or np.any(
                horizontal > hi + 1.0e-9):
            continue
        lower_node = min(
            lower_nodes,
            key=lambda name: float(np.linalg.norm(
                position[name][[0, 2]] - horizontal)))
        identity = f"platform.deck_contact.{len(made)}"
        g.edge(identity, lower_node, upper_node, "bump-stop-contact",
               kind="bump-stop", radius=0.020, alloy="4340qt",
               palette="actuator-yellow", in_view=False,
               structural_participation=False,
               part_role="platform-sheet-contact",
               contact_side="minimum-separation", slide_axis=tuple(UP),
               clearance_m=0.012, maximum_compression_m=0.010,
               linear_stiffness_n_per_m=30.0e6,
               cubic_stiffness_n_per_m3=3.0e10,
               compression_damping_n_s_per_m=120_000.0,
               lower_surface=lower_surface,
               upper_surface=upper_surface,
               load_path="upper-deck-sheet-seated-on-lower-deck-sheet")
        made.append(identity)
    if not made:
        raise ValueError("platform deck sheets have no contact overlap")
    return tuple(made)


@dataclass
class TwoStagePlatform:
    """THREE mechanisms in series, and what they cost together.

    THE SLIDE IS A STAGE AND IT IS THE FIRST ONE. The gun runs in its
    cradle on the recoil rail -- the existing travel, the existing
    absorber -- and only then does anything fold. In a series the
    SOFTEST stage moves first, and the slide is a straight run with a
    1:1 ratio and no gravity term, so it is soft where the others are
    not. That matters because the first milliseconds of a shot are when
    the force is largest and when neither parallelogram has any
    leverage: the links are near the ends of their arcs, where an arc
    barely moves for the angle it turns.

    Displacements add and RATIOS MULTIPLY, which is why quoting any one
    stage's stroke understates the mechanism by a factor of three.
    """
    identity: str
    arch_span_m: float
    dangling: PlatformStage
    gun: PlatformStage
    preload_forward_n: float = 0.0
    #: the gun's own linear travel in the cradle -- the first stage
    slide_travel_m: float = 0.844
    slide_spring_n_per_m: float = 96_000.0
    slide_preload_n: float = 34_000.0
    slide_piston_area_m2: float = 0.0113

    @property
    def total_rise_m(self) -> float:
        """The slide contributes none: it is a horizontal run."""
        return self.dangling.rise_m + self.gun.rise_m

    @property
    def fold_reach_m(self) -> float:
        """Just the two parallelograms. The gun platform's pins are
        carried by the dangling platform, so its motion is the VECTOR
        SUM of the two sweeps -- not either alone, and not the sum of
        their magnitudes."""
        return float(abs((self.dangling.swept + self.gun.swept) @ FORWARD))

    @property
    def total_reach_m(self) -> float:
        """All three: the gun's travel in the cradle, plus the fold."""
        return self.slide_travel_m + self.fold_reach_m

    @property
    def gravity_j(self) -> float:
        return self.dangling.stored_j + self.gun.stored_j

    def slide_damper(self):
        """The absorber on the gun's own travel.

        UNLIKE THE TWO PLATFORMS, THIS ONE HAS NO GRAVITY BEHIND IT.
        The run is horizontal, so nothing returns the gun to battery
        except the spring -- which is why this spring is a real
        recuperator and stiff, and the platforms' are not and are not.
        """
        from actuators import AdaptiveRecoilDamper
        return AdaptiveRecoilDamper(
            identity=f"{self.identity}.slide",
            piston_area_m2=self.slide_piston_area_m2,
            stroke_m=self.slide_travel_m,
            target_stroke_m=self.slide_travel_m * 0.80,
            spring_rate_n_per_m=self.slide_spring_n_per_m,
            spring_preload_n=self.slide_preload_n,
            min_orifice_m2=self.slide_piston_area_m2 * MIN_APERTURE_FRAC,
            max_orifice_m2=self.slide_piston_area_m2 * MAX_APERTURE_FRAC)


def describe(p: TwoStagePlatform, recoil_j: float) -> list:
    d, u = p.dangling, p.gun
    left = max(recoil_j - p.gravity_j, 0.0)
    return [
        f"ONE COUPLED GRAPH -- {p.identity}",
        f"  arches          2, feet on the deck, {p.arch_span_m:.2f} m apart,"
        f" the only rigid members",
        "",
        f"  recoil journal  the production gun moves in its square journal",
        f"                  back {p.slide_travel_m * 1000:6.0f} mm"
        f"   up      0 mm; MR, oil and recuperator act in parallel",
        f"  upper platform  stands on 4 inboard pins"
        f" ({u.inboard_m * 1000:.0f} mm in), link {u.link_m:.3f} m,"
        f" rest {u.rest_angle_deg:+.0f} deg",
        f"                  back {u.reach_m * 1000:6.0f} mm"
        f"   up {u.rise_m * 1000:6.0f} mm"
        f"   carries {u.carries_kg:5.0f} kg -> {u.stored_j / 1000:5.1f} kJ",
        f"  lower platform  hangs from 4 arch pins,"
        f" link {d.link_m:.3f} m,"
        f" rest {d.rest_angle_deg:+.0f} deg",
        f"                  back {d.reach_m * 1000:6.0f} mm"
        f"   up {d.rise_m * 1000:6.0f} mm"
        f"   carries {d.carries_kg:5.0f} kg -> {d.stored_j / 1000:5.1f} kJ",
        "",
        f"  platform envelope back {p.fold_reach_m * 1000:.0f} mm,"
        f"  up {p.total_rise_m * 1000:.0f} mm",
        f"  geometric envelope back {p.total_reach_m * 1000:.0f} mm",
        f"  motion order     none -- all freedoms solve together from force",
        f"  gravity takes   {p.gravity_j / 1000:.1f} kJ of"
        f" {recoil_j / 1000:.1f} kJ"
        f"  ({p.gravity_j / max(recoil_j, 1) * 100:.0f}%)",
        f"  left to damp    {left / 1000:.1f} kJ  ->"
        f" {left / max(p.total_reach_m, 1e-6) / 1000:.0f} kN mean"
        f" over the whole stroke",
        f"  preloaded       {p.preload_forward_n / 1000:.0f} kN forward",
    ]
