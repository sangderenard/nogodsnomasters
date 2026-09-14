"""THE TWO ARCHES AND THE TWO PLATFORMS THEY CARRY.

    python sled_reference.py            the drawing + the spec
    python sled_reference.py --render   the 3D stills
    python sled_reference.py --fold     animate the fold

WHAT IS HERE, bottom up:

    the DECK          the top of the rotating drum
    TWO ARCHES        one each side, both feet on the deck, straddling
                      the bulk of it. THE ONLY RIGID MEMBERS.
    four ARCH PINS    two per arch, fore and aft
    the DANGLING PLATFORM
                      a rectangle on four equal links hanging straight
                      down off those pins. Outboard on the pin.
    four INTERIOR PINS
                      carried by the dangling platform, INBOARD of its
                      own links, so the two stages fold through each
                      other instead of colliding
    the GUN PLATFORM  a rectangle on four equal links STANDING UP off
                      those interior pins, laid forward at rest, so it
                      lies forward and low on top of the dangling one.
                      The gun is mounted to it.

Recoil drives the gun aft. The gun platform's links rotate up toward
vertical -- back and up. The dangling platform swings aft out of the
bottom of its arc -- back and up. The two add.

AND THE STATIONS ARE THE POINT. Every link carries a row of attachment
holes. Where the absorber is pinned decides the moment arm, therefore
the force it must make and the stroke it must have. A 20 mm shot is
13 J and a 120 mm is 426 kJ, four orders apart, and the difference
between them is a pin in a different hole rather than a different
machine.
"""
from __future__ import annotations

import math
import sys
from copy import deepcopy
from functools import lru_cache

import numpy as np

from turret_production import ProductionGraph
from sled import (PlatformStage, TwoStagePlatform, emit_platform_stage,
                  emit_platform_deck_contacts, describe, UP, FORWARD)

# ---- THE DECK IS THE DRUM'S TOP. These are the drum's own numbers. ----
DECK_R = 1.588          # the boxed drum's outer radius
DECK_Y = 0.340          # its height plus the deck plate

ARCH_X = 0.950          # how far out each arch stands
ARCH_FOOT_Z = 1.150     # feet fore and aft -- r = 1.49 m, on the deck
ARCH_RISE = 1.800       # normal crown above the deck
ARCH_EXTREME_LIFT_M = 1.200  # hydraulic crown rise for near-vertical fire
ARCH_HAUNCH_RISE_M = 0.550
ARCH_LIFT_BASE_RISE_M = 1.050
ARCH_CROWN_SHOULDER_RISE_M = 1.620
PIN_Z = 0.780           # where the pins sit on the arch, fore and aft
PIN_DROP = 0.220        # pins below the crown
GUN_TRUNNION_CLEARANCE_M = 0.420
#: how far in the shoulders sit from the feet. 1.0 is a straight
#: leg; less leans the leg inboard, over the swept volume.
#: 1.0 = a straight leg. The 0.62 lean this started with was my own
#: invention and it put the legs over the swept volume: it cost 12
#: interferences on its own.
ARCH_SHOULDER_FRAC = 1.00
#: whether the two arches are tied across at shoulder height. That
#: tie spans the full width at exactly the height the gun platform
#: folds up into, so it is not free.
#: This tie was mine too, not asked for, and it was the single worst
#: offender -- 18 interferences, because it spans the full width at
#: exactly the height the gun platform folds up into.
ARCH_TIE = False

# ---- THE GUN ITSELF. Two diameters and a length; the rest follows.
BORE_M = 0.120
BARREL_M = 8.400
BARREL_OD_BREECH = 0.240
BARREL_OD_MUZZLE = 0.150
BREECH_KG = 900.0
BREECH_Z = 0.000            # the breech face, over the platform
HIGH_ADJUST_M = 0.480       # the coarse-lay column, at rest
HIGH_ADJUST_TRAVEL_M = 0.300
COUNTERWEIGHT_ARM_M = 2.500  # aft of the trunnion, clear of the breech
REAR_SUPPORT_M = 0.180       # the rear support above the aft cross beam
REAR_SUPPORT_TRAVEL_M = 0.520   # which is the elevation travel

# ---- THE GUN'S OWN LINEAR RECOIL, the existing mechanism ----
#: the travel the journal and its stops already had. It does not change
#: because the gun moved onto a cradle; it is the same stroke.
SLIDE_TRAVEL_M = 0.844
SLIDE_PISTON_AREA_M2 = 0.0113
#: how far along the bore each rail node stands off the
#: gun fitting it carries. It only has to be enough that the
#: slide member has a length and an unambiguous axis.
RAIL_STANDOFF_M = 0.400
#: the absorber returns the gun to battery along a HORIZONTAL run, so
#: unlike the two platforms there is no gravity to do it. This one is a
#: real recuperator and has to be stiff enough to be one.
SLIDE_SPRING_N_PER_M = 96_000.0
SLIDE_PRELOAD_N = 34_000.0

#: four links per stage means four bays, and they share the
#: stroke at one velocity.
BAYS_PER_STAGE = 4
#: how many plate nodes each arch foot pad bolts into.
PAD_BOLTS = 3


def _node_at(g, ident):
    for n in g.nodes:
        if n["identity"] == ident:
            return n["reference_position"]
    raise KeyError(ident)

STATIONS = (0.30, 0.50, 0.70, 0.90)


PRODUCTION_GUN_ASSEMBLIES = frozenset({
    "gun-gimbal", "breech", "gun-tube", "under-barrel-spine",
    "bore-evacuator", "muzzle-reference", "barrel-cooling", "cradle",
    # Keep the old basket feet as attachment hardpoints, but not its upper
    # twitch plate or six fine-aim actuators. They explicitly declare that
    # they carry no firing load; the arch platform's trunnion standards and
    # elevation anchors below are the installed gun support here.
    "fine-rig-base",
    "recoil-journal", "recoil-absorber", "trunnion-thrust-seat",
    "recoil-gear", "fabricated-shaped-strap", "fabricated-weld",
})


@lru_cache(maxsize=4)
def _production_gun_records(bore_mm: float = 20.0):
    """The existing production gun, selected intact at its mount boundary.

    This is intentionally extraction, not a second gun model. The square
    recoil tank/journal, original absorbers, barrel, washers and coolant
    circuit remain the records authored by ``turret_production``.
    """
    import turret_production as tp

    source = tp.balanced_station(bore_mm=float(bore_mm))
    keep = {node["identity"] for node in source.nodes
            if node.get("assembly") in PRODUCTION_GUN_ASSEMBLIES}
    nodes = tuple(deepcopy(node) for node in source.nodes
                  if node["identity"] in keep)
    edges = tuple(deepcopy(edge) for edge in source.edges
                  if edge["a"] in keep and edge["b"] in keep)
    volumes = tuple(deepcopy(volume) for volume in source.clear_volumes
                    if volume.get("owner") in keep)
    return nodes, edges, volumes


def _mount_production_gun(g, lower_platform, platform, dangling, gun,
                          bore_mm: float = 20.0):
    """Bolt the former production gun, unchanged, to this gun platform."""
    source_nodes, source_edges, source_volumes = \
        _production_gun_records(float(bore_mm))
    source_base = np.asarray(
        [node["reference_position"] for node in source_nodes
         if node["identity"].startswith("fine.base.")], float)
    target = np.mean(np.asarray([
        _node_at(g, platform["corners"][tag])
        for tag in ("aft.a", "fwd.a", "aft.b", "fwd.b")], float), axis=0)
    source_centre = source_base.mean(axis=0)
    base_shift = target - source_centre

    # The six fine.base feet stay registered to the arch-carried platform.
    # Everything above those feet is the gun and its mount. Move that whole
    # installed assembly until its actual mass centroid lies halfway between
    # the two-stage platform's full-forward and full-aft positions. This is
    # deliberately not a barrel-only visual offset: journal, dampers, cradle,
    # trunnion and their load paths all move with it.
    fixed_feet = {node["identity"] for node in source_nodes
                  if node["identity"].startswith("fine.base.")}
    moving_nodes = [node for node in source_nodes
                    if node["identity"] not in fixed_feet
                    and float(node.get("mass_kg", 0.0)) > 0.0]
    moving_mass = sum(float(node.get("mass_kg", 0.0))
                      for node in moving_nodes)
    if moving_mass <= 0.0:
        raise ValueError("production gun mount has no moving mass to balance")
    moving_cg = sum(
        float(node.get("mass_kg", 0.0))
        * np.asarray(node["reference_position"], float)
        for node in moving_nodes) / moving_mass
    travel_midpoint = target + 0.5 * (dangling.swept + gun.swept)
    balance_shift = np.zeros(3)
    balance_shift[2] = travel_midpoint[2] - (moving_cg + base_shift)[2]
    source_pitch = next(np.asarray(node["reference_position"], float)
                        for node in source_nodes
                        if node["identity"] == "turret.pitch")
    balance_shift[1] = (target[1] + GUN_TRUNNION_CLEARANCE_M
                        - (source_pitch + base_shift)[1])
    moved_ids = {node["identity"] for node in source_nodes
                 if node["identity"] not in fixed_feet}

    for original in source_nodes:
        node = deepcopy(original)
        shift = base_shift + (balance_shift
                              if node["identity"] in moved_ids else 0.0)
        node["reference_position"] = [
            float(v) for v in np.asarray(node["reference_position"], float)
            + shift]
        if node["identity"] in moved_ids:
            node["gun_platform_balance_shift_m"] = float(balance_shift[2])
        g.nodes.append(node)
    for original in source_edges:
        edge = deepcopy(original)
        # A member joining a retained foot to the translated upper mount is
        # fabricated in the new installed pose. Internal gun dimensions do
        # not change, but these support members need their honest new length
        # or the initial state would contain metres of false pre-strain.
        if ((edge["a"] in moved_ids) != (edge["b"] in moved_ids)):
            positions = {node["identity"]: np.asarray(
                node["reference_position"], float) for node in g.nodes}
            edge["rest_length"] = float(np.linalg.norm(
                positions[edge["b"]] - positions[edge["a"]]))
            edge["refabricated_for_balanced_mount"] = True
        g.edges.append(edge)
    for original in source_volumes:
        volume = deepcopy(original)
        volume_shift = (base_shift + balance_shift
                        if volume.get("owner") in moved_ids else base_shift)
        volume["min"] = tuple(float(v) for v in
                              np.asarray(volume["min"], float) + volume_shift)
        volume["max"] = tuple(float(v) for v in
                              np.asarray(volume["max"], float) + volume_shift)
        g.clear_volumes.append(volume)

    # Only the interface is new: the former gun's six fine-rig base pads
    # bolt to the nearest corners of the arch-carried platform, and the
    # trunnion standards rise from all four corners. Everything above those
    # bolts is the original production graph record.
    target_pos = {tag: np.asarray(_node_at(g, platform["corners"][tag]), float)
                  for tag in ("aft.a", "fwd.a", "aft.b", "fwd.b")}
    node_by_id = {node["identity"]: np.asarray(node["reference_position"], float)
                  for node in g.nodes}
    g.assembly = "production-gun-platform-interface"
    g.motion_group = "recoil"
    base_ids = [node["identity"] for node in source_nodes
                if node["identity"].startswith("fine.base.")]
    for i, base in enumerate(base_ids):
        nearest = min(target_pos, key=lambda tag: float(np.linalg.norm(
            node_by_id[base][[0, 2]] - target_pos[tag][[0, 2]])))
        g.edge(f"arch.gun_mount.fine_base.{i}",
               platform["corners"][nearest], base, "rigid-distance",
               radius=0.045, alloy="4340qt", palette="rollbar-silver",
               beam_solvable=True,
               load_path="former-fine-rig-base-bolted-to-arch-gun-platform")
    for tag in ("aft.a", "fwd.a", "aft.b", "fwd.b"):
        g.edge(f"arch.gun_mount.trunnion_standard.{tag}",
               platform["corners"][tag], "turret.pitch", "rigid-distance",
               radius=0.220, alloy="4340qt", palette="chassis-grey",
               beam_solvable=True, balanced_mount=True,
               provisional_overbuilt_test_support=True,
               load_path="overbuilt-platform-to-trunnion-test-standard")
    for side, tag in (("a", "aft.a"), ("b", "aft.b")):
        g.edge(f"arch.gun_mount.elevation_anchor.{side}",
               platform["corners"][tag], "turret.elevation_anchor",
               "rigid-distance", radius=0.180, alloy="4340qt",
               palette="chassis-grey", beam_solvable=True,
               balanced_mount=True, provisional_overbuilt_test_support=True,
               load_path="overbuilt-platform-to-elevation-anchor-test-brace")

    # Preserve the gun's declared downward case-ejection column. The upper
    # platform is a moment frame without its crossing diagonals; the lower
    # platform is open at the front where the chute passes. Those members
    # cannot coexist with the complete gun and pretending otherwise was the
    # exact interference the production gun's own clear-volume check found.
    ejection_crossings = {
        "dangling.deck.fwd.a-fwd.b",
        "gun.brace.aft.a-fwd.b",
        "gun.brace.fwd.a-aft.b",
    }
    g.edges[:] = [edge for edge in g.edges
                  if edge["identity"] not in ejection_crossings]

    # The selectable holes belong to the real platform links as metadata;
    # they are not extra bodies or a second set of structural members.
    for made in (lower_platform, platform):
        stage = made["stage"]
        for tag in ("aft.a", "fwd.a", "aft.b", "fwd.b"):
            link = next(edge for edge in g.edges
                        if edge["identity"] == f"{stage.identity}.link.{tag}")
            link["attachment_stations"] = [
                {"fraction": float(frac),
                 "moment_arm_m": round(stage.link_m * frac, 4)}
                for frac in STATIONS]

    document = g.as_document()
    gun.carries_kg = round(sum(float(node.get("mass_kg", 0.0))
                               for node in document["nodes"]
                               if node["identity"] in
                               {n["identity"] for n in source_nodes}), 1)
    dangling.carries_kg = round(gun.carries_kg + sum(
        float(node.get("mass_kg", 0.0)) for node in document["nodes"]
        if node["identity"].startswith("gun.")), 1)
    result = TwoStagePlatform(
        identity="arch-carried-production-gun", arch_span_m=2.0 * ARCH_X,
        dangling=dangling, gun=gun, preload_forward_n=40_000.0,
        slide_travel_m=0.844, slide_spring_n_per_m=SLIDE_SPRING_N_PER_M,
        slide_preload_n=SLIDE_PRELOAD_N,
        slide_piston_area_m2=SLIDE_PISTON_AREA_M2)
    platform["production_gun"] = True
    platform["bore_mm"] = float(bore_mm)
    platform["gun_balance_shift_m"] = float(balance_shift[2])
    platform["gun_vertical_shift_m"] = float(balance_shift[1])
    platform["installed_trunnion_clearance_m"] = GUN_TRUNNION_CLEARANCE_M
    platform["gun_travel_midpoint_m"] = tuple(float(v)
                                                for v in travel_midpoint)
    return g, result, lower_platform, platform


PALETTE = {
    "deck": "#3a4048",
    "arch": "#e0572c",          # the rigid thing
    "dangling": "#8a4fff",      # hangs off the arch pins
    "gun-platform": "#19c3b2",  # stands on the interior pins
    "gun": "#9aa7b5",
    "station": "#ffe066",
    "actuator": "#e8465a",
}

ROLE = {"deck.": "deck", "arch.": "arch",
        "dangling.": "dangling", "gun.": "gun-platform",
        "dangling.spring": "actuator", "dangling.ram": "actuator",
        "gun.spring": "actuator", "gun.ram": "actuator",
        "station.": "station", "weapon.": "gun"}


def build(fold: float = 0.0, g=None, deck_y: float | None = None,
          deck_nodes=None):
    """`fold` is 0 at rest and 1 fully folded.

    Pass `g` to emit into an EXISTING graph -- the real drum, with its
    own deck -- instead of onto a stand-in ring. `deck_nodes` are the
    plate nodes each arch foot should be bolted to; given them, the
    feet become a real load path into the plate instead of a boundary
    condition. That is the whole difference between a picture of the
    mechanism and the mechanism on the machine."""
    standalone = g is None
    if standalone:
        g = ProductionGraph(identity="arch-platforms")
    if deck_y is None:
        deck_y = DECK_Y

    # ---- THE DECK: the drum top the arches stand on ----
    g.motion_group, g.assembly = "traverse", "deck"
    ring = 16
    for i in range(ring) if standalone else ():
        a = 2.0 * math.pi * i / ring
        g.node(f"deck.rim.{i}",
               [DECK_R * math.cos(a), DECK_Y, DECK_R * math.sin(a)],
               "chassis-load-node", material="6061t6", fixed_to="world",
               mass_in_total=False, mass_kg=40.0,
               part_role="deck-rim", half_extent_m=(0.09, 0.05, 0.09))
    for i in range(ring) if standalone else ():
        g.edge(f"deck.rim_run.{i}", f"deck.rim.{i}",
               f"deck.rim.{(i + 1) % ring}", "rigid-distance", radius=0.052,
               palette="chassis-grey", load_path="the-deck-rim")

    # ---- THE TWO ARCHES: feet on the deck, straddling the bulk ----
    g.assembly = "arch"
    crown_y = deck_y + ARCH_RISE
    for side, sx in (("a", +1.0), ("b", -1.0)):
        for fz, z in (("f", +ARCH_FOOT_Z), ("r", -ARCH_FOOT_Z)):
            g.node(f"arch.foot.{side}.{fz}", [sx * ARCH_X, deck_y, z],
                   "load-bearing-structure", material="4340qt",
                   mass_in_total=False, mass_kg=140.0,
                   part_role="arch-foot", rigid=True,
                   fixed_to=("world" if standalone else None),
                   stands_on="the-rotating-deck",
                   half_extent_m=(0.13, 0.10, 0.13))
            # BOLT THE FOOT INTO THE PLATE. An arch foot that is only a
            # fixed point is a boundary condition; the deck never sees
            # the load and the drum is never asked to carry it. Bolted
            # to the plate nodes nearest it, the foot is a real path and
            # the beam solve has to answer for the deck as well.
            if deck_nodes:
                here = np.array([sx * ARCH_X, deck_y, z])
                near = sorted(deck_nodes, key=lambda k: float(np.linalg.norm(
                    np.asarray(_node_at(g, k)) - here)))[:PAD_BOLTS]
                for bi, other in enumerate(near):
                    g.edge(f"arch.footpad.{side}.{fz}.{bi}",
                           f"arch.foot.{side}.{fz}", other,
                           "rigid-distance", radius=0.044, alloy="4340qt",
                           palette="chassis-grey", beam_solvable=True,
                           part_role="arch-foot-bolt",
                           load_path="the-arch-standing-on-the-drum-plate")
            direction = 1.0 if z > 0.0 else -1.0
            haunch = f"arch.haunch.{side}.{fz}"
            lift_base = f"arch.lift_base.{side}.{fz}"
            g.node(haunch,
                   [sx * ARCH_X, deck_y + ARCH_HAUNCH_RISE_M,
                    direction * 0.98],
                   "load-bearing-structure", material="4340qt",
                   mass_in_total=False, mass_kg=72.0,
                   part_role="arch-haunch", rigid=True,
                   half_extent_m=(0.11, 0.10, 0.11))
            g.node(lift_base,
                   [sx * ARCH_X, deck_y + ARCH_LIFT_BASE_RISE_M,
                    direction * PIN_Z],
                   "load-bearing-structure", material="4340qt",
                   mass_in_total=False, mass_kg=88.0,
                   part_role="arch-hydraulic-lift-base", rigid=True,
                   half_extent_m=(0.13, 0.12, 0.13))
            g.node(f"arch.shoulder.{side}.{fz}",
                   [sx * ARCH_X, deck_y + ARCH_CROWN_SHOULDER_RISE_M,
                    direction * PIN_Z],
                   "load-bearing-structure", material="4340qt",
                   motion_group="arch-crown-lift",
                   mass_in_total=False, mass_kg=96.0,
                   part_role="arch-moving-crown-shoulder", rigid=True,
                   maximum_vertical_travel_m=ARCH_EXTREME_LIFT_M,
                   half_extent_m=(0.14, 0.13, 0.14))
            g.edge(f"arch.leg.lower.{side}.{fz}",
                   f"arch.foot.{side}.{fz}", haunch, "rigid-distance",
                   radius=0.110, alloy="4340qt", palette="rollbar-silver",
                   beam_solvable=True,
                   load_path="deep-lower-arch-leg-standing-on-the-deck")
            g.edge(f"arch.leg.haunch.{side}.{fz}", haunch, lift_base,
                   "rigid-distance", radius=0.105, alloy="4340qt",
                   palette="rollbar-silver", beam_solvable=True,
                   load_path="second-segment-forming-the-deep-arch-haunch")
            installed_riser_m = (ARCH_CROWN_SHOULDER_RISE_M
                                  - ARCH_LIFT_BASE_RISE_M)
            g.edge(f"arch.crown_lift.{side}.{fz}", lift_base,
                   f"arch.shoulder.{side}.{fz}",
                   "linear-hydraulic-actuator", radius=0.105,
                   alloy="4340qt", palette="actuator-yellow",
                   beam_solvable=True, part_role="arch-crown-lift-actuator",
                   bore_m=0.180, rod_m=0.120,
                   closed_length_m=installed_riser_m,
                   commanded_rest_length_m=installed_riser_m,
                   stroke_m=ARCH_EXTREME_LIFT_M,
                   travel_m=ARCH_EXTREME_LIFT_M,
                   actuator_extension_m=0.0,
                   actuator_state="normal-height-retracted",
                   lifts_for="extreme-vertical-gun-elevation",
                   load_path="hydraulic-column-raising-the-complete-arch-crown")
            g.edge(f"arch.crown_lock.{side}.{fz}", lift_base,
                   f"arch.shoulder.{side}.{fz}", "direct-drive-lockup",
                   radius=0.095, alloy="4340qt", palette="rollbar-silver",
                   beam_solvable=True, in_view=False,
                   part_role="arch-crown-positive-height-lock",
                   lock_engaged=True, unlocks_for="crown-height-change",
                   maximum_lock_travel_m=ARCH_EXTREME_LIFT_M,
                   load_path="positive-telescopic-lock-carrying-crown-load")
        apex = f"arch.crown.{side}.apex"
        g.node(apex, [sx * ARCH_X, crown_y, 0.0],
               "load-bearing-structure", material="4340qt",
               motion_group="arch-crown-lift", mass_in_total=False,
               mass_kg=110.0, part_role="arch-crown-apex", rigid=True,
               maximum_vertical_travel_m=ARCH_EXTREME_LIFT_M,
               half_extent_m=(0.15, 0.13, 0.15))
        for fz in ("f", "r"):
            g.edge(f"arch.crown.{side}.{fz}",
                   f"arch.shoulder.{side}.{fz}", apex,
                   "rigid-distance", radius=0.115, alloy="4340qt",
                   palette="rollbar-silver", beam_solvable=True,
                   load_path="deep-peaked-arch-crown-segment")
        g.edge(f"arch.haunch.tie.{side}",
               f"arch.haunch.{side}.f", f"arch.haunch.{side}.r",
               "rigid-distance", radius=0.085, alloy="4340qt",
               palette="rollbar-silver", beam_solvable=True,
               load_path="deep-arch-haunch-longitudinal-tie")
        # THE PINS. Two per arch, fore and aft, under the crown.
        for fz, z in (("f", +PIN_Z), ("r", -PIN_Z)):
            g.node(f"arch.pin.{side}.{fz}",
                   [sx * ARCH_X,
                    deck_y + ARCH_CROWN_SHOULDER_RISE_M - PIN_DROP, z],
                   "chassis-load-node", material="4340qt",
                   motion_group="arch-crown-lift",
                   mass_in_total=False, mass_kg=18.0,
                   part_role="arch-pin", rigid=True, pin_level=1,
                   shape="drum", drum_axis=(1.0, 0.0, 0.0),
                   drum_radius_m=0.055, drum_length_m=0.180,
                   half_extent_m=(0.090, 0.055, 0.055))
            g.edge(f"arch.pin_boss.{side}.{fz}", f"arch.pin.{side}.{fz}",
                   f"arch.shoulder.{side}.{fz}",
                   "rigid-distance", radius=0.046, alloy="4340qt",
                   beam_solvable=True, palette="rollbar-silver",
                   load_path="the-pin-hung-off-the-arch")
    g.edge("arch.crown.apex_tie", "arch.crown.a.apex",
           "arch.crown.b.apex", "rigid-distance", radius=0.095,
           alloy="4340qt", palette="rollbar-silver", beam_solvable=True,
           part_role="moving-crown-cross-tie",
           rises_with="arch-crown-lift",
           load_path="the-two-moving-arch-crowns-braced-as-one")
    # the two arches tied together so they are one rigid portal
    for fz in (("f", "r") if ARCH_TIE else ()):
        g.edge(f"arch.tie.{fz}", f"arch.shoulder.a.{fz}",
               f"arch.shoulder.b.{fz}", "rigid-distance", radius=0.060,
               alloy="4340qt", palette="rollbar-silver",
               beam_solvable=True, load_path="the-two-arches-tied-across")

    # ---- STAGE ONE: the dangling platform, off the four arch pins ----
    dangling = PlatformStage(
        identity="dangling", link_m=0.950, rises=-1.0,
        anchors=("arch.pin.a.r", "arch.pin.a.f",
                 "arch.pin.b.r", "arch.pin.b.f"),
        # Installed eight degrees forward of dead-hang. The packing rams'
        # declared preload holds this working side; gravity still supplies
        # the large-angle return instead of a fictitious giant spring.
        rest_angle_deg=8.0 - 50.0 * fold, travel_deg=50.0,
        inboard_m=0.0, carries_kg=0.0,
        pin_radius_m=0.055, pin_length_m=0.180,
        # GRAVITY IS THE RECUPERATOR, so the spring is not one. Both
        # stages rest at the bottom of their own travel and gravity
        # returns them to battery on its own. A spring stiff enough to
        # do that job as well would be a spring fighting the stroke it
        # exists to permit -- 240 kN/m over three quarters of a metre
        # is 177 kN of it, which swallowed the whole shot and left the
        # dampers doing nothing. These rates are slack take-up and
        # end-of-travel cushion, and nothing else.
        preload_n=9_000.0,
        spring_rate_n_per_m=26_000.0, damping_n_s_per_m=120_000.0)
    d = emit_platform_stage(g, dangling, motion_group="recoil",
                            assembly="dangling-platform")

    # ---- STAGE TWO: the gun platform, standing on interior pins ----
    gun = PlatformStage(
        identity="gun", link_m=1.050, rises=+1.0,
        anchors=d["anchors_for_next"],
        rest_angle_deg=68.0 - 52.0 * fold, travel_deg=52.0,
        inboard_m=0.260, carries_kg=0.0, preload_n=12_000.0,
        pin_radius_m=0.040, pin_length_m=0.120,
        link_radius_m=0.045,
        spring_rate_n_per_m=18_000.0, damping_n_s_per_m=90_000.0)
    u = emit_platform_stage(g, gun, motion_group="recoil",
                            assembly="gun-platform")
    emit_platform_deck_contacts(g, d, u)

    # The gun is not redrawn here. Mount the existing coherent production
    # gun -- square recoil tank/journal, original actuators and coolant-
    # jacketed barrel included -- on the platform just emitted above.
    return _mount_production_gun(g, d, u, dangling, gun, bore_mm=20.0)


def arch_height_document(document: dict, lift_m: float) -> dict:
    """Restage the real crown cylinders and everything they carry.

    This produces a physical pose for a reference/atlas solve.  It is not a
    render offset: the four lift-column and positive-lock lengths are changed
    to the installed height, while every crown, platform and gun body moves
    together. Runtime motion can later interpolate between validated atlas
    poses while the HCU owns unlocking, oil flow and relocking.
    """
    lift = float(lift_m)
    if not 0.0 <= lift <= ARCH_EXTREME_LIFT_M:
        raise ValueError(
            f"arch crown lift must be in [0, {ARCH_EXTREME_LIFT_M}] m")
    posed = deepcopy(document)
    moved = set()
    for node in posed["nodes"]:
        identity = str(node["identity"])
        assembly = str(node.get("assembly", ""))
        carries_crown = (
            node.get("motion_group") == "arch-crown-lift"
            or identity.startswith(("dangling.", "gun.", "fine.base."))
            or assembly in PRODUCTION_GUN_ASSEMBLIES)
        if not carries_crown:
            continue
        node["reference_position"][1] += lift
        node["arch_crown_lift_m"] = lift
        moved.add(identity)

    positions = {node["identity"]: np.asarray(node["reference_position"], float)
                 for node in posed["nodes"]}
    changed = []
    for edge in posed["edges"]:
        if (edge["a"] in moved) == (edge["b"] in moved):
            continue
        installed = float(np.linalg.norm(positions[edge["b"]]
                                         - positions[edge["a"]]))
        edge["rest_length"] = installed
        if edge.get("part_role") == "arch-crown-lift-actuator":
            edge["commanded_rest_length_m"] = installed
            edge["actuator_extension_m"] = lift
            edge["actuator_state"] = ("normal-height-retracted" if lift == 0.0
                                      else "high-angle-extended")
        elif edge.get("part_role") == "arch-crown-positive-height-lock":
            edge["lock_engaged"] = True
            edge["locked_height_m"] = lift
        changed.append(edge["identity"])
    expected = {f"arch.crown_{kind}.{side}.{end}"
                for kind in ("lift", "lock")
                for side in ("a", "b") for end in ("f", "r")}
    if set(changed) != expected:
        raise ValueError(
            "arch crown pose crosses undeclared members: "
            f"{sorted(set(changed) ^ expected)}")
    posed.pop("_graph_columns", None)
    posed["arch_crown_state"] = {
        "lift_m": lift,
        "fraction": lift / ARCH_EXTREME_LIFT_M,
        "actuators": sorted(identity for identity in changed
                            if ".crown_lift." in identity),
        "positive_locks": sorted(identity for identity in changed
                                 if ".crown_lock." in identity),
    }
    return posed

    # =================================================================
    #  THE MOUNT: high adjustment, trunnion, and the gun balanced on it
    # =================================================================
    # The gun is CENTRED on the platform and its centre of gravity is
    # put AT THE FRONT of it, where two things live:
    #
    #   the HIGH ADJUSTMENT -- a column standing off the forward cross
    #       beam. This is the coarse lay: it moves the whole gun up and
    #       down bodily and holds there, and because it is under the
    #       centre of gravity it is a column in pure compression rather
    #       than a jack fighting a moment.
    #
    #   the TRUNNION -- the pivot on top of that column, the FINE
    #       articulation. The gun turns on it for elevation.
    #
    # BALANCING ON THE PIVOT IS THE WHOLE POINT. With the centre of
    # gravity on the trunnion axis the elevating gear holds nothing at
    # any angle: it fights inertia and friction and nothing else, and
    # the mount does not care where the gun is pointed. Off balance it
    # would be a multi-tonne moment that reverses through the arc.
    g.assembly = "weapon"
    g.motion_group = "elevation"
    pos = {n["identity"]: np.asarray(n["reference_position"], float)
           for n in g.nodes}
    fwd_mid = (pos[u["corners"]["fwd.a"]] + pos[u["corners"]["fwd.b"]]) / 2.0
    aft_mid = (pos[u["corners"]["aft.a"]] + pos[u["corners"]["aft.b"]]) / 2.0

    # ---- the high adjustment, on the forward cross beam ----
    trunnion = fwd_mid + UP * HIGH_ADJUST_M
    g.node("mount.column", tuple(float(v) for v in
                                 fwd_mid + UP * (HIGH_ADJUST_M * 0.5)),
           "load-bearing-structure", material="4340qt",
           mass_in_total=False, mass_kg=180.0,
           part_role="high-adjustment-column", adjusts="coarse-lay",
           travel_m=HIGH_ADJUST_TRAVEL_M, half_extent_m=(0.14, 0.20, 0.14))
    g.edge("mount.column.foot", u["corners"]["fwd.a"], "mount.column",
           "rigid-distance", radius=0.070, alloy="4340qt",
           palette="chassis-grey", beam_solvable=True,
           load_path="the-high-adjustment-standing-on-the-forward-beam")
    g.edge("mount.column.foot.b", u["corners"]["fwd.b"], "mount.column",
           "rigid-distance", radius=0.070, alloy="4340qt",
           palette="chassis-grey", beam_solvable=True,
           load_path="the-high-adjustment-standing-on-the-forward-beam")
    g.node("mount.trunnion", tuple(float(v) for v in trunnion),
           "chassis-load-node", material="4340qt", mass_in_total=False,
           mass_kg=95.0, part_role="elevation-trunnion",
           articulates="fine", axis="across-the-machine",
           half_extent_m=(0.16, 0.08, 0.08))
    g.edge("mount.column.screw", "mount.column", "mount.trunnion",
           "linear-hydraulic-actuator", radius=0.056, alloy="4340qt",
           palette="actuator-yellow", part_role="high-adjustment",
           bore_m=0.110, rod_m=0.070, travel_m=HIGH_ADJUST_TRAVEL_M,
           holding_force_n=0.0, holds_no_moment=True,
           load_path="the-coarse-lay-under-the-centre-of-gravity")

    # ---- the gun: a real tapered tube, so its mass is its shape ----
    # 120 mm bore, 8.40 m, thick at the chamber and thin at the muzzle.
    # Nothing here is chosen except the two outside diameters; the
    # station masses are the frustum they imply.
    n_st = 12
    stations, tube_kg = [], 0.0
    for k in range(n_st):
        f0, f1 = k / n_st, (k + 1) / n_st
        z0 = BREECH_Z + BARREL_M * f0
        z1 = BREECH_Z + BARREL_M * f1
        od = BARREL_OD_BREECH + (BARREL_OD_MUZZLE - BARREL_OD_BREECH) \
            * (f0 + f1) / 2.0
        area = math.pi / 4.0 * (od * od - BORE_M * BORE_M)
        m = area * (z1 - z0) * 7850.0
        stations.append(((z0 + z1) / 2.0, m, od))
        tube_kg += m

    # ---- BALANCE: solve the counterweight, do not guess it ----
    # Moments about the trunnion, positive forward. The breech block and
    # the counterweight are the only things aft; everything else is the
    # tube, and the tube is what it is.
    tz = float(trunnion @ FORWARD)
    moment = sum(m * (z - tz) for z, m, _od in stations)
    moment += BREECH_KG * (BREECH_Z - 0.18 - tz)
    cw_z = tz - COUNTERWEIGHT_ARM_M
    counterweight_kg = -moment / (cw_z - tz)

    bore_y = float(trunnion @ UP)
    for k, (z, m, od) in enumerate(stations):
        g.node(f"weapon.tube.{k}", (0.0, bore_y, float(z)),
               "load-bearing-structure", material="4340qt",
               mass_in_total=False, mass_kg=round(m, 1),
               part_role="barrel-station", outer_diameter_m=round(od, 4),
               bore_m=BORE_M, rides_on="mount.trunnion",
               half_extent_m=(od / 2.0, od / 2.0, od / 2.0))
        if k:
            g.edge(f"weapon.tube_run.{k}", f"weapon.tube.{k-1}",
                   f"weapon.tube.{k}", "rigid-distance",
                   radius=round(od / 2.0, 4), alloy="4340qt",
                   palette="chassis-grey", beam_solvable=True,
                   load_path="the-8.40-m-outer-barrel")
    g.node("weapon.breech", (0.0, bore_y, BREECH_Z - 0.18),
           "load-bearing-structure", material="4340qt",
           mass_in_total=False, mass_kg=BREECH_KG, part_role="breech",
           half_extent_m=(0.26, 0.26, 0.26))
    g.edge("weapon.breech_face", "weapon.breech", "weapon.tube.0",
           "rigid-distance", radius=0.130, alloy="4340qt",
           palette="chassis-grey", beam_solvable=True,
           load_path="the-breech-onto-the-tube")
    g.node("weapon.counterweight", (0.0, bore_y, float(cw_z)),
           "load-bearing-structure", material="steel-plate",
           mass_in_total=False, mass_kg=round(counterweight_kg, 1),
           part_role="balance-counterweight",
           balances="the-gun-onto-the-trunnion-axis",
           arm_m=COUNTERWEIGHT_ARM_M, half_extent_m=(0.34, 0.26, 0.20))
    g.edge("weapon.tail", "weapon.breech", "weapon.counterweight",
           "rigid-distance", radius=0.090, alloy="4340qt",
           palette="chassis-grey", beam_solvable=True,
           load_path="the-tail-carrying-the-counterweight-aft")
    # ---- THE CRADLE: front on the trunnion, BACK ON THE AFT BEAM ----
    # The gun does not sit on the platform. It sits in a CRADLE, and
    # the cradle is carried at two points: the trunnion at the front,
    # on top of the high adjustment, and a REAR SUPPORT on the gun
    # platform's aft cross beam. Two points, because a tube eight
    # metres long held at one of them is a cantilever and an eight
    # metre cantilever is not a gun mount.
    #
    # THE REAR SUPPORT IS ALSO THE ELEVATING GEAR. The cradle pitches
    # about the front trunnion, so the only thing that has to move to
    # change elevation is its back end -- which is exactly what the
    # screw on the aft beam does.
    rear = aft_mid + UP * REAR_SUPPORT_M
    g.node("mount.rear_support", tuple(float(v) for v in rear),
           "chassis-load-node", material="4340qt", mass_in_total=False,
           mass_kg=110.0, part_role="rear-cradle-support",
           carries="the-back-of-the-gun", sets="elevation",
           half_extent_m=(0.18, 0.10, 0.12))
    for t in ("aft.a", "aft.b"):
        g.edge(f"mount.rear_screw.{t}", u["corners"][t], "mount.rear_support",
               "linear-hydraulic-actuator", radius=0.052, alloy="4340qt",
               palette="actuator-yellow", part_role="elevating-screw",
               bore_m=0.090, rod_m=0.056, travel_m=REAR_SUPPORT_TRAVEL_M,
               holding_force_n=0.0, holds_no_moment=True,
               load_path="the-back-of-the-gun-platform-holding-the-gun-up")
    # the cradle itself: one rigid body between the two supports
    g.node("mount.cradle", tuple(float(v) for v in (trunnion + rear) / 2.0),
           "load-bearing-structure", material="4340qt", mass_in_total=False,
           mass_kg=420.0, part_role="gun-cradle",
           half_extent_m=(0.22, 0.16, 0.70))
    for end in ("mount.trunnion", "mount.rear_support"):
        g.edge(f"mount.cradle.{end.split('.')[-1]}", "mount.cradle", end,
               "rigid-distance", radius=0.086, alloy="4340qt",
               palette="rollbar-silver", beam_solvable=True,
               load_path="the-cradle-spanning-its-two-supports")
    g.edge("mount.trunnion_pivot", "mount.trunnion", "mount.cradle",
           "pinned-trunnion-mount", radius=0.090, alloy="4340qt",
           palette="rollbar-silver", part_role="elevation-pivot",
           free_axis="pitch", balanced=True,
           load_path="the-cradle-turning-on-the-gun's-own-centre-of-gravity")

    # ---- THE GUN'S OWN LINEAR TRAVEL, inside the cradle -------------
    # THE STAGE THAT MOVES FIRST. Three mechanisms are now in series --
    # this slide, then the gun platform, then the dangling platform --
    # and in a series the SOFTEST one moves first. The slide is a
    # straight run with no gravity term and a 1:1 ratio, so it takes the
    # first milliseconds of the shot, which is when the force is largest
    # and when neither parallelogram has any leverage yet: both sets of
    # links are near the ends of their arcs, where an arc barely moves
    # for the angle it turns.
    #
    # It is also the EXISTING recoil mechanism. Nothing new -- the same
    # travel, the same absorber, the same stops. They work between the
    # gun and the cradle instead of between the gun and the structure.
    #
    #     THE RAIL MUST LIE ALONG THE BORE, and this is not cosmetic.
    #     `single-axis-slider` releases the MEMBER'S OWN AXIAL freedom
    #     -- that is what the constraint means, and `free_axis="bore"`
    #     is a label on the edge that no solver reads. Built as a short
    #     strut from the gun down to a rail beneath it, the released
    #     axis is VERTICAL: the gun is then free to fall and rigid
    #     along the bore, which is the exact opposite of a recoil
    #     slide, and the whole shot goes into the structure. Both slide
    #     members here run fore and aft on the bore line, so the
    #     freedom released is the one the gun recoils along.
    # TWO PHYSICAL WAYS AND FOUR SHOES.  The earlier graph drew two abstract
    # slider edges through the bore centre.  They released the right degree
    # of freedom, but there was no carriage one could point at.  These rails
    # sit below and to either side of the tube; rigid shoe brackets attach
    # them to the gun, and the cross-bars are the equalizer that prevents one
    # absorber from twisting the breech around a single guide.
    slide_axis = tuple(float(v) for v in FORWARD)
    rail_x, rail_y = 0.22, bore_y - 0.22
    moving_z = {"aft": BREECH_Z - 0.18, "fwd": 1.75}
    fixed_z = {tag: z - RAIL_STANDOFF_M for tag, z in moving_z.items()}
    slide_edges = []
    for side, x in (("a", -rail_x), ("b", rail_x)):
        for tag, holds in (("aft", "weapon.breech"),
                           ("fwd", "weapon.tube.2")):
            rail = f"mount.rail.{side}.{tag}"
            shoe = f"weapon.slide.shoe.{side}.{tag}"
            g.node(rail, (x, rail_y, fixed_z[tag]),
                   "load-bearing-structure", material="4340qt",
                   mass_in_total=False, mass_kg=42.0,
                   part_role="recoil-rail-bearing",
                   half_extent_m=(0.075, 0.060, 0.10))
            g.node(shoe, (x, rail_y, moving_z[tag]),
                   "load-bearing-structure", material="4340qt",
                   mass_in_total=False, mass_kg=18.0,
                   part_role="recoil-slide-shoe", recoils=True,
                   half_extent_m=(0.085, 0.070, 0.12))
            g.edge(f"mount.rail_seat.{side}.{tag}", rail, "mount.cradle",
                   "rigid-distance", radius=0.036, alloy="4340qt",
                   palette="chassis-grey", beam_solvable=True,
                   load_path="the-cradle-carrying-one-recoil-way")
            guide = f"weapon.slide.{side}.{tag}"
            g.edge(guide, rail, shoe, "single-axis-slider",
                   radius=0.046, alloy="4340qt", palette="chassis-grey",
                   part_role="recoil-slide-guide", free_axis="bore",
                   slide_axis=slide_axis, travel_m=SLIDE_TRAVEL_M,
                   load_path="one-shoe-running-on-one-physical-way")
            slide_edges.append(guide)
            g.edge(f"weapon.slide_bracket.{side}.{tag}", shoe, holds,
                   "rigid-distance", radius=0.034, alloy="4340qt",
                   palette="rollbar-silver", beam_solvable=True,
                   load_path="the-slide-shoe-bolted-to-the-gun")
        g.edge(f"mount.rail_run.{side}", f"mount.rail.{side}.aft",
               f"mount.rail.{side}.fwd", "rigid-distance", radius=0.048,
               alloy="4340qt", palette="chassis-grey", beam_solvable=True,
               load_path="one-continuous-recoil-way")
    for tag in ("aft", "fwd"):
        g.edge(f"weapon.slide_crosshead.{tag}",
               f"weapon.slide.shoe.a.{tag}",
               f"weapon.slide.shoe.b.{tag}", "rigid-distance",
               radius=0.042, alloy="4340qt", palette="rollbar-silver",
               beam_solvable=True,
               load_path="the-crosshead-sharing-load-between-both-ways")
    # THE PRODUCTION GUN'S TRIPLE-REDUNDANT RECOIL SYSTEM.  These are
    # parallel force paths across one slide, not three sequential stages:
    # the pneumatic recuperator reacts immediately and returns to battery,
    # the passive oil circuit dissipates velocity-squared energy, and the
    # MR circuit supplies controllable yield force even near zero velocity.
    # The declaration mirrors turret_production's real gun absorber.
    absorber_ids = ["weapon.absorber.pneumatic", "weapon.absorber.oil",
                    "weapon.absorber.magnetorheological"]
    # Three separate bores in one compact pack beneath the breech.  Each
    # end is one STRAIGHT CROSSHEAD spanning the two ways.  The former
    # triangular manifold had a third diagonal attachment through the
    # centre; it balanced algebraically but was physically asymmetric and
    # impossible to read in the mesh.  Here the outer ends attach to the
    # left and right ways symmetrically, and the centre MR cylinder loads
    # the crosshead between them.
    names = ("pneumatic", "oil", "magnetorheological")
    offsets = {"pneumatic": (-0.14, rail_y - 0.16),
               "magnetorheological": (0.0, rail_y - 0.16),
               "oil": (0.14, rail_y - 0.16)}
    reaction_z = moving_z["aft"] - SLIDE_TRAVEL_M - 0.08
    for name in names:
        x, y = offsets[name]
        g.node(f"mount.absorber.reaction.{name}", (x, y, reaction_z),
               "load-bearing-structure", material="4340qt",
               mass_in_total=False, mass_kg=9.0,
               part_role="recoil-absorber-fixed-clevis", element=name,
               half_extent_m=(0.060, 0.060, 0.070))
        g.node(f"weapon.absorber.head.{name}", (x, y, moving_z["aft"]),
               "load-bearing-structure", material="4340qt",
               mass_in_total=False, mass_kg=9.0, recoils=True,
               part_role="recoil-absorber-moving-clevis", element=name,
               half_extent_m=(0.065, 0.065, 0.075))
    crosshead_order = ("pneumatic", "magnetorheological", "oil")
    for side, (left, right) in enumerate(zip(crosshead_order,
                                              crosshead_order[1:])):
        g.edge(f"mount.absorber.reaction_ring.{side}",
               f"mount.absorber.reaction.{left}",
               f"mount.absorber.reaction.{right}", "rigid-distance",
               radius=0.026, alloy="4340qt", palette="rollbar-silver",
               load_path="the-straight-fixed-crosshead-between-both-ways")
        g.edge(f"weapon.absorber.moving_ring.{side}",
               f"weapon.absorber.head.{left}",
               f"weapon.absorber.head.{right}", "rigid-distance",
               radius=0.026, alloy="4340qt", palette="rollbar-silver",
               load_path="the-straight-moving-crosshead-between-both-shoes")
    for name, anchor in (("pneumatic", "mount.rail.a.aft"),
                         ("oil", "mount.rail.b.aft")):
        g.edge(f"mount.absorber.anchor.{name}",
               f"mount.absorber.reaction.{name}",
               anchor, "rigid-distance",
               radius=0.030, alloy="4340qt", palette="rollbar-silver",
               load_path="the-absorber-reaction-ring-into-the-two-ways")
    for name, anchor in (("pneumatic", "weapon.slide.shoe.a.aft"),
                         ("oil", "weapon.slide.shoe.b.aft")):
        g.edge(f"weapon.absorber.equalizer.{name}",
               f"weapon.absorber.head.{name}",
               anchor, "rigid-distance",
               radius=0.030, alloy="4340qt", palette="rollbar-silver",
               load_path="the-absorber-moving-ring-into-the-slide-crosshead")

    common = dict(radius=0.052, alloy="4340qt", palette="actuator-yellow",
                  part_role="recoil-absorber-element",
                  stroke_m=SLIDE_TRAVEL_M,
                  piston_area_m2=SLIDE_PISTON_AREA_M2,
                  parallel_with=absorber_ids,
                  alone_takes_full_charge=True)
    g.edge(absorber_ids[0], "mount.absorber.reaction.pneumatic",
           "weapon.absorber.head.pneumatic",
           "spring-damper", **common,
           force_components=("pneumatic-recuperator",),
           spring_rate_n_per_m=SLIDE_SPRING_N_PER_M,
           spring_preload_n=SLIDE_PRELOAD_N,
           load_path="passive-pneumatic-recuperator-on-the-recoil-slide")
    g.edge(absorber_ids[1], "mount.absorber.reaction.oil",
           "weapon.absorber.head.oil",
           "oleo-recoil-slide", **common,
           force_components=("oil-orifice",),
           orifice_area_m2=SLIDE_PISTON_AREA_M2 * 0.06,
           charge_pressure_pa=1.0, gas_volume_m3=1.0,
           load_path="passive-oil-brake-on-the-recoil-slide")
    g.edge(absorber_ids[2], "mount.absorber.reaction.magnetorheological",
           "weapon.absorber.head.magnetorheological",
           "linear-hydraulic-actuator", **common,
           kind="magnetorheological", force_components=("mr-yield",),
           gap_m=0.0012, active_length_m=0.060,
           yield_min_pa=1.0e3, yield_max_pa=6.0e4,
           plastic_viscosity_pa_s=0.28, coil_watts_max=22.0,
           current_frac=1.0,
           load_path="controllable-mr-brake-on-the-recoil-slide")

    # ---- THE ATTACHMENT STATIONS, a row of holes on every link ----
    # A hole is not another body and the two halves of the link are not
    # extra members.  The former representation added a mass node at every
    # hole and two rigid boss edges back to the link ends.  Besides drawing
    # dozens of two-part elbows whenever a marker missed an articulation
    # update, those duplicate members falsely stiffened the mechanism.
    # Keep the selectable stations on the one physical link as metadata.
    for made in (d, u):
        st = made["stage"]
        for tag in ("aft.a", "fwd.a", "aft.b", "fwd.b"):
            identity = f"{st.identity}.link.{tag}"
            link = next(edge for edge in g.edges
                        if edge["identity"] == identity)
            link["attachment_stations"] = [
                {"fraction": float(frac),
                 "moment_arm_m": round(st.link_m * frac, 4)}
                for frac in STATIONS
            ]

    # ---- WHAT EACH STAGE ACTUALLY CARRIES, read off the graph ----
    # `carries_kg` decides how much of the shot gravity takes back, so
    # it must be the mass that is really up there rather than a number
    # typed in beside the stage. The gun platform carries the weapon
    # and the mount; the dangling platform carries all of that plus the
    # gun platform itself.
    doc = g.as_document()
    above = {"gun": ("weapon.", "mount.", "gun."),
             "dangling": ("weapon.", "mount.", "gun.", "dangling.")}
    for stage, prefixes in ((gun, above["gun"]),
                            (dangling, above["dangling"])):
        stage.carries_kg = round(sum(
            float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
            if x["identity"].startswith(prefixes)), 1)

    plat = TwoStagePlatform(identity="arch-carried-gun-platform",
                            arch_span_m=2.0 * ARCH_X, dangling=dangling,
                            gun=gun, preload_forward_n=40_000.0,
                            slide_travel_m=SLIDE_TRAVEL_M,
                            slide_spring_n_per_m=SLIDE_SPRING_N_PER_M,
                            slide_preload_n=SLIDE_PRELOAD_N,
                            slide_piston_area_m2=SLIDE_PISTON_AREA_M2)
    return g, plat, d, u


# =====================================================================
#  THE DRAWING. A side elevation you can actually read.
# =====================================================================
def elevation(plat, path: str = "sled_elevation.png"):
    """One side of the machine, at rest and folded, labelled.

    Everything drawn is computed from the two stages' own declared
    geometry -- `PlatformStage.arm()` is the same function the graph
    was built with, so the picture cannot disagree with the parts.
    The barrel is BROKEN rather than drawn out: 8.40 m of tube beside
    a 2.5 m mechanism means either the tube is legible or the
    mechanism is, and the mechanism is the thing being shown."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1700, 1060
    img = Image.new("RGB", (W, H), (22, 24, 28))
    dr = ImageDraw.Draw(img)
    try:
        fn = ImageFont.truetype("arial.ttf", 18)
        fb = ImageFont.truetype("arialbd.ttf", 23)
        fs = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        fn = fb = fs = ImageFont.load_default()

    S, OX, OY = 214.0, 980.0, 905.0        # m -> px
    BREAK_Z = 2.55                         # where the barrel is cut

    def P(z, y):
        """bore-forward is +z and draws LEFT: the muzzle points left."""
        return (OX - z * S, OY - (y - DECK_Y) * S)

    def line(a, b, colour, w=5):
        dr.line([a, b], fill=colour, width=w)

    def dot(q, colour, r=8):
        dr.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=colour,
                   outline=(20, 20, 24), width=2)

    def tag(xy, lines, colour, anchor, head=None):
        """A label in clear space with a leader back to the part."""
        tx, ty = xy
        if head:
            dr.text((tx, ty), head, font=fb, fill=colour)
            ty += 30
        for t in lines:
            dr.text((tx, ty), t, font=fs, fill=colour)
            ty += 20
        dr.line([(tx - 12, xy[1] + 12), anchor], fill=colour, width=2)

    crown = DECK_Y + ARCH_RISE
    d, u = plat.dangling, plat.gun

    def geometry(fold):
        da = d.rest_angle_deg - d.travel_deg * fold
        ua = u.rest_angle_deg - u.travel_deg * fold
        pins = {t: np.array([0.0, crown - PIN_DROP,
                             PIN_Z * (1 if t == "fwd" else -1)])
                for t in ("fwd", "aft")}
        dcorn = {t: pins[t] + d.arm(da) for t in pins}
        ucorn = {t: dcorn[t] + u.arm(ua) for t in dcorn}
        return pins, dcorn, ucorn

    # ---------- the deck ----------
    line(P(DECK_R, DECK_Y), P(-DECK_R, DECK_Y), (74, 82, 95), 10)
    line(P(DECK_R, DECK_Y - 0.26), P(-DECK_R, DECK_Y - 0.26), (58, 64, 74), 8)
    for z in (DECK_R, -DECK_R):
        line(P(z, DECK_Y), P(z, DECK_Y - 0.26), (58, 64, 74), 8)
    dr.text(P(DECK_R - 0.02, DECK_Y - 0.44),
            "ROTATING DECK -- the drum top, 3.18 m across",
            font=fn, fill=(140, 150, 165))

    # ---------- the folded ghost ----------
    gp, gd, gu = geometry(1.0)
    G = (72, 79, 94)
    for t in ("fwd", "aft"):
        line(P(gp[t][2], gp[t][1]), P(gd[t][2], gd[t][1]), G, 4)
        line(P(gd[t][2], gd[t][1]), P(gu[t][2], gu[t][1]), G, 4)
    line(P(gd["fwd"][2], gd["fwd"][1]), P(gd["aft"][2], gd["aft"][1]), G, 7)
    line(P(gu["fwd"][2], gu["fwd"][1]), P(gu["aft"][2], gu["aft"][1]), G, 9)

    # ---------- the arch ----------
    A = (232, 96, 52)
    shz = ARCH_FOOT_Z * ARCH_SHOULDER_FRAC
    line(P(ARCH_FOOT_Z, DECK_Y), P(shz, crown), A, 11)
    line(P(-ARCH_FOOT_Z, DECK_Y), P(-shz, crown), A, 11)
    line(P(shz, crown), P(-shz, crown), A, 11)

    pins, dcorn, ucorn = geometry(0.0)

    # ---------- stage one: the dangling platform ----------
    D = (155, 102, 255)
    for t in ("fwd", "aft"):
        line(P(pins[t][2], pins[t][1]), P(dcorn[t][2], dcorn[t][1]), D, 10)
    line(P(dcorn["fwd"][2], dcorn["fwd"][1]),
         P(dcorn["aft"][2], dcorn["aft"][1]), D, 14)

    # ---------- stage two: the gun platform ----------
    U = (25, 210, 190)
    for t in ("fwd", "aft"):
        line(P(dcorn[t][2], dcorn[t][1]), P(ucorn[t][2], ucorn[t][1]), U, 10)
    line(P(ucorn["fwd"][2], ucorn["fwd"][1]),
         P(ucorn["aft"][2], ucorn["aft"][1]), U, 16)

    # ---------- the mount and the gun, balanced on the trunnion -----
    col_top = (ucorn["fwd"][2], ucorn["fwd"][1] + HIGH_ADJUST_M)
    line(P(ucorn["fwd"][2], ucorn["fwd"][1]), P(*col_top), (232, 70, 90), 16)
    gy = col_top[1]
    tz = col_top[0]
    cwz = tz - COUNTERWEIGHT_ARM_M
    # the tail aft to the counterweight, then the tube forward
    line(P(cwz, gy), P(BREAK_Z, gy), (154, 167, 181), 11)
    dr.rectangle([P(cwz + 0.17, gy + 0.15), P(cwz - 0.17, gy - 0.15)],
                 fill=(96, 106, 120), outline=(150, 162, 176), width=2)
    dr.polygon([P(BREECH_Z - 0.40, gy - 0.21), P(BREECH_Z + 0.24, gy - 0.16),
                P(BREECH_Z + 0.24, gy + 0.16), P(BREECH_Z - 0.40, gy + 0.21)],
               fill=(118, 130, 144))
    dr.text(P(cwz + 0.20, gy + 0.42), "trim weight", font=fs,
            fill=(150, 160, 175))
    dr.text(P(BREECH_Z + 0.10, gy + 0.42), "breech", font=fs,
            fill=(150, 160, 175))
    bx, by = P(BREAK_Z, gy)
    for k in (-1, 1):
        dr.line([(bx + k * 9 - 13, by - 22), (bx + k * 9 + 13, by + 22)],
                fill=(22, 24, 28), width=7)
        dr.line([(bx + k * 9 - 13, by - 22), (bx + k * 9 + 13, by + 22)],
                fill=(154, 167, 181), width=2)
    dr.text((bx + 26, by - 46), "8.40 m OUTER BARREL, 120 mm  -->",
            font=fn, fill=(172, 184, 198))
    dot(P(tz, gy), (255, 120, 90), 11)

    # ---------- the ARC each corner sweeps ----------
    # Drawn as the arc it actually is, dotted, and in a colour that is
    # not a member's. An earlier version drew these as straight chords
    # in the stages' own colours, which made them read as two extra
    # links -- and a parallelogram has four, so a fifth line in the
    # same colour is a lie about the mechanism.
    def sweep_path(which, corner_at):
        pts = []
        for k in range(41):
            _p, dc, uc = geometry(k / 40.0)
            q = (dc if which == "dangling" else uc)[corner_at]
            pts.append(P(q[2], q[1]))
        return pts

    for which, colour in (("dangling", (120, 96, 170)),
                          ("gun", (70, 150, 145))):
        for t in ("fwd", "aft"):
            pts = sweep_path(which, t)
            for k in range(0, len(pts) - 1, 2):
                dr.line([pts[k], pts[k + 1]], fill=colour, width=3)
    dr.text((1250, 745), "dotted = the arc each corner sweeps",
            font=fs, fill=(120, 130, 148))

    # ---------- the pins, drawn last so nothing covers them ----------
    for t in ("fwd", "aft"):
        dot(P(pins[t][2], pins[t][1]), (255, 224, 102))
        dot(P(dcorn[t][2], dcorn[t][1]), (255, 224, 102))
        dot(P(ucorn[t][2], ucorn[t][1]), (255, 224, 102))

    # ---------- labels, all in clear space, all with leaders ----------
    tag((1245, 250),
        ["two of them, one each side.",
         "Both feet on the deck,",
         "straddling the bulk of it."],
        A, P(shz - 0.30, crown), head="ARCH -- the only rigid member")
    tag((1250, 400), ["two per arch, fore and aft"],
        (255, 224, 102), P(pins["aft"][2], pins["aft"][1]),
        head="ARCH PINS")
    tag((1250, 530),
        ["a rectangle on four equal links,",
         f"hanging straight down.  link {d.link_m * 1000:.0f} mm.",
         "Its links take the pin OUTBOARD."],
        D, P(dcorn["aft"][2], dcorn["aft"][1]), head="DANGLING PLATFORM")
    tag((60, 560),
        ["a rectangle on four equal links STANDING UP",
         f"off the interior pins.  link {u.link_m * 1000:.0f} mm, laid",
         f"{u.rest_angle_deg:.0f} deg forward at rest -- so it lies forward",
         "and low, ON TOP of the dangling one.",
         "The gun is mounted to it."],
        U, P(ucorn["fwd"][2], ucorn["fwd"][1]), head="GUN PLATFORM")
    tag((60, 800),
        ["carried by the dangling platform,",
         f"{u.inboard_m * 1000:.0f} mm INBOARD of its own links, so the",
         "two stages fold through each other",
         "instead of colliding."],
        (255, 224, 102), P(dcorn["fwd"][2], dcorn["fwd"][1]),
        head="INTERIOR PINS")
    tag((60, 300),
        ["the COARSE lay: a column off the forward cross beam.",
         "It sits UNDER the centre of gravity, so it is a column",
         f"in compression, not a jack fighting a moment."
         f"  {HIGH_ADJUST_TRAVEL_M * 1000:.0f} mm travel."],
        (232, 70, 90), P(ucorn["fwd"][2], ucorn["fwd"][1] + HIGH_ADJUST_M
                         * 0.5), head="HIGH ADJUSTMENT")
    tag((640, 240),
        ["the FINE articulation. The gun's centre of gravity is put",
         "ON this axis by the counterweight aft, so the elevating",
         "gear holds nothing at any angle -- only inertia and friction."],
        (255, 120, 90), P(tz, gy), head="TRUNNION")
    tag((1250, 800),
        ["both stages swept fully aft.",
         "Every corner travels back AND up."],
        (140, 148, 166), P(gu["aft"][2], gu["aft"][1]),
        head="FULLY FOLDED")

    # ---------- the numbers ----------
    tx, ty = 38, 34
    rows = [("ARCH-CARRIED TWO-STAGE GUN PLATFORM", (235, 238, 244), fb),
            ("", None, fn),
            (f"stage 1  dangling      link {d.link_m * 1000:4.0f} mm"
             f"   sweep {d.travel_deg:2.0f} deg"
             f"   ->  back {d.reach_m * 1000:4.0f} mm"
             f"   up {d.rise_m * 1000:4.0f} mm", D, fn),
            (f"stage 2  gun platform  link {u.link_m * 1000:4.0f} mm"
             f"   sweep {u.travel_deg:2.0f} deg"
             f"   ->  back {u.reach_m * 1000:4.0f} mm"
             f"   up {u.rise_m * 1000:4.0f} mm", U, fn),
            ("IN SERIES -- the gun platform's pins are carried by the"
             " dangling one, so they add:", (200, 206, 216), fn),
            (f"         back {plat.total_reach_m * 1000:.0f} mm"
             f"   up {plat.total_rise_m * 1000:.0f} mm"
             f"   -- gravity stores {plat.gravity_j / 1000:.1f} kJ"
             f" over the full stroke", (255, 224, 102), fn)]
    for text, colour, font in rows:
        if colour is not None:
            dr.text((tx, ty), text, font=font, fill=colour)
        ty += 30
    img.save(path)
    return path


def station_table(document: dict, recoil_j: float, stroke_m: float) -> list:
    seen = {}
    link = next((edge for edge in document["edges"]
                 if edge["identity"] == "gun.link.fwd.a"), None)
    for station in (link or {}).get("attachment_stations", ()):
        seen[station["fraction"]] = station["moment_arm_m"]
    out = [f"  ATTACHMENT STATIONS on a gun-platform link "
           f"(absorbing {recoil_j / 1000:.0f} kJ)",
           "    hole   arm       stroke seen    mean force needed"]
    for f, arm in sorted(seen.items()):
        s = stroke_m * f
        out.append(f"    {f:4.2f}   {arm:5.3f} m   {s * 1000:7.0f} mm"
                   f"      {recoil_j / max(s, 1e-6) / 1000:8.0f} kN")
    return out



# =====================================================================
#  FIRING IT. Every number below comes out of an existing part.
# =====================================================================
def fire(g, plat, d, u, bores=(20.0, 120.0), rounds=3) -> list:
    """Put some rounds through it and see where the energy goes.

    NOTHING IS SIMULATED HERE THAT THE ENGINE ALREADY DOES:

        calibres.cannon         a bore is very nearly a whole cartridge
        calibres.recoil_of      projectile impulse + gas impulse, and
                                the gas is a third of it
        PlatformStage.damper    the existing AdaptiveRecoilDamper, whose
                                aperture is one algebraic inversion of
                                the energy balance per shot

    What this adds is only the BOOKKEEPING between them: the recoiling
    mass is read off the graph, the free-recoil energy follows from the
    impulse and that mass, gravity's share follows from the two stages'
    own `rise_m`, and what is left is what the sixteen bay dampers
    have to turn into heat.
    """
    import calibres
    doc = g.as_document()
    recoiling = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
                    if x["identity"].startswith(("weapon.", "mount.",
                                                 "gun.")))
    bays = [e for e in doc["edges"]
            if e.get("part_role") == "platform-damper"]
    out = [f"FIRING  --  {recoiling:.0f} kg recoiling on two stages,"
           f" {len(bays)} bay dampers",
           f"          stroke {plat.total_reach_m * 1000:.0f} mm back,"
           f" {plat.total_rise_m * 1000:.0f} mm up", ""]
    out += [f"  GRAVITY'S CAPACITY over the full stroke is"
            f" {plat.gravity_j / 1000:.1f} kJ. That is a CEILING, not a"
            f" share:",
            f"  a round too small to sweep the stages does not get all"
            f" of it, and one",
            f"  too large has the dampers make up the rest.", ""]
    for bore in bores:
        cal = calibres.cannon(bore)
        wrench = calibres.recoil_of(cal)
        imp = wrench.net_impulse_n_s
        v = imp / recoiling
        free_j = 0.5 * recoiling * v * v
        lift_j = min(plat.gravity_j, free_j)
        damp_j = max(free_j - lift_j, 0.0)
        per_bay = damp_j / max(len(bays), 1)
        # how far up the stroke this round actually gets, if the bays
        # are set to take what gravity does not
        used = min(1.0, free_j / max(plat.gravity_j, 1e-9)) \
            if damp_j <= 0.0 else 1.0
        stroke_m = plat.total_reach_m * used
        mean_n = damp_j / max(stroke_m, 1e-6)
        bare_n = wrench.peak_force_n()
        out_ms = 2.0 * stroke_m / max(v, 1e-9) * 1000.0
        # PER BAY, NOT PER STAGE. Four bays carry a stage in parallel
        # at one velocity, so each takes a quarter of the impulse and a
        # quarter of the mass -- same recoil velocity, a quarter of the
        # energy. Sizing one damper against the whole shot, which is
        # what a single call with the full mass does, asks each of four
        # dampers to do the job of four.
        share = 1.0 / BAYS_PER_STAGE
        sized = [("slide", plat.slide_damper().orifice_for(imp, recoiling))]
        sized += [(st.identity, st.damper().orifice_for(imp * share,
                                                        recoiling * share))
                  for st in (plat.gun, plat.dangling)]
        out += [f"  {cal.name}   {cal.mass_kg:.1f} kg at"
                f" {cal.muzzle_m_s:.0f} m/s,"
                f" {cal.muzzle_energy_j / 1e6:.2f} MJ at the muzzle"]
        out += ["  " + ln for ln in wrench.describe()]
        out += [f"    free recoil     {v:7.3f} m/s  ->"
                f" {free_j / 1000:8.1f} kJ into the mount"
                f"   ({free_j / max(cal.muzzle_energy_j, 1) * 100:.1f} %"
                f" of muzzle energy)"]
        if damp_j <= 0.0:
            out += [f"    gravity alone   takes all of it in"
                    f" {stroke_m * 1000:.0f} mm of the"
                    f" {plat.total_reach_m * 1000:.0f} mm available --"
                    f" the dampers do nothing",
                    f"                    and hand it straight back as"
                    f" return to battery"]
        else:
            out += [f"    gravity stores  {lift_j / 1000:8.1f} kJ"
                    f"  ({lift_j / free_j * 100:4.1f} %), returned as"
                    f" battery",
                    f"    dampers eat     {damp_j / 1000:8.1f} kJ"
                    f"  = {per_bay / 1000:6.1f} kJ in each of the"
                    f" {len(bays)} bays",
                    f"    mean force      {mean_n / 1000:8.0f} kN over the"
                    f" {stroke_m * 1000:.0f} mm stroke"
                    f"  -- {bare_n / 1000:.0f} kN with no recoil system",
                    f"    stroke time     {out_ms:8.0f} ms out, so"
                    f" the cycle is that plus return"]
        for ident, sz in sized:
            note = sz.get("note", "")
            if note:
                out.append(f"    {ident:9s} {note}")
            else:
                # the slide is ONE absorber, not four bays
                n_of = 1 if ident == "slide" else BAYS_PER_STAGE
                stage_n = sz["peak_force_n"] * n_of / 1000.0
                out.append(f"    {ident:9s} {'aperture' if n_of == 1 else 'each bay: aperture'}"
                           f" {sz['orifice_mm']:5.1f} mm dia, peak"
                           f" {sz['peak_force_n'] / 1000:6.0f} kN"
                           f"  ->  {stage_n:6.0f} kN"
                           + (f" across the stage's {n_of} bays"
                              if n_of > 1 else " total")
                           + ("   AT ITS LIMIT" if sz["at_limit"] else ""))
        # WHAT THE ARCH ACTUALLY SEES. An orifice's force goes with the
        # SQUARE of the velocity through it, and the velocity is highest
        # in the first millimetre, so the peak is several times the mean.
        # The mean is what sizes the stroke; the peak is what sizes the
        # structure, and quoting the mean at a structural engineer is
        # how mounts get broken.
        peaks = [sz.get("peak_force_n", 0.0)
                 * (1 if i == "slide" else BAYS_PER_STAGE)
                 for i, sz in sized]
        if max(peaks) > 0.0:
            out += [f"    INTO THE ARCH   {max(peaks) / 1000:8.0f} kN peak"
                    f"  --  {max(peaks) / max(mean_n, 1e-9):.1f} x the mean,"
                    f" because orifice force goes as v squared",
                    f"                    a metering pin that closes"
                    f" through the stroke is what flattens that"]
        oil_kg, cp = 40.0, 1900.0
        rise = damp_j * rounds / (oil_kg * cp)
        out += [f"    {rounds} rounds      {damp_j * rounds / 1e6:5.2f} MJ"
                f" into the dampers  ->  {rise:5.1f} K in {oil_kg:.0f} kg"
                f" of oil, uncooled", ""]
    return out



def mount_table(document: dict) -> list:
    """What the balance came out as.

    The trim weight is SOLVED, not chosen: the moment of everything
    else about the trunnion, divided by the arm it sits on."""
    n = {x["identity"]: x for x in document["nodes"]}
    gun = [x for x in document["nodes"]
           if x["identity"].startswith("weapon.")]
    m = sum(float(x["mass_kg"]) for x in gun)
    cg = sum(np.asarray(x["reference_position"], float) * x["mass_kg"]
             for x in gun) / m
    tz = float(np.asarray(n["mount.trunnion"]["reference_position"],
                          float)[2])
    tube = sum(float(x["mass_kg"]) for x in document["nodes"]
               if x.get("part_role") == "barrel-station")
    off = abs(float(cg[2]) - tz)
    return [
        "  THE MOUNT, at the front of the gun platform",
        f"    high adjustment  column {HIGH_ADJUST_M * 1000:.0f} mm at rest,"
        f" {HIGH_ADJUST_TRAVEL_M * 1000:.0f} mm of coarse lay,"
        f" in pure compression",
        "    trunnion         on top of it -- the fine articulation",
        f"    gun              {tube:.0f} kg of tube"
        f" ({BARREL_OD_BREECH * 1000:.0f} mm at the chamber to"
        f" {BARREL_OD_MUZZLE * 1000:.0f} mm at the muzzle)",
        f"                     + {BREECH_KG:.0f} kg breech"
        f" + {n['weapon.counterweight']['mass_kg']:.1f} kg trim weight at"
        f" {COUNTERWEIGHT_ARM_M:.2f} m aft   =   {m:.0f} kg",
        f"    balance          CG lands {off * 1000:.2f} mm off the"
        f" trunnion axis  ->  {m * 9.80665 * off:.0f} N.m for the"
        f" elevating gear to hold",
        f"                     one metre out of balance would be"
        f" {m * 9.80665 / 1000:.1f} kN.m, which is the whole reason"
        f" for the trim weight",
    ]



def _segment_gap(p0, p1, q0, q1) -> float:
    """Closest approach of two finite segments, centre line to centre
    line. Real members have radius, so the clearance is this minus both
    radii -- which is what the caller subtracts."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c = u @ u, u @ v, v @ v
    dd, e = u @ w, v @ w
    den = a * c - b * b
    if den < 1e-12:                      # parallel
        sc, tc = 0.0, (e / c if c > 1e-12 else 0.0)
    else:
        sc = (b * e - c * dd) / den
        tc = (a * e - b * dd) / den
    sc = min(max(sc, 0.0), 1.0)
    tc = min(max(tc, 0.0), 1.0)
    return float(np.linalg.norm((p0 + u * sc) - (q0 + v * tc)))


def clearance_table(steps: int = 41) -> list:
    """Does anything that moves hit anything that does not?

    THE ARCH IS RIGID AND EVERYTHING ELSE SWINGS, so the arch is the
    thing to be hit. This walks the fold, rebuilds the graph at each
    step -- the same `build()` the drawing and the solve use, so there
    is no second geometry to drift out of step -- and measures every
    moving member against every arch member, surface to surface.

    A mechanism that has not been swept through its own travel against
    its own structure is a mechanism that has not been checked."""
    worst = {}
    for k in range(steps):
        gg, _plat, _d, _u = build(fold=k / (steps - 1))
        doc = gg.as_document()
        pos = {x["identity"]: np.asarray(x["reference_position"], float)
               for x in doc["nodes"]}
        fixed, moving = [], []
        for e in doc["edges"]:
            if e["a"] not in pos or e["b"] not in pos:
                continue
            item = (e["identity"], pos[e["a"]], pos[e["b"]],
                    float(e.get("radius", 0.02)))
            if e["identity"].startswith("arch."):
                fixed.append(item)
            elif e["identity"].startswith(("gun.", "dangling.", "weapon.",
                                           "mount.")):
                moving.append(item)
        for mi, ma, mb, mr in moving:
            for fi, fa, fb, fr in fixed:
                gap = _segment_gap(ma, mb, fa, fb) - mr - fr
                key = (mi, fi)
                if key not in worst or gap < worst[key][0]:
                    worst[key] = (gap, k / (steps - 1))
    hits = sorted(((g, f, m, a) for (m, a), (g, f) in worst.items()),
                  key=lambda r: r[0])
    out = ["  SWEEP CLEARANCE -- everything that moves, against the arch"]
    bad = [h for h in hits if h[0] < 0.0]
    if not bad:
        out.append(f"    closest approach {hits[0][0] * 1000:.0f} mm"
                   f"  ({hits[0][2]} vs {hits[0][3]} at"
                   f" {hits[0][1] * 100:.0f} % of the fold)  --  CLEAR")
        return out
    out.append(f"    {len(bad)} INTERFERENCES. The travel as declared"
               f" drives parts through the arch:")
    for gap, at, mv, fx in bad[:8]:
        out.append(f"      {mv:34s} into {fx:22s}"
                   f" by {-gap * 1000:6.0f} mm at {at * 100:3.0f} % fold")
    if len(bad) > 8:
        out.append(f"      ... and {len(bad) - 8} more")
    return out


def _paint(document: dict):
    from engine_mesh import (Material, _rgb, DECLARED_MATERIALS, MATERIALS,
                             MATERIAL_INDEX)
    for i, (name, colour) in enumerate(PALETTE.items()):
        key = f"archplat::{name}"
        if key not in DECLARED_MATERIALS:
            m = Material(f"ap{i}", name, _rgb(colour), 1.0, 0.32, 0.22,
                         0.6, 45.0, 0.8)
            DECLARED_MATERIALS[key] = m
            MATERIALS.append(m)
            MATERIAL_INDEX[m.name] = len(MATERIALS) - 1

    def pick(ident):
        best = None
        for prefix, name in ROLE.items():
            if ident.startswith(prefix) and (best is None
                                             or len(prefix) > best[0]):
                best = (len(prefix), name)
        return f"archplat::{best[1] if best else 'deck'}"

    return dict(document,
                nodes=[dict(x, material=pick(x["identity"]))
                       for x in document["nodes"]],
                edges=[dict(e, material=pick(e["identity"]),
                            in_view=(not e.get("rigid", False)
                                     or e["identity"].startswith("arch.")))
                       for e in document["edges"]])


def _view(painted, W, H, live):
    import pygame
    pygame.display.init()
    pygame.display.set_mode((W, H) if live else (64, 64),
                            pygame.OPENGL | pygame.DOUBLEBUF
                            | (0 if live else pygame.HIDDEN))
    from engine_gl_view import EngineGLView
    view = EngineGLView(width=W, height=H, covers_off=True,
                        animation_divisions="dense", detail=0.8,
                        spring_style="cylinder")
    view.set_graph(painted)
    while view.bake_next():
        pass
    return view


def main(argv):
    g, plat, d, u = build()
    doc = g.as_document()
    pr = g.check()
    # the design case is the largest round, and its free recoil comes
    # out of `calibres`, not out of a number typed here
    import calibres
    recoiling = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
                    if x["identity"].startswith(("weapon.", "mount.",
                                                 "gun.")))
    design_j = (calibres.recoil_of(calibres.cannon(120.0)).net_impulse_n_s
                ** 2) / (2.0 * recoiling)
    print("\n".join(describe(plat, design_j)), flush=True)
    print()
    print("\n".join(station_table(doc, design_j, plat.total_reach_m)),
          flush=True)
    print(f"\n  check  {'PASS' if not pr else list(dict.fromkeys(pr))[:2]}")
    print(f"  graph  {len(doc['nodes'])} nodes, {len(doc['edges'])} edges",
          flush=True)
    print()
    print("\n".join(mount_table(doc)), flush=True)
    print("  wrote", elevation(plat), flush=True)
    print()
    print("\n".join(clearance_table()), flush=True)
    print()
    print("\n".join(fire(g, plat, d, u)), flush=True)

    if "--fold" in argv:
        import pygame
        from PIL import Image
        shots = []
        for k in range(16):
            gg, _p, _d, _u = build(fold=k / 15.0)
            view = _view(_paint(gg.as_document()), 1100, 800, False)
            view._angle_rad, view._elevation = 1.571, 0.12
            img = view.render(crank_angle_deg=0.0, spin=False, dt=0.0,
                              throttle_frac=0.0)
            shots.append(Image.fromarray(np.ascontiguousarray(img[:, :, :3])))
            pygame.display.quit()
        shots[0].save("sled_fold.png", save_all=True,
                      append_images=shots[1:], duration=110, loop=0)
        print("  wrote sled_fold.png", flush=True)
        return

    if "--render" in argv:
        import pygame
        view = _view(_paint(doc), 1400, 950, False)
        for az, el, name in ((1.571, 0.12, "sled_ref_side.png"),
                             (0.95, 0.35, "sled_ref_q.png")):
            view._angle_rad, view._elevation, view._zoom = az, el, 1.0
            img = view.render(crank_angle_deg=0.0, spin=False, dt=0.0,
                              throttle_frac=0.0)
            rgb = np.ascontiguousarray(img[:, :, :3])
            pygame.image.save(pygame.image.frombuffer(
                rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"), name)
            print("  wrote", name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
