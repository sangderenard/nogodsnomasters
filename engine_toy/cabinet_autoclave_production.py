"""A glass-faced cabinet autoclave: square, walk-in sized, multipurpose.

THE TUBE AUTOCLAVE IS SMALL BECAUSE A TUBE IS CHEAP. Hoop stress sizes a
round shell at a few millimetres, and that is why almost every pressure
vessel is round. The machine actually wanted here is a different thing:
a square cabinet with a glass front, big enough to take an assembled
engine or a composite lay-up on racks, with the full suite of pressure,
vacuum, heat and venting -- and with glove ports as one of the things
it can be supplied with. A square cabinet pays for its shape in plate,
and this module makes it pay honestly.

FLAT WALLS ARE SIZED BY BENDING, NOT HOOP. A flat plate under pressure
does not carry it in tension the way a shell does; it bends, and the
stress goes as the square of the span over the thickness. So the walls
are stiffened by ribs at a pitch, the plate spans between the ribs,
and its thickness is

    t = span . sqrt(beta . p / sigma_allow) + corrosion allowance

with beta about a half for a strip clamped along its long edges. At
three bar over a 400 mm rib pitch that is 18 mm of boiler plate; at six
bar it is 25 mm. The tube vessel at three bar is under 4 mm. That ratio
IS the design decision, and it is reported rather than hidden.

THE GLASS IS A PLATE TOO, and a worse one: laminated toughened glass
has a long-term allowable around 25 MPa against boiler plate's 100, so
a single pane the width of the door would be ninety millimetres thick.
Real pressure windows are mullioned into strips for exactly this
reason; here the door carries vertical mullions at the same pitch as
the wall ribs, and the pane thickness falls out of the strip span --
31 mm at three bar. Glove ports go through the middle pane when that
option is fitted, and the pane is not thinner for having holes in it.

THE CASE IS THE FRAME. Five plate walls welded to one another along
their edges are a unibody; nothing else holds the chamber's shape, and
the walk sees it that way because every plate-to-plate seam is a
shell-attachment-weld and the skid under the floor is welded on the
same way. The door is the one thing that is not welded: it is clamped
shut on dogs, which are bolted-flange-mounts, and hangs on a hinge that
is one when it is open.

WHAT IT CAN BE SUPPLIED WITH is declared in machine_options terms --
racks, an engine cradle, glove ports, a heater, a vacuum set, an air
set for pressure hold -- and racks and the cradle want the same volume,
so they exclude each other by geometry as the tube autoclave's boiler
and long chamber do.
"""
from __future__ import annotations

import math
import re

import numpy as np

from prism_bodies import Prism, emit_prism
from turret_production import ProductionGraph
from milspec import AngleSection, SquareTubeSection
from machine_options import Claim, Configuration, OptionalAssembly
from operating_states import Cycle, OperatingState, run_up

# --- the chamber, inside, in metres ------------------------------------
INSIDE_W_M = 1.20      # x
INSIDE_H_M = 1.50      # y
INSIDE_D_M = 1.20      # z; the door is the +z face
#: Rib pitch on the walls and mullion pitch on the door. One number,
#: because a plate spans between whatever stiffens it and the door is
#: a wall that opens.
STIFFENER_PITCH_M = 0.40
#: A strip clamped along its long edges: sigma = beta.p.(b/t)^2 with
#: beta = 0.5. Simply supported would be 0.75; a real ribbed wall is
#: between and this is the value ordinary vessel practice uses.
PLATE_BETA = 0.50
BOILER_PLATE_ALLOWABLE_PA = 100e6
JOINT_EFFICIENCY = 0.85
CORROSION_ALLOWANCE_M = 0.0015
STEEL_DENSITY_KG_M3 = 7850.0
#: Laminated, toughened. The long-term allowable for tempered glass is
#: about a quarter of its short-term strength, which is what a window
#: that holds pressure for hours has to be designed to.
GLASS_ALLOWABLE_PA = 25e6
GLASS_DENSITY_KG_M3 = 2500.0
#: The ribs and the door frame: commodity sections in mild steel.
RIB_SECTION = AngleSection(leg_m=0.060, thickness_m=0.006, material="a36",
                           designation="60x60x6 angle A36")
DOOR_SECTION = SquareTubeSection(side_m=0.080, wall_m=0.005, material="a36",
                                 designation="80x80x5 SHS A36")
SKID_SECTION = SquareTubeSection(side_m=0.080, wall_m=0.005, material="a36",
                                 designation="80x80x5 SHS A36")
SKID_HEIGHT_M = 0.15
#: Rack levels, and what each carries when loaded.
RACK_LEVELS_Y = (0.35, 0.70, 1.05, 1.40)
RACK_MASS_KG = 18.0
RACK_LOAD_KG = 30.0
#: The cradle, and the engine it is there for.
CRADLE_MASS_KG = 60.0
ENGINE_LOAD_KG = 250.0
HEATER_KW = 24.0
HEATER_MASS_KG = 25.0


def plate_thickness_m(pressure_pa: float, span_m: float,
                      allowable_pa: float, *, beta: float = PLATE_BETA,
                      efficiency: float = 1.0, allowance_m: float = 0.0) -> float:
    """A flat plate strip under pressure, by bending."""
    return span_m * math.sqrt(beta * abs(pressure_pa) / (allowable_pa * efficiency)) + allowance_m


def wall_thickness_m(pressure_pa: float, pitch_m: float = STIFFENER_PITCH_M) -> float:
    return plate_thickness_m(pressure_pa, pitch_m, BOILER_PLATE_ALLOWABLE_PA,
                             efficiency=JOINT_EFFICIENCY,
                             allowance_m=CORROSION_ALLOWANCE_M)


def glass_thickness_m(pressure_pa: float, pitch_m: float = STIFFENER_PITCH_M) -> float:
    return plate_thickness_m(pressure_pa, pitch_m, GLASS_ALLOWABLE_PA)


def _rib_mass(span_m: float, run_m: float, pitch_m: float = STIFFENER_PITCH_M) -> float:
    """Ribs across a wall at the pitch, each the wall's other dimension."""
    count = max(1, int(round(run_m / pitch_m)) - 1)
    return count * span_m * RIB_SECTION.mass_per_m_kg


# ---------------------------------------------------------------------
# what it can be supplied with
# ---------------------------------------------------------------------

def _rack_claim(i: int, y: float) -> Claim:
    # headroom above the shelf, and never into the ceiling plate: the
    # first draft claimed to 1.70 m in a 1.50 m chamber and the graph
    # reported the ceiling's own skin as an intruder
    return Claim(identity=f"cabinet.rack_{i}",
                 min=(-INSIDE_W_M / 2.0 + 0.05, y - 0.02, -INSIDE_D_M / 2.0 + 0.05),
                 max=(INSIDE_W_M / 2.0 - 0.05, min(y + 0.30, INSIDE_H_M - 0.02),
                      INSIDE_D_M / 2.0 - 0.05),
                 note=f"rack level {i} and the headroom above it")


RACK_SET = OptionalAssembly(
    identity="cabinet.rack_set", label="four-level rack set",
    provides=("racking",), requires=(),
    claims=tuple(_rack_claim(i, y) for i, y in enumerate(RACK_LEVELS_Y)),
    mass_kg=RACK_MASS_KG * len(RACK_LEVELS_Y), casing_ports=(),
    note="many small things: the whole interior in shelves")

ENGINE_CRADLE = OptionalAssembly(
    identity="cabinet.engine_cradle", label="engine cradle",
    provides=("cradle",), requires=(),
    # WANTS THE LOWER HALF OF THE CHAMBER, which is where the lower two
    # racks are: they exclude each other by volume, and the upper racks
    # do not, so a cradle with two racks above it is a configuration
    claims=(Claim(identity="cabinet.cradle_bay",
                  min=(-0.50, 0.0, -0.45), max=(0.50, 0.90, 0.45),
                  note="the cradle and an assembled engine on it"),),
    mass_kg=CRADLE_MASS_KG, casing_ports=(),
    note="one large thing: an assembled engine, baked or pressure-tested")

GLOVE_PORTS = OptionalAssembly(
    identity="cabinet.glove_ports", label="glove ports in the door",
    provides=("manipulation",), requires=(),
    # the reach of two arms through the middle pane, inside the chamber
    claims=(Claim(identity="cabinet.glove_reach",
                  min=(-0.45, 0.60, 0.05), max=(0.45, 1.30, INSIDE_D_M / 2.0),
                  note="where the arms go; nothing may be racked here"),),
    mass_kg=9.0, casing_ports=("glove-port-left", "glove-port-right"),
    note="two sleeved ports through the middle pane, sealed to it")

INTEGRAL_HEATER = OptionalAssembly(
    identity="cabinet.heater", label="electric element bank",
    provides=("heat",), requires=("electricity",),
    claims=(Claim(identity="cabinet.heater_bay",
                  min=(-0.55, 0.0, -0.58), max=(0.55, 0.12, -0.40),
                  note="the elements along the back of the floor, and their guard"),),
    mass_kg=HEATER_MASS_KG, casing_ports=("heater-conduit",),
    note="dry heat without steam: what a composite cure or an engine bake wants")

VACUUM_SET = OptionalAssembly(
    identity="cabinet.vacuum_set", label="pre-vacuum pump set",
    provides=("vacuum",), requires=("electricity",),
    claims=(Claim(identity="cabinet.pump_bay",
                  min=(0.70, -0.15, -0.60), max=(1.10, 0.30, -0.10),
                  note="pump, motor and the room to change its oil"),),
    mass_kg=48.0, casing_ports=("vacuum-line", "pump-drain"),
    note="without it a porous lay-up keeps its air")

AIR_SET = OptionalAssembly(
    identity="cabinet.air_set", label="compressed-air pressure set",
    provides=("pressure-hold",), requires=("electricity",),
    claims=(Claim(identity="cabinet.air_bay",
                  min=(0.70, -0.15, 0.10), max=(1.10, 0.30, 0.60),
                  note="receiver, regulator and the safety set"),),
    mass_kg=64.0, casing_ports=("air-inlet", "air-vent"),
    note="pressure without steam, for a cure that must stay dry")

SHORE_STEAM = OptionalAssembly(
    identity="cabinet.shore_steam_set", label="shore steam connection",
    provides=("steam",), requires=(),
    claims=(Claim(identity="cabinet.steam_manifold",
                  min=(-1.10, 0.60, -0.30), max=(-0.75, 0.90, 0.30),
                  note="reducing valve, strainer and trap set"),),
    mass_kg=24.0, casing_ports=("steam-inlet-flange", "condensate-return"),
    note="the sterilising duty needs steam, and the site has to have it")

CATALOGUE = (RACK_SET, ENGINE_CRADLE, GLOVE_PORTS, INTEGRAL_HEATER,
             VACUUM_SET, AIR_SET, SHORE_STEAM)
SITE_SUPPLIES = ("steam", "electricity", "compressed-air")

DUTIES = {
    "steam-sterilising": ("steam",),
    "composite-cure": ("heat", "vacuum", "pressure-hold"),
    "engine-bake": ("heat", "cradle"),
    # GLOVE WORK IS NOT RACK WORK. The arms' reach through the door is a
    # claimed volume, and it overlaps two rack levels: a full rack set in
    # front of the gloves is found as a space conflict by geometry, which
    # is the truth of it -- you cannot reach past a shelf.
    "glove-work": ("manipulation",),
}


def configuration(*options, duty: str | None = None, base_requires=(),
                  shore=SITE_SUPPLIES) -> Configuration:
    needs = tuple(DUTIES[duty]) if duty else tuple(base_requires)
    return Configuration(machine="plant.cabinet", fitted=tuple(options),
                         base_requires=needs, shore_supplies=tuple(shore))


# ---------------------------------------------------------------------
# the machine
# ---------------------------------------------------------------------

def build(identity: str = "plant.cabinet", working_pressure_pa: float = 300_000.0,
          *, options: tuple = (RACK_SET, INTEGRAL_HEATER, VACUUM_SET)
          ) -> ProductionGraph:
    """Author the cabinet, with the named options fitted.

    Everything the options claim is asserted on the graph as a clear
    volume, so a rack fitted where the cradle wants to be is caught by
    the graph's own check and not by a rule here."""
    fitted = {o.identity for o in options}
    g = ProductionGraph(identity=identity)
    g.assembly = "cabinet"
    p = working_pressure_pa
    t = wall_thickness_m(p)
    tg = glass_thickness_m(p)
    W, H, D = INSIDE_W_M, INSIDE_H_M, INSIDE_D_M
    hw, hh, hd = W / 2.0, H / 2.0, D / 2.0

    # ---- five plate walls, each a real box of real plate -------------
    # centre, half extents, and the (span, run) the ribs are laid on
    walls = {
        "floor":   ((0.0, -t / 2.0, 0.0), (hw + t, t / 2.0, hd + t), (W, D)),
        "ceiling": ((0.0, H + t / 2.0, 0.0), (hw + t, t / 2.0, hd + t), (W, D)),
        "back":    ((0.0, hh, -hd - t / 2.0), (hw + t, hh, t / 2.0), (W, H)),
        "left":    ((-hw - t / 2.0, hh, 0.0), (t / 2.0, hh, hd), (D, H)),
        "right":   ((+hw + t / 2.0, hh, 0.0), (t / 2.0, hh, hd), (D, H)),
    }
    wall_nodes = {}
    for name, (centre, half, (span, run)) in walls.items():
        # plate area is the wall's face; its thickness is t
        face = {"floor": (W + 2 * t) * (D + 2 * t), "ceiling": (W + 2 * t) * (D + 2 * t),
                "back": (W + 2 * t) * H, "left": D * H, "right": D * H}[name]
        mass = face * t * STEEL_DENSITY_KG_M3 + _rib_mass(span, run)
        wall = Prism(identity=f"{identity}.wall.{name}", kind="pressure-wall",
                     centre=centre, shape="box", half_extent=half,
                     material="boiler-plate", mass_kg=round(mass, 1),
                     attributes={"in_view": "plant", "plate_thickness_m": t,
                                 "rib_section": RIB_SECTION.designation,
                                 "rib_pitch_m": STIFFENER_PITCH_M,
                                 "working_pressure_pa": p, "mass_is_derived": True})
        wall_nodes[name] = (wall, None)
    # ports on each wall: its four edge midpoints, for the seams, plus
    # what each wall carries
    ports = {
        "floor": {"e_back": ("-z", (0.0, 0.0)), "e_left": ("-x", (0.0, 0.0)),
                  "e_right": ("+x", (0.0, 0.0)), "e_front": ("+z", (0.0, 0.0)),
                  "drain": ("+y", (-0.85, -0.85)), "heater": ("+y", (0.0, -0.80)),
                  "cradle": ("+y", (0.0, 0.0)),
                  "skid_fl": ("-y", (-0.85, +0.85)), "skid_fr": ("-y", (+0.85, +0.85)),
                  "skid_rl": ("-y", (-0.85, -0.85)), "skid_rr": ("-y", (+0.85, -0.85)),
                  "skid_ml": ("-y", (-0.85, 0.0)), "skid_mr": ("-y", (+0.85, 0.0))},
        "ceiling": {"e_back": ("-z", (0.0, 0.0)), "e_left": ("-x", (0.0, 0.0)),
                    "e_right": ("+x", (0.0, 0.0)), "e_front": ("+z", (0.0, 0.0)),
                    "vent": ("+y", (0.0, 0.0)), "relief": ("+y", (0.5, -0.5))},
        "back": {"e_floor": ("-y", (0.0, 0.0)), "e_ceiling": ("+y", (0.0, 0.0)),
                 "e_left": ("-x", (0.0, 0.0)), "e_right": ("+x", (0.0, 0.0)),
                 "steam_in": ("-z", (0.0, 0.85)), "gauge": ("-z", (0.6, 0.85)),
                 "conduit": ("-z", (0.0, -0.9)),
                 "spreader": ("+z", (0.0, 0.85))},
        "left": {"e_floor": ("-y", (0.0, 0.0)), "e_ceiling": ("+y", (0.0, 0.0)),
                 "e_back": ("-z", (0.0, 0.0)), "e_front": ("+z", (0.0, 0.0)),
                 "hinge_top": ("+z", (0.9, 0.0)), "hinge_bottom": ("+z", (-0.9, 0.0)),
                 **{f"rail_{i}": ("+x", ((y - hh) / hh, 0.0))
                    for i, y in enumerate(RACK_LEVELS_Y)}},
        "right": {"e_floor": ("-y", (0.0, 0.0)), "e_ceiling": ("+y", (0.0, 0.0)),
                  "e_back": ("-z", (0.0, 0.0)), "e_front": ("+z", (0.0, 0.0)),
                  "dog_top": ("+z", (0.9, 0.0)), "dog_bottom": ("+z", (-0.9, 0.0)),
                  # ON THE WALL THE PUMP STANDS BESIDE. Drawn on the back
                  # wall first, the vacuum line to a pump on the right ran
                  # straight through the chamber's rear corner -- a pipe
                  # through 18 mm of plate and a rack. A port goes where
                  # its line can reach it from outside.
                  "vacuum": ("+x", (0.85, -0.6)),
                  **{f"rail_{i}": ("-x", ((y - hh) / hh, 0.0))
                     for i, y in enumerate(RACK_LEVELS_Y)}},
    }
    for name, (wall, _) in wall_nodes.items():
        pts = {pn: wall.face_point(face, across=across)
               for pn, (face, across) in ports[name].items()}
        wall_nodes[name] = (wall, emit_prism(g, wall, pts, assembly="cabinet"))

    def wn(wall, port):
        return wall_nodes[wall][1][port]

    # ---- the seams: plate welded to plate along every edge -----------
    # THE CASE IS THE FRAME. Each pair of edge midpoints is one seam
    # weld; the members are short because the plates meet, and welded
    # because that is what a pressure cabinet is.
    seams = (("floor", "e_back", "back", "e_floor"), ("floor", "e_left", "left", "e_floor"),
             ("floor", "e_right", "right", "e_floor"),
             ("ceiling", "e_back", "back", "e_ceiling"), ("ceiling", "e_left", "left", "e_ceiling"),
             ("ceiling", "e_right", "right", "e_ceiling"),
             ("back", "e_left", "left", "e_back"), ("back", "e_right", "right", "e_back"))
    for a, ap_, b, bp in seams:
        g.edge(f"{identity}.seam.{a}_{b}", wn(a, ap_), wn(b, bp),
               "shell-attachment-weld", radius=t / 2.0, part_role="plate-seam-weld",
               load_path="the-case-is-the-frame")

    # ---- the door: a frame of square tube, mullioned, glazed ---------
    mullions = max(1, int(round(W / STIFFENER_PITCH_M)) - 1)
    frame_len = 2.0 * (W + H) + mullions * H
    door = Prism(identity=f"{identity}.door", kind="glazed-door",
                 centre=(0.0, hh, hd + DOOR_SECTION.side_m / 2.0), shape="box",
                 half_extent=(hw + DOOR_SECTION.side_m, hh + DOOR_SECTION.side_m,
                              DOOR_SECTION.side_m / 2.0),
                 material="steel-plate", mass_kg=round(frame_len * DOOR_SECTION.mass_per_m_kg, 1),
                 attributes={"in_view": "plant", "frame_section": DOOR_SECTION.designation,
                             "mullions": mullions, "mass_is_derived": True})
    d_ports = {"hinge_top": door.face_point("-x", across=(0.9, 0.0)),
               "hinge_bottom": door.face_point("-x", across=(-0.9, 0.0)),
               "dog_top": door.face_point("+x", across=(0.9, 0.0)),
               "dog_bottom": door.face_point("+x", across=(-0.9, 0.0)),
               "seal": door.face_point("-z"),
               **{f"pane_{i}": door.face_point("-z", across=(((i + 0.5) / (mullions + 1)) * 2.0 - 1.0, 0.0))
                  for i in range(mullions + 1)}}
    d_nodes = emit_prism(g, door, d_ports, assembly="cabinet")
    # the hinge carries the door open; the dogs and the hinge together
    # hold it shut, and shut is the state the pressure sees
    for side, wall in (("hinge_top", "left"), ("hinge_bottom", "left"),
                       ("dog_top", "right"), ("dog_bottom", "right")):
        g.edge(f"{identity}.door.{side}", d_nodes[side], wn(wall, side),
               "bolted-flange-mount", radius=0.014,
               part_role="door-hinge" if "hinge" in side else "door-dog")
    # the gasket: the pressure boundary is only closed when this is
    g.node(f"{identity}.port.door_face", d_ports["seal"].position, "vessel-port",
           port_kind="door_face", closure="glazed-clamped-door", bore_m=W,
           normal=(0.0, 0.0, 1.0), mating=True, in_view="plant")
    g.edge(f"{identity}.door.seal", d_nodes["seal"], f"{identity}.port.door_face",
           "port-face-seal", radius=0.008, part_role="door-gasket")
    # the ring the gasket seats on is a flange welded round the opening
    for wall in ("left", "right"):
        g.edge(f"{identity}.door_ring.{wall}", f"{identity}.port.door_face",
               wn(wall, "e_front"), "rigid-distance", radius=0.012,
               part_role="door-opening-flange")
    pane_w = W / (mullions + 1)
    for i in range(mullions + 1):
        cx = (i + 0.5) * pane_w - hw
        pane = Prism(identity=f"{identity}.pane.{i}", kind="pressure-window",
                     centre=(cx, hh, hd + tg / 2.0), shape="box",
                     half_extent=(pane_w / 2.0, hh, tg / 2.0), material="laminated-glass",
                     mass_kg=round(pane_w * H * tg * GLASS_DENSITY_KG_M3, 1),
                     attributes={"in_view": "plant", "glass_thickness_m": tg,
                                 "span_m": pane_w, "mass_is_derived": True,
                                 "working_pressure_pa": p})
        pn = emit_prism(g, pane, {"retainer": pane.face_point("+z")}, assembly="cabinet")
        # glazing retained by a bolted frame: clamped, not welded, because
        # glass is not welded to steel
        g.edge(f"{identity}.pane.{i}.retainer", pn["retainer"], d_nodes[f"pane_{i}"],
               "bolted-flange-mount", radius=0.006, part_role="glazing-retainer")
        if "cabinet.glove_ports" in fitted and i == mullions // 2:
            for side, x_off in (("left", -0.12), ("right", +0.12)):
                g.node(f"{identity}.port.glove_{side}",
                       (cx + x_off, 1.10, hd + tg), "vessel-port",
                       port_kind=f"glove_{side}", closure="glove-sleeve",
                       bore_m=0.20, normal=(0.0, 0.0, 1.0), in_view="plant",
                       note="through the middle pane, which is no thinner for it")
                g.edge(f"{identity}.port.glove_{side}.seat", pn["retainer"],
                       f"{identity}.port.glove_{side}", "bolted-flange-mount",
                       radius=0.004, part_role="glove-port-ring")

    # ---- the services through the walls --------------------------------
    closures = {
        ("back", "steam_in"): ("steam-stop-valve", 0.040),
        ("ceiling", "vent"): ("vent-valve", 0.065),
        ("ceiling", "relief"): ("safety-relief-valve", 0.040),
        ("back", "gauge"): ("pressure-gauge", 0.008),
        ("floor", "drain"): ("drain-trap", 0.050),
    }
    if "cabinet.vacuum_set" in fitted:
        closures[("right", "vacuum")] = ("vacuum-isolation-valve", 0.050)
    if "cabinet.heater" in fitted:
        closures[("back", "conduit")] = ("gland-and-junction-box", 0.032)
    for (wall, name), (closure, bore) in closures.items():
        pos = wall_nodes[wall][0].face_point(*ports[wall][name][:1],
                                             across=ports[wall][name][1]).position
        g.node(f"{identity}.port.{name}", pos, "vessel-port", port_kind=name,
               closure=closure, bore_m=bore, in_view="plant")
        g.edge(f"{identity}.port.{name}.seat", wn(wall, name), f"{identity}.port.{name}",
               "shell-attachment-weld", radius=0.004, part_role="port-seat")

    # ---- inside: racks or a cradle, and the heater ---------------------
    if "cabinet.rack_set" in fitted:
        for i, y in enumerate(RACK_LEVELS_Y):
            for side in ("left", "right"):
                g.node(f"{identity}.rail.{side}.{i}",
                       wall_nodes[side][0].face_point("+x" if side == "left" else "-x",
                                                      across=((y - hh) / hh, 0.0)).position,
                       "chamber-rail", in_view="plant", on_inner_surface=True)
                g.edge(f"{identity}.rail.{side}.{i}.weld", wn(side, f"rail_{i}"),
                       f"{identity}.rail.{side}.{i}", "shell-attachment-weld",
                       radius=0.006, part_role="rail-foot-weld")
            rack = Prism(identity=f"{identity}.rack.{i}", kind="load-rack",
                         centre=(0.0, y, 0.0), shape="box",
                         half_extent=(hw - 0.05, 0.015, hd - 0.05),
                         material="stainless-mesh", mass_kg=RACK_MASS_KG,
                         attributes={"in_view": "plant", "inside": True, "level": i})
            rn = emit_prism(g, rack, {"rail_l": rack.face_point("-x"),
                                      "rail_r": rack.face_point("+x")}, assembly="cabinet")
            g.edge(f"{identity}.rack.{i}.mount_l", rn["rail_l"], f"{identity}.rail.left.{i}",
                   "bolted-flange-mount", radius=0.008, part_role="rack-mount")
            g.edge(f"{identity}.rack.{i}.mount_r", rn["rail_r"], f"{identity}.rail.right.{i}",
                   "bolted-flange-mount", radius=0.008, part_role="rack-mount")
    if "cabinet.engine_cradle" in fitted:
        cradle = Prism(identity=f"{identity}.cradle", kind="engine-cradle",
                       centre=(0.0, 0.15, 0.0), shape="box", half_extent=(0.45, 0.15, 0.35),
                       material="steel-plate", mass_kg=CRADLE_MASS_KG,
                       attributes={"in_view": "plant", "inside": True})
        cn = emit_prism(g, cradle, {"base": cradle.face_point("-y")}, assembly="cabinet")
        g.edge(f"{identity}.cradle.mount", cn["base"], wn("floor", "cradle"),
               "bolted-flange-mount", radius=0.012, part_role="cradle-mount")
    if "cabinet.heater" in fitted:
        heater = Prism(identity=f"{identity}.heater", kind="heater-bank",
                       centre=(0.0, 0.06, -hd + 0.10), shape="box",
                       half_extent=(0.55, 0.06, 0.09), material="stainless-plate",
                       mass_kg=HEATER_MASS_KG,
                       attributes={"in_view": "plant", "inside": True, "heater_kw": HEATER_KW})
        hn = emit_prism(g, heater, {"base": heater.face_point("-y"),
                                    "terminals": heater.face_point("-z")}, assembly="cabinet")
        g.edge(f"{identity}.heater.mount", hn["base"], wn("floor", "heater"),
               "bolted-flange-mount", radius=0.008, part_role="heater-mount")
        # THE CONDUIT IS STRUCTURAL WHEN IT WANTS TO BE. A steel conduit
        # from the gland to the element terminals carries the terminal
        # box on its end, and says so; the walk then sees a real
        # cantilever rather than a hose.
        g.edge(f"{identity}.heater.conduit", f"{identity}.port.conduit", hn["terminals"],
               "electrical-conduit", radius=0.016, wall_m=0.002, alloy="a36",
               load_bearing=True, circuit_identity="heater-power",
               part_role="through-port")

    # ---- the steam spreader, inside along the back wall ---------------
    spreader = Prism(identity=f"{identity}.spreader", kind="steam-spreader",
                     centre=(0.0, H - 0.12, -hd + 0.06), shape="tube",
                     axis=(1.0, 0.0, 0.0), radius=0.020, inner_radius=0.016,
                     length=W * 0.85, material="stainless-pipe",
                     attributes={"in_view": "plant", "inside": True})
    sn = emit_prism(g, spreader, {"feed": spreader.side_point(math.radians(180.0))},
                    assembly="cabinet")
    g.edge(f"{identity}.spreader.flange", wn("back", "spreader"), sn["feed"],
           "bolted-flange-mount", radius=0.012, part_role="spreader-flange")
    g.edge(f"{identity}.steam.through", f"{identity}.port.steam_in", sn["feed"],
           "steam-line", radius=0.020, circuit_identity="steam", part_role="through-port")

    # ---- the skid, and the feet it stands on ---------------------------
    skid = Prism(identity=f"{identity}.skid", kind="machine-frame",
                 centre=(0.0, -t - SKID_HEIGHT_M / 2.0, 0.0), shape="box",
                 half_extent=(hw + t, SKID_HEIGHT_M / 2.0, hd + t),
                 material="a36",
                 mass_kg=round((4.0 * (W + D) + 2.0 * D) * SKID_SECTION.mass_per_m_kg, 1),
                 attributes={"in_view": "plant", "section": SKID_SECTION.designation,
                             "mass_is_derived": True})
    s_ports = {"floor_fl": skid.face_point("+y", across=(-0.85, +0.85)),
               "floor_fr": skid.face_point("+y", across=(+0.85, +0.85)),
               "floor_rl": skid.face_point("+y", across=(-0.85, -0.85)),
               "floor_rr": skid.face_point("+y", across=(+0.85, -0.85)),
               "floor_ml": skid.face_point("+y", across=(-0.85, 0.0)),
               "floor_mr": skid.face_point("+y", across=(+0.85, 0.0)),
               "pump": skid.face_point("+x", across=(0.0, -0.6))}
    for tag in ("fl", "fr", "rl", "rr", "ml", "mr"):
        s_ports[f"foot_{tag}"] = skid.face_point("-y", across=s_ports[f"floor_{tag}"].position[[0, 2]] / np.array([hw + t, hd + t]))
    s_nodes = emit_prism(g, skid, s_ports, assembly="cabinet",
                         port_kwargs={"bolt_radius_m": 0.012})
    for tag in ("fl", "fr", "rl", "rr", "ml", "mr"):
        node = next(n for n in g.nodes if n["identity"] == s_nodes[f"foot_{tag}"])
        node.update(port_role="structural-mount", joint="bolted-flange",
                    outward=(0.0, -1.0, 0.0))
        g.edge(f"{identity}.skid.weld.{tag}", s_nodes[f"floor_{tag}"], wn("floor", f"skid_{tag}"),
               "shell-attachment-weld", radius=0.010, part_role="skid-to-floor-weld")

    # ---- the vacuum pump, when fitted, on isolators ---------------------
    if "cabinet.vacuum_set" in fitted:
        pump = Prism(identity=f"{identity}.vacuum_pump", kind="vacuum-pump",
                     centre=(hw + t + 0.30, 0.05, -0.35), shape="box",
                     half_extent=(0.18, 0.14, 0.13), material="cast-iron", mass_kg=48.0,
                     attributes={"in_view": "plant", "stages": 2,
                                 "rotor_axis": (1.0, 0.0, 0.0), "rotor_rated_rpm": 1450.0,
                                 "rotor_shape": "vaned-rotor", "rotor_mass_kg": 12.0,
                                 "rotor_radius_m": 0.06, "balance_grade_mm_s": 6.3,
                                 "runs_in": ("evacuate",)})
        pn = emit_prism(g, pump, {"suction": pump.face_point("-z"),
                                  "base": pump.face_point("-y")}, assembly="cabinet")
        g.edge(f"{identity}.vacuum.line", f"{identity}.port.vacuum", pn["suction"],
               "vacuum-line", radius=0.020, circuit_identity="vacuum", part_role="through-port")
        g.edge(f"{identity}.vacuum_pump.isolator", pn["base"], s_nodes["pump"],
               "engine-mount-isolator", radius=0.012, part_role="pump-isolator-bracket")

    # ---- what the options claim, asserted on the graph ------------------
    # A CLAIM IS OCCUPIED BY THE THING THAT CLAIMS IT. turret_production
    # enforces a clear volume against every member, and exempts a member
    # that declares `clears=<volume>`; an assembly's own members are
    # exactly the ones that belong in its claim, so each is tagged with
    # the claim it lives in. Anything ELSE crossing it is still caught --
    # which is how the vacuum line through the rear corner was found.
    owned = {
        "cabinet.rack_set": [f"{identity}.rack.", f"{identity}.rail."],
        "cabinet.engine_cradle": [f"{identity}.cradle"],
        "cabinet.heater": [f"{identity}.heater"],
        "cabinet.vacuum_set": [f"{identity}.vacuum_pump", f"{identity}.vacuum.line"],
    }
    for o in options:
        for c in o.claims:
            g.clear_volumes.append(c.as_clear_volume(o.identity))
        prefixes = owned.get(o.identity, ())
        for e in g.edges:
            if any(e["identity"].startswith(px) for px in prefixes):
                # a rack lives in ITS level's claim, not every level's
                level = None
                if o.identity == "cabinet.rack_set":
                    m = re.search(r"\.(?:rack|rail\.(?:left|right))\.(\d+)", e["identity"])
                    level = int(m.group(1)) if m else None
                claim = (f"cabinet.rack_{level}" if level is not None
                         else o.claims[0].identity)
                e.setdefault("clears", claim)
    return g


# ---------------------------------------------------------------------
# what it does
# ---------------------------------------------------------------------

def cycle(identity: str = "plant.cabinet", *, options: tuple = (RACK_SET, INTEGRAL_HEATER, VACUUM_SET),
          live_vibration: bool = False) -> Cycle:
    """Load, evacuate, heat, hold, vent -- with whatever is fitted loaded."""
    fitted = {o.identity for o in options}
    charges = {}
    if "cabinet.rack_set" in fitted:
        charges.update({f"{identity}.rack.{i}": RACK_LOAD_KG for i in range(len(RACK_LEVELS_Y))})
    if "cabinet.engine_cradle" in fitted:
        charges[f"{identity}.cradle"] = ENGINE_LOAD_KG
    states = [
        OperatingState("load", note="door open; nothing runs"),
        OperatingState("evacuate", charges_kg=dict(charges),
                       curve=run_up(10) if "cabinet.vacuum_set" in fitted else (1.0,),
                       note="pump runs up and holds, if fitted"),
        OperatingState("heat", charges_kg=dict(charges), note="elements on; nothing spins"),
        OperatingState("hold", charges_kg=dict(charges), note="at pressure and temperature"),
        OperatingState("vent", charges_kg=dict(charges), note="blowing down"),
    ]
    return Cycle(machine=identity, states=tuple(states), live_vibration=live_vibration)


def describe(g: ProductionGraph, working_pressure_pa: float = 300_000.0) -> str:
    doc = g.as_document()
    total = sum(float(n.get("mass_kg", 0.0)) for n in doc["nodes"]
                if not n.get("wrench_point"))
    walls = sum(float(n["mass_kg"]) for n in doc["nodes"] if n.get("kind") == "pressure-wall")
    glass = sum(float(n["mass_kg"]) for n in doc["nodes"] if n.get("kind") == "pressure-window")
    return "\n".join([
        f"{doc['identity']}: {INSIDE_W_M:.2f} x {INSIDE_H_M:.2f} x {INSIDE_D_M:.2f} m inside, "
        f"{working_pressure_pa / 1e5:.1f} bar",
        f"  plate {wall_thickness_m(working_pressure_pa) * 1000:.1f} mm on "
        f"{RIB_SECTION.designation} ribs at {STIFFENER_PITCH_M * 1000:.0f} mm: {walls:.0f} kg of wall",
        f"  glass {glass_thickness_m(working_pressure_pa) * 1000:.1f} mm laminated in "
        f"{max(1, int(round(INSIDE_W_M / STIFFENER_PITCH_M)))} panes: {glass:.0f} kg",
        f"  {total:.0f} kg all told, on {sum(1 for n in doc['nodes'] if n.get('port_role') == 'structural-mount')} feet",
    ])
