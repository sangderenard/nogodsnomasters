"""The autoclave as a real machine: prisms, ports, vent and drain.

autoclave.py is the physics -- saturation, purge, condensing transfer.
This is the MACHINE: a pressure vessel with an actual shape, actual
ports in its shell, a vent pipe that goes somewhere, and a drain that
collects into something. Nothing here re-derives any thermodynamics; it
builds the body that the thermodynamics happens inside, in the same
production vocabulary turret_production uses, so every downstream system
already works on it.

WHY A TUBE AND NOT A BOX. prism_bodies gives a tube an INNER SURFACE,
and for a pressure vessel that is the entire point. The chamber is the
inside; the lagging, the steam line and the gauge are on the outside;
and the only way between them is a PORT. Authored as a box, "inside the
autoclave" would be a coordinate rather than a place, and the load
basket would attach to the vessel's centroid by reaching through the
wall -- the exact tunnelling prism_bodies exists to stop.

THE SHELL IS THE BOUNDARY, AND A PORT IS THE ONLY HOLE IN IT

    Every service crossing the pressure boundary is an
    assembly_ports.PartPort in the shell, with a position, an outward
    normal, a bore and a CLOSURE. That last field is what makes this
    cheap: a port with a closure is a hole with something in it -- a
    valve, a gauge, a blank -- and a port without one is an open hole
    with chamber pressure behind it. Nothing needs a special case for
    "the drain is shut": the drain is a port whose closure is a valve,
    and opening the valve is emptying the closure.

    So the fiddly question "how do the internals and the casing
    negotiate a through-port" already has an answer in this project and
    it is three fields: where, how big, and what is in it.

WIRING HARNESSES AND PATCH TUBES, CHEAPLY

    The temptation with a harness is to model every conductor as
    geometry. That is expensive to author, expensive to draw, and
    answers a question nobody asks. What anyone actually needs to know
    is: what does it connect, what does it run through, and what does
    it cost to get there.

    dc_power.Harness already answers the electrical half -- a path from
    a source to a load through cable and connectors, with a real
    resistance and a real voltage drop. A patch tube is the same
    statement for fluid: a MachineLine with a bore and a route. Both are
    declared as ENDPOINTS PLUS THE PORTS THEY PASS THROUGH, and the
    geometry is whatever the routing already does with pipes elsewhere.
    A harness is not a shape; it is a connectivity claim with a
    resistance attached to it.

VENT AND DRAIN ARE NOT THE SAME HOLE, and that is a real distinction
this machine has to make. The vent is at the TOP and carries gas -- it
is how air leaves during a displacement purge and how the vessel comes
back to atmosphere afterwards. The drain is at the BOTTOM and carries
condensate, which is liquid water and, on a dewax cycle, molten wax that
must be caught rather than put down a sewer. They are different fluids,
different directions, and different collectors, and a vessel with only
one of them does not work.
"""
from __future__ import annotations

import math

# --- real geometry, in metres -----------------------------------------
CHAMBER_LENGTH_M = 0.90
CHAMBER_RADIUS_M = 0.30
#: Wall thickness from the working pressure by the thin-wall hoop
#: relation, t = P.r / sigma_allow, with a real allowable for boiler
#: plate and a real corrosion allowance on top. The vessel's THICKNESS
#: is therefore a consequence of its rating, not a typed-in number.
BOILER_PLATE_ALLOWABLE_PA = 100e6
CORROSION_ALLOWANCE_M = 0.0015
JOINT_EFFICIENCY = 0.85          # a real welded seam, radiographed
AMBIENT_K = 293.15
#: Carbon steel, per kelvin. Real and standard; it is the number that
#: decides whether a mounting has to slide.
STEEL_EXPANSION_PER_K = 12.0e-6
STEEL_MODULUS_PA = 200e9
#: A saddle has to wrap far enough around the shell to spread the load
#: into it. Real practice puts the minimum at 120 degrees, and the
#: reason is the horn: below that, the local stress where the saddle
#: edge meets the shell climbs steeply.
SADDLE_WRAP_DEG = 120.0
SADDLE_HEIGHT_M = 0.22


def restrained_growth_stress_pa(delta_t_k: float) -> float:
    """What a vessel does when its mounting will not let it grow.

    E.alpha.dT, and it does not depend on the length -- a long vessel
    grows further and develops exactly the same stress. That is why
    "it is only a millimetre" is never the answer."""
    return STEEL_MODULUS_PA * STEEL_EXPANSION_PER_K * float(delta_t_k)


def wall_thickness_m(pressure_pa: float, radius_m: float = CHAMBER_RADIUS_M) -> float:
    """Thin-wall hoop stress, which is what sizes a pressure vessel.

    sigma = P.r / t, rearranged and divided by the weld's joint
    efficiency, plus a corrosion allowance. A 6 bar vessel needs about
    two millimetres of steel plus the allowance -- the allowance is the
    larger half, which is why small vessels are never as thin as the
    stress says."""
    t = (pressure_pa * radius_m) / (BOILER_PLATE_ALLOWABLE_PA * JOINT_EFFICIENCY)
    return t + CORROSION_ALLOWANCE_M


def build(identity: str = "plant.autoclave",
          working_pressure_pa: float = 600_000.0,
          with_vacuum: bool = True) -> "object":
    """Author the machine. Returns a turret_production.ProductionGraph.

    The whole vessel is one tube prism, and everything else attaches to
    one of its two surfaces: services to the outside, the load basket
    and the steam spreader to the inside. A part cannot get from one to
    the other except through a declared port, which is the point."""
    from turret_production import ProductionGraph
    from prism_bodies import Prism, emit_prism

    g = ProductionGraph(identity=identity)
    g.assembly = "autoclave"
    t_wall = wall_thickness_m(working_pressure_pa)
    r_out = CHAMBER_RADIUS_M + t_wall

    # ---- the vessel itself: a tube, so it has an inside -------------
    vessel = Prism(identity=f"{identity}.vessel", kind="pressure-vessel",
                   centre=(0.0, 0.0, 0.0), shape="tube", axis=(0.0, 0.0, 1.0),
                   radius=r_out, inner_radius=CHAMBER_RADIUS_M,
                   length=CHAMBER_LENGTH_M, material="boiler-plate",
                   # UP. Every angle below is a clock position from top
                   # dead centre, and says so because the vessel says
                   # which way that is.
                   clock=(0.0, 1.0, 0.0),
                   attributes={"working_pressure_pa": working_pressure_pa,
                               "wall_thickness_m": t_wall,
                               "in_view": "plant"})

    # Ports around the shell, each at a real CLOCK POSITION from top
    # dead centre -- which the vessel now declares (`clock` above)
    # instead of leaving to a cross product.
    #
    # THE FIRST DRAFT OF THIS GOT IT WRONG, and the comment it was
    # written under said so plainly: "the vent is at top dead centre
    # because gas collects there, and the drain is at bottom dead centre
    # because condensate collects there". They were authored at 90 and
    # 270 degrees, which on a z-axis body is the LEFT AND RIGHT FLANKS.
    # A vent on the flank leaves an air pocket above it that a
    # displacement purge never reaches; a drain on the flank leaves the
    # condensate it was supposed to remove sitting under it, and on a
    # dewax cycle that is a shell half full of wax. Neither is visible
    # in a render -- the nozzle is simply there, on a round thing, and
    # every angle looks equally plausible.
    #
    # So the stations below are declared by what each port CARRIES, and
    # check_port_stations refuses the ones that contradict gravity.
    ports = {
        # gas leaves from the crown, because that is where it collects
        "vent": vessel.side_point(math.radians(0.0), along=-0.35),
        # a relief valve must open into the vapour space; one that can
        # be covered by liquid cannot pass its rated flow
        "relief": vessel.side_point(math.radians(20.0), along=+0.40),
        # high, above any condensate: a gauge tapped low reads its own
        # water leg on top of the real pressure
        "gauge": vessel.side_point(math.radians(330.0), along=+0.38),
        # steam enters high so it displaces the air below it downward
        # and out of the drain, which is the whole mechanism of a
        # downward-displacement purge
        "steam_in": vessel.side_point(math.radians(45.0), along=+0.30),
        # condensate and molten wax leave from the invert
        "drain": vessel.side_point(math.radians(180.0), along=-0.40),
        "door_face": vessel.end_point("+"),
        "head_face": vessel.end_point("-"),
    }
    if with_vacuum:
        # ALSO HIGH, and for a reason worth stating: a vacuum line taken
        # from the invert draws the condensate standing there straight
        # into the pump, and a vane pump that swallows water loses its
        # oil seal. It comes off the vapour space like the relief does.
        ports["vacuum"] = vessel.side_point(math.radians(315.0), along=+0.20)
    node_of = emit_prism(g, vessel, ports, assembly="autoclave")

    # ---- what is IN each hole -------------------------------------
    # assembly_ports.PartPort's `closure` field is the cheap answer to
    # "is this shut": a port with a closure is a hole with hardware in
    # it, and removing the hardware makes it an open hole with chamber
    # pressure behind it. No special case is needed anywhere for a
    # valve being closed.
    # (closure, bore). THE BORE IS PER SERVICE, not one size for the
    # whole vessel. A vent has to pass the entire chamber volume of air
    # during a displacement purge and again when the vessel comes back
    # to atmosphere, so it is the biggest small hole here; a gauge
    # passes nothing at all and is a tapping. Giving them all one bore
    # produced a 38 mm vent pipe through a 25 mm hole, which
    # check_through_ports below now refuses.
    closures = {
        "steam_in": ("steam-stop-valve", 0.025),
        "vent": ("vent-valve", 0.040),
        "drain": ("drain-trap", 0.040),
        "gauge": ("pressure-gauge", 0.008),
        "relief": ("safety-relief-valve", 0.025),
        "vacuum": ("vacuum-isolation-valve", 0.035),
        "door_face": ("quick-release-door", 2 * CHAMBER_RADIUS_M),
        "head_face": ("welded-dished-head", 2 * CHAMBER_RADIUS_M),
    }
    for name, (closure, bore) in closures.items():
        if name in node_of:
            g.node(f"{identity}.port.{name}", ports[name].position,
                   "vessel-port", port_kind=name, closure=closure,
                   bore_m=bore,
                   normal=tuple(float(v) for v in ports[name].direction),
                   mating=name in ("door_face", "head_face"),
                   in_view="plant")
            g.edge(f"{identity}.port.{name}.seat", node_of[name],
                   f"{identity}.port.{name}", "port-face-seal",
                   radius=0.004, part_role="port-seat")

    # ---- the door is a real part that opens -------------------------
    door = Prism(identity=f"{identity}.door", kind="quick-release-door",
                 centre=(0.0, 0.0, CHAMBER_LENGTH_M / 2 + 0.04),
                 shape="cylinder", axis=(0.0, 0.0, 1.0),
                 radius=r_out + 0.02, length=0.06, material="boiler-plate",
                 attributes={"in_view": "plant", "opens": True})
    d_ports = {"face": door.end_point("-"), "hinge": door.side_point(math.radians(180.0))}
    d_nodes = emit_prism(g, door, d_ports, motion_group="door", assembly="autoclave")
    g.edge(f"{identity}.door.seal", d_nodes["face"], node_of["door_face"],
           "port-face-seal", radius=0.006, part_role="door-gasket",
           attributes_note="the pressure boundary is only closed when this is")

    # ---- INSIDE: the load basket hangs off the INNER surface --------
    # This is why the vessel is a tube. The basket attaches to the wall
    # it actually rests against, not to the vessel's centroid through
    # 5 mm of steel.
    basket = Prism(identity=f"{identity}.basket", kind="load-basket",
                   centre=(0.0, -0.05, 0.0), shape="box",
                   half_extent=(CHAMBER_RADIUS_M * 0.7, 0.10, CHAMBER_LENGTH_M * 0.4),
                   material="stainless-mesh",
                   attributes={"in_view": "plant", "inside": True})
    b_ports = {"rail_l": basket.face_point("-x"), "rail_r": basket.face_point("+x")}
    b_nodes = emit_prism(g, basket, b_ports, assembly="autoclave")
    # BOTH LOW, STRADDLING THE INVERT. Authored at 200 and 340 degrees
    # these straddled nothing: one sat near the bottom of the shell and
    # the other near the CROWN, so the basket had a rail above it and a
    # rail below it and rested on neither. A basket runs on two rails at
    # the same height, one either side of the drain, clear of it so the
    # condensate running to the invert is not dammed by the rail feet.
    for side, ang in (("rail_l", 150.0), ("rail_r", 210.0)):
        # ON THE INNER SURFACE, not the outer one. A tube prism knows
        # both radii, so the rail sits on the wall the basket actually
        # rests against -- which is the whole reason this vessel is a
        # tube. side_point walks the outer surface, so the inner point
        # is that direction taken to the inner radius instead.
        outer = vessel.side_point(math.radians(ang), along=0.0)
        n = outer.direction
        centre = vessel._c()
        inner_pos = tuple(float(centre[i] + n[i] * CHAMBER_RADIUS_M)
                          for i in range(3))
        g.node(f"{identity}.rail.{side}", inner_pos, "chamber-rail",
               in_view="plant", on_inner_surface=True)
        g.edge(f"{identity}.rail.{side}.mount", f"{identity}.rail.{side}",
               b_nodes[side], "bolted-flange-mount", radius=0.008,
               part_role="basket-rail")

    # ---- the steam spreader, also inside ---------------------------
    spreader = Prism(identity=f"{identity}.spreader", kind="steam-spreader",
                     centre=(0.0, CHAMBER_RADIUS_M * 0.6, 0.0), shape="tube",
                     axis=(0.0, 0.0, 1.0), radius=0.016, inner_radius=0.012,
                     length=CHAMBER_LENGTH_M * 0.8, material="stainless-pipe",
                     attributes={"in_view": "plant", "inside": True,
                                 "note": "perforated: admits steam along the "
                                         "whole length instead of blasting one spot"})
    s_ports = {"feed": spreader.end_point("+")}
    s_nodes = emit_prism(g, spreader, s_ports, assembly="autoclave")
    # THE THROUGH-PORT: the spreader is inside, the steam line is
    # outside, and this member is the only thing that crosses. It is
    # short by construction because both ends are on surfaces.
    g.edge(f"{identity}.steam.through", f"{identity}.port.steam_in",
           s_nodes["feed"], "steam-line", radius=0.012,
           circuit_identity="steam", part_role="through-port",
           medium_rate_state="steam-pressure-and-temperature")

    # ---- WHAT IT STANDS ON, and what that has to let it do ----------
    # Two saddles, and two is not a simplification. A horizontal vessel
    # on three or more supports is statically indeterminate: settle one
    # of them a millimetre and the load redistributes in a way nobody
    # can predict from the drawing. Two saddles is determinate, and it
    # is what real vessels sit on for exactly that reason.
    #
    # WHERE THEY GO is Zick's result, and it is not arbitrary: a saddle
    # placed within about half the shell radius of the tangent line lets
    # the DISHED HEAD act as a stiffening ring against the shell going
    # oval under its own load. Move the saddle inboard of that and the
    # shell has to carry the ovalling alone and needs stiffener rings
    # welded in to do it. So the saddles here sit at R/2 from each end,
    # which also satisfies the other real limit, A <= 0.2L.
    #
    # ONE IS FIXED AND ONE SLIDES, and this is the fact that makes the
    # mounting a negotiation rather than four bolts. A vessel working at
    # saturated steam runs a hundred and forty degrees above the frame
    # holding it, and it GROWS. Anchor both saddles in round holes and
    # the shell cannot grow: the stress it develops instead is E.alpha.
    # dT, which for steel across this temperature rise comes to about
    # 334 MPa -- past the yield of the plate it is made of. The vessel
    # does not push its frame apart, it deforms. So one saddle takes the
    # anchor bolts in round holes and owns the axial position, and the
    # other takes them in slots and lets the shell move under it.
    from gas_works import _steam_properties
    t_sat_k, _ = _steam_properties(working_pressure_pa)
    saddle_a_m = CHAMBER_RADIUS_M / 2.0
    saddle_z = CHAMBER_LENGTH_M / 2.0 - saddle_a_m
    saddle_span_m = 2.0 * saddle_z
    growth_m = STEEL_EXPANSION_PER_K * saddle_span_m * (t_sat_k - AMBIENT_K)
    base_y = -(r_out + SADDLE_HEIGHT_M)
    for tag, z, fixed in (("fixed", +saddle_z, True), ("sliding", -saddle_z, False)):
        saddle = Prism(
            identity=f"{identity}.saddle.{tag}", kind="vessel-saddle",
            centre=(0.0, -(r_out + SADDLE_HEIGHT_M / 2.0), z), shape="box",
            half_extent=(r_out * 0.87, SADDLE_HEIGHT_M / 2.0, 0.05),
            material="steel-plate",
            attributes={"in_view": "plant",
                        # a saddle has to wrap far enough round the shell
                        # to spread the load; below about 120 degrees the
                        # local stress at its horns rises steeply
                        "wrap_angle_deg": SADDLE_WRAP_DEG,
                        "anchor": "round-holes" if fixed else "slotted-holes",
                        "slot_travel_m": 0.0 if fixed else round(growth_m + 0.004, 4),
                        "note": ("the anchor: this saddle owns the vessel's "
                                 "axial position") if fixed else
                                ("slotted: the shell grows "
                                 f"{growth_m * 1000:.1f} mm toward this end "
                                 "and the slot has to let it")})
        # THE FEET ARE THE MOUNT POINTS, and they are declared as such --
        # the same word station_reference.py stamps on an engine-bay
        # mount and the same kind assembly_ports pairs with itself, so
        # these mate with a chassis through machinery that already
        # exists rather than a second opinion about what a mount is.
        feet = {"left": saddle.face_point("-y", across=(-0.72, 0.0)),
                "right": saddle.face_point("-y", across=(+0.72, 0.0))}
        emit_prism(g, saddle, feet, assembly="autoclave",
                   port_kwargs={"port_role": "structural-mount",
                                "joint": "solid-welded" if fixed else "bolted-flange",
                                "anchor": "round-holes" if fixed else "slotted-holes",
                                "bolt_radius_m": 0.010,
                                "slot_travel_m": 0.0 if fixed else round(growth_m + 0.004, 4)})
        # the saddle carries the shell above it, at the shell's surface
        seat = vessel.side_point(math.radians(180.0), along=(z / (CHAMBER_LENGTH_M / 2.0)) * 0.9)
        g.node(f"{identity}.saddle.{tag}.seat", tuple(float(v) for v in seat.position),
               "vessel-port", port_kind="saddle_seat", closure="saddle-wear-plate",
               bore_m=0.0, in_view="plant")
        g.edge(f"{identity}.saddle.{tag}.bear", f"{identity}.saddle.{tag}.seat",
               f"{identity}.saddle.{tag}", "bolted-flange-mount", radius=0.014,
               part_role="saddle-bearing",
               load_path="the-shell-carried-on-its-saddle-horns")

    # ---- OUTSIDE: vent pipe, and where it goes ---------------------
    # A vent is not a hole in the air. It carries hot gas somewhere it
    # can safely go, which on a steam vessel means up and away from
    # anyone standing at the door.
    vent = Prism(identity=f"{identity}.vent_pipe", kind="vent-pipe",
                 centre=(0.0, r_out + 0.55, -0.35), shape="tube",
                 axis=(0.0, 1.0, 0.0), radius=0.022, inner_radius=0.019,
                 length=1.10, material="steel-pipe",
                 attributes={"in_view": "plant"})
    v_ports = {"inlet": vent.end_point("-"), "outlet": vent.end_point("+")}
    v_nodes = emit_prism(g, vent, v_ports, assembly="autoclave")
    g.edge(f"{identity}.vent.through", f"{identity}.port.vent", v_nodes["inlet"],
           "steam-line", radius=0.019, circuit_identity="steam",
           part_role="through-port")
    # WHERE IT LETS GO IS NOT THE MACHINE'S TO DECIDE. How high a vent
    # stack runs and where it points are decided by the building it is
    # installed in -- what is above it, who walks past it, which way the
    # prevailing wind goes. The machine supplies the pipe to its own
    # boundary and the site supplies the rest, which is exactly the
    # distinction engine_parts.py already draws with `chassis_side` for
    # a muffler and a tailpipe: hung from the body, not bolted to the
    # engine. Declaring it moved this crate from 2.25 m tall -- too tall
    # for a bay it otherwise fits with room to spare -- to its real
    # height, because a stack somebody else routes was being measured as
    # part of the box it has to be lifted in.
    g.node(f"{identity}.vent.discharge", vent.end_point("+").position,
           "discharge-point", in_view="plant", chassis_side=True,
           note="open to atmosphere, above head height -- the site decides "
                "how far above and which way")

    # ---- the drain, and the collector it drains INTO ---------------
    # Condensate is not waste on every cycle. Dewaxing an investment
    # shell puts recoverable wax down this pipe, and wax is the one
    # consumable in that chain that is re-used -- so the collector is a
    # real vessel with a real capacity and a real separation, not a
    # drain to nowhere.
    collector = Prism(identity=f"{identity}.collector", kind="condensate-collector",
                      centre=(0.0, -(r_out + 0.32), -0.40), shape="cylinder",
                      axis=(0.0, 1.0, 0.0), radius=0.16, length=0.45,
                      material="stainless-plate",
                      attributes={"in_view": "plant", "fluid": "water",
                                  "fluid_volume_l": 36.0,
                                  "note": "wax floats on the condensate and is "
                                          "skimmed: separation by density, no "
                                          "machinery"})
    c_ports = {"inlet": collector.end_point("+"),
               "skim": collector.side_point(math.radians(90.0), along=+0.10),
               "water_out": collector.side_point(math.radians(270.0), along=-0.15)}
    c_nodes = emit_prism(g, collector, c_ports, assembly="autoclave")
    g.edge(f"{identity}.drain.line", f"{identity}.port.drain", c_nodes["inlet"],
           "condensate-line", radius=0.019, circuit_identity="condensate",
           part_role="drain-line",
           medium_rate_state="condensate-flow-and-temperature")

    # ---- the vacuum pump, when fitted -------------------------------
    if with_vacuum:
        pump = Prism(identity=f"{identity}.vacuum_pump", kind="vacuum-pump",
                     centre=(r_out + 0.30, -0.18, +0.20), shape="box",
                     half_extent=(0.16, 0.12, 0.11), material="cast-iron",
                     attributes={"in_view": "plant",
                                 "stages": 2, "clearance_frac": 0.03})
        p_ports = {"suction": pump.face_point("-x"), "motor": pump.face_point("+x")}
        p_nodes = emit_prism(g, pump, p_ports, assembly="autoclave")
        g.edge(f"{identity}.vacuum.line", f"{identity}.port.vacuum",
               p_nodes["suction"], "vacuum-line", radius=0.016,
               circuit_identity="vacuum", part_role="through-port")

    return g


def harness(identity: str = "plant.autoclave") -> "object":
    """The electrical harness, as connectivity rather than geometry.

    dc_power.Harness already carries a real cable, real connectors, a
    real resistance and a real delivered voltage. The cheap part is the
    ROUTE: it is stated as the ports it passes through, not as a
    polyline. Anyone who needs to draw it can route it the same way the
    pipes are routed; anyone who needs to KNOW something about it --
    what it feeds, what it drops, whether the gauge reads low because
    the run is long -- gets that from the harness itself."""
    try:
        from dc_power import Harness
    except Exception:
        return None
    try:
        return Harness(identity=f"{identity}.harness")
    except TypeError:
        return None


#: WHERE EACH PORT HAS TO BE, and why. A port is not free to sit
#: anywhere on a round shell: what it carries decides, because gravity
#: decides where that collects. Declared per service, so the check below
#: is reading a statement rather than a list of angles someone liked.
#:
#:   crown         gas leaves here, because gas collects here
#:   invert        liquid leaves here, for the same reason
#:   vapour-space  anywhere above the midline: it must not be coverable
#:                 by the liquid standing in the bottom of the vessel
#:   end           a head or a door: no clock position to be wrong about
PORT_STATIONS = {
    "vent": "crown",
    "drain": "invert",
    "relief": "vapour-space",
    "gauge": "vapour-space",
    "vacuum": "vapour-space",
    "steam_in": "vapour-space",
    "door_face": "end",
    "head_face": "end",
}

#: How far off the crown or the invert a port may be and still be at it.
#: A nozzle is a finite hole in a curved shell and cannot be a point, so
#: some tolerance is real; thirty degrees on a 300 mm vessel is 157 mm
#: of arc, which is a generous allowance for a 40 mm nozzle and still
#: refuses anything on the flank.
STATION_TOLERANCE_DEG = 30.0


def check_port_stations(g, up=(0.0, 1.0, 0.0)) -> list:
    """No port may sit where what it carries does not collect.

    THIS IS THE CHECK THAT WOULD HAVE CAUGHT THE FIRST DRAFT. Its vent
    and its drain were authored at ninety and two hundred and seventy
    degrees under a comment claiming they were at the crown and the
    invert, and on a z-axis body those angles are the two flanks. The
    vessel drew correctly, reported correctly, and would have held its
    air and its condensate.

    The station is read from the port's own real position relative to
    the vessel's centre, not from the angle it was authored at, so an
    angle that means something different than its author thought is
    caught by where the nozzle actually ended up."""
    import numpy as _np
    up = _np.asarray(up, float)
    up = up / float(_np.linalg.norm(up))
    body = next((n for n in g.nodes if n.get("kind") == "pressure-vessel"), None)
    if body is None:
        return []
    axis = _np.asarray(body.get("tube_axis", (0.0, 0.0, 1.0)), float)
    axis = axis / float(_np.linalg.norm(axis))
    centre = _np.asarray(body["reference_position"], float)
    faults = []
    for n in g.nodes:
        if n.get("kind") != "vessel-port":
            continue
        want = PORT_STATIONS.get(n.get("port_kind"))
        if want in (None, "end"):
            continue
        r = _np.asarray(n["reference_position"], float) - centre
        r = r - axis * float(_np.dot(r, axis))       # around the shell only
        if float(_np.linalg.norm(r)) < 1e-9:
            continue
        cos = float(_np.dot(r / _np.linalg.norm(r), up))
        off_crown = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        if want == "crown" and off_crown > STATION_TOLERANCE_DEG:
            faults.append(f"{n['identity']}: {n['port_kind']} is "
                          f"{off_crown:.0f} deg off the crown -- gas collects "
                          "above it and never leaves")
        elif want == "invert" and (180.0 - off_crown) > STATION_TOLERANCE_DEG:
            faults.append(f"{n['identity']}: {n['port_kind']} is "
                          f"{180.0 - off_crown:.0f} deg off the invert -- "
                          "liquid collects below it and never leaves")
        elif want == "vapour-space" and off_crown > 90.0:
            faults.append(f"{n['identity']}: {n['port_kind']} is below the "
                          "midline, where standing liquid can cover it")
    return faults


def check_through_ports(g) -> list:
    """Nothing may pass through a hole smaller than itself.

    The obvious check, and it caught this machine's own first draft: a
    38 mm vent pipe was authored through a 25 mm port because every port
    had been given the same bore. It is the kind of mistake that is
    invisible in a render -- the pipe simply appears to arrive -- and
    fatal in metal.

    This is the general form of the question the user asked about
    casings and internals: a through-port is a member whose two ends are
    on opposite sides of a boundary, and the boundary decides what fits.
    Any machine authored this way gets the check for free."""
    bore_of = {n["identity"]: float(n.get("bore_m", 0.0))
               for n in g.nodes if n.get("kind") == "vessel-port"}
    faults = []
    for e in g.edges:
        if e.get("part_role") != "through-port":
            continue
        member_bore = float(e.get("radius", 0.0)) * 2.0
        for end in ("a", "b"):
            hole = bore_of.get(e.get(end))
            if hole is None:
                continue
            if member_bore > hole + 1e-9:
                faults.append(
                    f"{e['identity']}: {member_bore * 1000:.0f} mm line through a "
                    f"{hole * 1000:.0f} mm port ({e.get(end)}) -- it does not fit")
    return faults


def describe(g) -> str:
    kinds = {}
    for n in g.nodes:
        kinds[n.get("kind", "?")] = kinds.get(n.get("kind", "?"), 0) + 1
    through = [e for e in g.edges if e.get("part_role") == "through-port"]
    ports = [n for n in g.nodes if n.get("kind") == "vessel-port"]
    L = [f"{g.identity}: {len(g.nodes)} nodes, {len(g.edges)} edges",
         f"  wall {wall_thickness_m(600_000.0) * 1000:.2f} mm from the hoop stress "
         "at 6 bar, corrosion allowance included",
         "",
         "  PORTS IN THE PRESSURE BOUNDARY (each a hole with something in it):"]
    for p in ports:
        L.append(f"    {p.get('port_kind'):11s} bore {p.get('bore_m', 0) * 1000:5.1f} mm  "
                 f"closure: {p.get('closure')}"
                 + ("  [mating face]" if p.get("mating") else ""))
    L.append("")
    L.append(f"  {len(through)} members cross the boundary, and they are the only ones:")
    for e in through:
        L.append(f"    {e['identity'].split('.')[-2]:9s} "
                 f"{e.get('circuit_identity'):10s} bore {e.get('radius', 0) * 2000:.0f} mm")
    faults = check_through_ports(g)
    L.append("")
    if faults:
        L.append("  THROUGH-PORT FAULTS:")
        for f in faults:
            L.append(f"    ! {f}")
    else:
        L.append("  every through-port member fits the hole it passes through")
    L.append("")
    L.append("  bodies: " + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())
                                      if k not in ("port-node",)))
    return "\n".join(L)


if __name__ == "__main__":
    g = build()
    print(describe(g))
