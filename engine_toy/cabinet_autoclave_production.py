"""A glass-faced cabinet autoclave: square, walk-in sized, multipurpose.

THE TUBE AUTOCLAVE IS SMALL BECAUSE A TUBE IS CHEAP. Hoop stress sizes a
round shell at a few millimetres, and that is why almost every pressure
vessel is round. The machine actually wanted here is a different thing:
a square cabinet with a glass front, big enough to take an assembled
engine or a composite lay-up on racks, with the full suite of pressure,
vacuum, heat and venting -- and with glove ports as one of the things
it can be supplied with. A square cabinet pays for its shape in plate,
and this module makes it pay honestly.

THE BOX IS GENERIC AND THE AUTOCLAVE IS WHAT GOES THROUGH ITS WALLS.
cabinet.py builds the plate box, its mullioned glass door on a piano
hinge, its racks and its skid, and sizes the plate and the glass by
bending -- 18 mm of boiler plate and 31 mm of laminated glass at three
bar, against the tube vessel's under-four. What makes it an autoclave
is declared here and added through the box's `extras` hook: the steam
spreader, the vent, relief, gauge and drain, the heater bank and its
load-bearing conduit, the vacuum pump on isolators. The same box with
a drum behind the door is a washer; see appliance_production.

WHAT IT CAN BE SUPPLIED WITH is declared in machine_options terms --
racks, an engine cradle, glove ports, a heater, a vacuum set, an air
set for pressure hold -- and racks and the cradle want the same volume,
so they exclude each other by geometry, as do a full rack set and the
reach of the glove arms.
"""
from __future__ import annotations

import math

import cabinet as cb
from cabinet import Rack, Door, Window, CabinetSpec, Claim, CORROSION_ALLOWANCE_M, RIB_SECTION
from prism_bodies import Prism, emit_prism
from machine_options import Configuration, OptionalAssembly
from operating_states import Cycle, OperatingState, run_up

INSIDE_W_M = 1.20
INSIDE_H_M = 1.50
INSIDE_D_M = 1.20
STIFFENER_PITCH_M = 0.40
RACK_LEVELS_Y = (0.35, 0.70, 1.05, 1.40)
RACK_MASS_KG = 18.0
RACK_LOAD_KG = 30.0
CRADLE_MASS_KG = 60.0
ENGINE_LOAD_KG = 250.0
HEATER_KW = 24.0
HEATER_MASS_KG = 25.0


def wall_thickness_m(pressure_pa: float, pitch_m: float = STIFFENER_PITCH_M) -> float:
    return cb.wall_thickness_m(pressure_pa, pitch_m)


def glass_thickness_m(pressure_pa: float, pitch_m: float = STIFFENER_PITCH_M) -> float:
    return cb.glass_thickness_m(pressure_pa, pitch_m)


# ---------------------------------------------------------------------
# what it can be supplied with
# ---------------------------------------------------------------------

RACK_SET = OptionalAssembly(
    identity="cabinet.rack_set", label="four-level rack set",
    provides=("racking",), requires=(),
    claims=tuple(Claim(identity=f"cabinet.rack_{i}",
                       min=(-INSIDE_W_M / 2.0 + 0.05, y - 0.02, -INSIDE_D_M / 2.0 + 0.05),
                       max=(INSIDE_W_M / 2.0 - 0.05, min(y + 0.30, INSIDE_H_M - 0.02),
                            INSIDE_D_M / 2.0 - 0.05),
                       note=f"rack level {i} and the headroom above it")
                 for i, y in enumerate(RACK_LEVELS_Y)),
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
    # GLOVE WORK IS NOT RACK WORK: the arms' reach is a claimed volume
    # that overlaps two rack levels, and geometry refuses the pair
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

def spec_for(options, identity: str = "plant.cabinet",
             working_pressure_pa: float = 300_000.0) -> CabinetSpec:
    """The generic box this autoclave is, given what is fitted."""
    fitted = {o.identity for o in options}
    racks = (tuple(Rack(y, mass_kg=RACK_MASS_KG, load_kg=RACK_LOAD_KG) for y in RACK_LEVELS_Y)
             if "cabinet.rack_set" in fitted else ())
    window = Window(kind="panes",
                    accessories=("glove-ports",) if "cabinet.glove_ports" in fitted else ())
    ports = {"ceiling": {"vent": ("+y", (0.0, 0.0)), "relief": ("+y", (0.5, -0.5))},
             "back": {"steam_in": ("-z", (0.0, 0.85)), "gauge": ("-z", (0.6, 0.85)),
                      "conduit": ("-z", (0.0, -0.9)), "spreader": ("+z", (0.0, 0.85))},
             "floor": {"drain": ("+y", (-0.85, -0.85)), "heater": ("+y", (0.0, -0.80)),
                       "cradle": ("+y", (0.0, 0.0))},
             # ON THE WALL THE PUMP STANDS BESIDE, so its line reaches it
             # from outside instead of through the rear corner
             "right": {"vacuum": ("+x", (0.85, -0.6))}}
    return CabinetSpec(identity=identity, inside_m=(INSIDE_W_M, INSIDE_H_M, INSIDE_D_M),
                       working_pressure_pa=working_pressure_pa, construction="welded-plate",
                       stiffener_pitch_m=STIFFENER_PITCH_M, racks=racks,
                       door=Door(kind="glazed", hinge="piano-hinge", dogs=2, window=window),
                       feet_per_side=3, wall_ports=ports, label="cabinet")


def build(identity: str = "plant.cabinet", working_pressure_pa: float = 300_000.0,
          *, options: tuple = (RACK_SET, INTEGRAL_HEATER, VACUUM_SET)):
    """Author the cabinet, with the named options fitted.

    Returns the ProductionGraph, as the tube autoclave's build does."""
    fitted = {o.identity for o in options}
    spec = spec_for(options, identity, working_pressure_pa)
    W, H, D = spec.inside_m
    hw, hh, hd = W / 2.0, H / 2.0, D / 2.0

    def extras(built: cb.Built):
        g = built.graph
        t = built.t
        # ---- services through the walls ----
        built.port("back", "steam_in", "steam-stop-valve", 0.040)
        built.port("ceiling", "vent", "vent-valve", 0.065)
        built.port("ceiling", "relief", "safety-relief-valve", 0.040)
        built.port("back", "gauge", "pressure-gauge", 0.008)
        built.port("floor", "drain", "drain-trap", 0.050)
        # ---- the steam spreader, inside along the back wall ----
        spreader = Prism(identity=f"{identity}.spreader", kind="steam-spreader",
                         centre=(0.0, H - 0.12, -hd + 0.06), shape="tube",
                         axis=(1.0, 0.0, 0.0), radius=0.020, inner_radius=0.016,
                         length=W * 0.85, material="stainless-pipe",
                         attributes={"in_view": "plant", "inside": True})
        sn = emit_prism(g, spreader, {"feed": spreader.side_point(math.pi)}, assembly="cabinet")
        g.edge(f"{identity}.spreader.flange", built.wn("back", "spreader"), sn["feed"],
               "bolted-flange-mount", radius=0.012, part_role="spreader-flange")
        g.edge(f"{identity}.steam.through", f"{identity}.port.steam_in", sn["feed"],
               "steam-line", radius=0.020, circuit_identity="steam", part_role="through-port")
        # ---- the cradle ----
        if "cabinet.engine_cradle" in fitted:
            cradle = Prism(identity=f"{identity}.cradle", kind="engine-cradle",
                           centre=(0.0, 0.15, 0.0), shape="box", half_extent=(0.45, 0.15, 0.35),
                           material="steel-plate", mass_kg=CRADLE_MASS_KG,
                           attributes={"in_view": "plant", "inside": True})
            cn = emit_prism(g, cradle, {"base": cradle.face_point("-y")}, assembly="cabinet")
            g.edge(f"{identity}.cradle.mount", cn["base"], built.wn("floor", "cradle"),
                   "bolted-flange-mount", radius=0.012, part_role="cradle-mount")
        # ---- the heater and its load-bearing conduit ----
        if "cabinet.heater" in fitted:
            built.port("back", "conduit", "gland-and-junction-box", 0.032)
            heater = Prism(identity=f"{identity}.heater", kind="heater-bank",
                           centre=(0.0, 0.06, -hd + 0.10), shape="box",
                           half_extent=(0.55, 0.06, 0.09), material="stainless-plate",
                           mass_kg=HEATER_MASS_KG,
                           attributes={"in_view": "plant", "inside": True, "heater_kw": HEATER_KW})
            hn = emit_prism(g, heater, {"base": heater.face_point("-y"),
                                        "terminals": heater.face_point("-z")}, assembly="cabinet")
            g.edge(f"{identity}.heater.mount", hn["base"], built.wn("floor", "heater"),
                   "bolted-flange-mount", radius=0.008, part_role="heater-mount")
            # THE CONDUIT IS STRUCTURAL WHEN IT WANTS TO BE: it carries the
            # terminal box on its end, and says so
            g.edge(f"{identity}.heater.conduit", f"{identity}.port.conduit", hn["terminals"],
                   "electrical-conduit", radius=0.016, wall_m=0.002, alloy="a36",
                   load_bearing=True, circuit_identity="heater-power", part_role="through-port")
        # ---- the vacuum pump, on isolators, beside the skid ----
        if "cabinet.vacuum_set" in fitted:
            built.port("right", "vacuum", "vacuum-isolation-valve", 0.050)
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
            g.edge(f"{identity}.vacuum_pump.isolator", pn["base"], built.skid_nodes["side_mount"],
                   "engine-mount-isolator", radius=0.012, part_role="pump-isolator-bracket")
        # ---- what the options claim, and whose members live in them ----
        owned = {"cabinet.engine_cradle": [f"{identity}.cradle"],
                 "cabinet.heater": [f"{identity}.heater"],
                 "cabinet.vacuum_set": [f"{identity}.vacuum_pump", f"{identity}.vacuum.line"]}
        for o in options:
            if o.identity == "cabinet.rack_set":
                continue            # the box claimed and tagged its own racks
            for c in o.claims:
                built.claim(c, o.identity)
            if o.identity in owned:
                built.tag(owned[o.identity], o.claims[0].identity)

    return cb.build(spec, extras=extras).graph


def cycle(identity: str = "plant.cabinet", *,
          options: tuple = (RACK_SET, INTEGRAL_HEATER, VACUUM_SET),
          live_vibration: bool = False) -> Cycle:
    """Load, evacuate, heat, hold, vent -- with whatever is fitted loaded."""
    fitted = {o.identity for o in options}
    charges = cb.charges(spec_for(options, identity))
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


def describe(g, working_pressure_pa: float = 300_000.0) -> str:
    doc = g.as_document()
    total = sum(float(n.get("mass_kg", 0.0)) for n in doc["nodes"] if not n.get("wrench_point"))
    walls = sum(float(n["mass_kg"]) for n in doc["nodes"] if n.get("kind") == "pressure-wall")
    glass = sum(float(n["mass_kg"]) for n in doc["nodes"] if n.get("kind") == "pressure-window")
    return "\n".join([
        f"{doc['identity']}: {INSIDE_W_M:.2f} x {INSIDE_H_M:.2f} x {INSIDE_D_M:.2f} m inside, "
        f"{working_pressure_pa / 1e5:.1f} bar",
        f"  plate {wall_thickness_m(working_pressure_pa) * 1000:.1f} mm on "
        f"{RIB_SECTION.designation} ribs at {STIFFENER_PITCH_M * 1000:.0f} mm: {walls:.0f} kg of wall",
        f"  glass {glass_thickness_m(working_pressure_pa) * 1000:.1f} mm laminated in "
        f"{max(1, int(round(INSIDE_W_M / STIFFENER_PITCH_M)))} panes: {glass:.0f} kg",
        f"  {total:.0f} kg all told, on "
        f"{sum(1 for n in doc['nodes'] if n.get('port_role') == 'structural-mount')} feet",
    ])
