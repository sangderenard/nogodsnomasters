"""A cabinet: the generic box that a great many machines are.

An autoclave is a box with a door and racks in it. A washing machine is
a box with a door and a drum behind it, machinery underneath and a
drawer on top. A dryer is the same box with the drum on rollers and a
heater where the pump was. A curing oven, a kiln, a fume cupboard, a
glove box: a box, a door, something inside, plant above or below. The
first cabinet was written as an autoclave and every one of its parts
was really one of these, so this is that cabinet with the autoclave
taken out of it.

WHAT IS DECLARED. A CabinetSpec says what the box is:

    inside          (W, H, D) in metres; the door is the +z face
    construction    welded-plate, sized by pressure and bending, or
                    screwed-sheet, sized by a declared gauge and held
                    by seams -- see milspec and joints
    racks           any number, each at its own height with its own
                    size and load; zero is a cabinet with nothing in it
    top, bottom     optional plant compartments of a declared height,
                    each carrying a loadout of PlantItems -- motors,
                    pumps, heaters, drawers -- bolted, isolated or
                    suspended, with rotors declared where they spin
    door            glazed (mullioned panes) or solid, on a piano
                    hinge, with dogs to shut it, and window accessories:
                    a porthole, glove ports
    drum            optionally, a horizontal drum behind the door: on a
                    suspension (a washer), or on rollers (a dryer)

WHAT IS NOT DECLARED HERE. The services -- steam, vacuum, heat, water,
drain -- are the machine's own and are added by the machine module
through the `extras` hook, which gets a Built context with every wall
port and skid node by name. The generic box does not know what an
autoclave is; the autoclave module knows what a box is.

THE THREE THINGS EVERY CABINET GETS RIGHT ONCE. The plates or sheets
are joined by the construction's own seam so the walk sees the case as
the frame. Every optional thing claims its volume and its own members
are tagged as belonging there, so the graph check catches intruders
and only intruders. And the feet are declared mounts on a skid, so
feet.from_mounts and the mode table work on any cabinet unchanged.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from prism_bodies import Prism, emit_prism
from turret_production import ProductionGraph
from milspec import AngleSection, SquareTubeSection, SheetSection
from joints import Seam
from machine_options import Claim

# --- construction constants ------------------------------------------
PLATE_BETA = 0.50
BOILER_PLATE_ALLOWABLE_PA = 100e6
JOINT_EFFICIENCY = 0.85
CORROSION_ALLOWANCE_M = 0.0015
STEEL_DENSITY_KG_M3 = 7850.0
GLASS_ALLOWABLE_PA = 25e6
GLASS_DENSITY_KG_M3 = 2500.0
RIB_SECTION = AngleSection(leg_m=0.060, thickness_m=0.006, material="a36",
                           designation="60x60x6 angle A36")
DOOR_SECTION = SquareTubeSection(side_m=0.080, wall_m=0.005, material="a36",
                                 designation="80x80x5 SHS A36")
SKID_SECTION = SquareTubeSection(side_m=0.080, wall_m=0.005, material="a36",
                                 designation="80x80x5 SHS A36")
LIGHT_SKID_SECTION = SquareTubeSection(side_m=0.040, wall_m=0.002, material="a36",
                                       designation="40x40x2 SHS A36")
LIGHT_DOOR_SECTION = SquareTubeSection(side_m=0.025, wall_m=0.0015, material="a36",
                                       designation="25x25x1.5 SHS A36")
#: A tinsmith's seam: a #10 screw every three inches.
SHEET_SEAM = Seam(fastener="sheet-metal-screw", per_inch=1.0 / 3.0,
                  sheet_thickness_m=0.0008, fastener_diameter_m=0.0048)


def plate_thickness_m(pressure_pa: float, span_m: float, allowable_pa: float, *,
                      beta: float = PLATE_BETA, efficiency: float = 1.0,
                      allowance_m: float = 0.0) -> float:
    """A flat plate strip under pressure, by bending: t = b.sqrt(beta.p/sigma)."""
    return span_m * math.sqrt(beta * abs(pressure_pa) / (allowable_pa * efficiency)) + allowance_m


def wall_thickness_m(pressure_pa: float, pitch_m: float) -> float:
    return plate_thickness_m(pressure_pa, pitch_m, BOILER_PLATE_ALLOWABLE_PA,
                             efficiency=JOINT_EFFICIENCY, allowance_m=CORROSION_ALLOWANCE_M)


def glass_thickness_m(pressure_pa: float, span_m: float, minimum_m: float = 0.004) -> float:
    """Glass by bending, and never thinner than a pane can be handled at."""
    return max(minimum_m, plate_thickness_m(pressure_pa, span_m, GLASS_ALLOWABLE_PA))


# ---------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Rack:
    """One shelf: where it is, how big, what it weighs and carries."""
    y_m: float
    #: (half x, half y, half z); None is the full chamber less a clearance
    half_extent_m: tuple | None = None
    mass_kg: float = 18.0
    load_kg: float = 30.0
    headroom_m: float = 0.30
    material: str = "stainless-mesh"


@dataclass(frozen=True)
class PlantItem:
    """One thing in a plant compartment."""
    name: str
    kind: str
    half_extent_m: tuple
    mass_kg: float
    #: where on the compartment's deck, as fractions of its half extents
    at: tuple = (0.0, 0.0)
    #: bolted | isolated (engine-mount-isolator) | suspended (own springs)
    mount: str = "bolted"
    suspension_n_per_m: float = 0.0
    #: rotor declaration, passed straight onto the body, or None
    rotor: dict | None = None
    material: str = "cast-iron"


@dataclass(frozen=True)
class Compartment:
    """A plant box above or below the chamber, with its loadout."""
    height_m: float
    plant: tuple = ()
    label: str = "plant"


@dataclass(frozen=True)
class Window:
    """What is in the door to see through, and what is fitted to it."""
    kind: str = "panes"            # panes (mullioned, full door) | porthole | none
    porthole_radius_m: float = 0.15
    #: accessories through the window: glove-ports so far
    accessories: tuple = ()


@dataclass(frozen=True)
class Door:
    kind: str = "glazed"           # glazed | solid
    hinge: str = "piano-hinge"
    dogs: int = 2
    window: Window = Window()


@dataclass(frozen=True)
class DrumDrive:
    """The motor a drum is specced with, because it is part of the drum.

    A mixer, an agitator, a washer's tub: the motor is not a separate
    thing that happens to be near the drum, it is sized for the drum's
    inertia and load and it goes where the drum's mount puts it.

        direct   the motor is bolted to the drum's rear bearing and
                 moves with it -- on a suspended tub, its mass rides
                 the suspension too, which is why direct-drive washers
                 carry big counterweights
        belt     the motor is bolted to the deck under the drum and a
                 belt couples them: the belt has a real span stiffness
                 and is declared as one, so a suspended tub stays
                 suspended in the walk rather than being welded to the
                 floor through its own belt

    Its rotor is declared like any other: rated speed, a shape, a mass,
    a balance grade."""
    kind: str = "belt"                 # belt | direct
    motor_kw: float = 0.5
    rated_rpm: float = 2800.0
    mass_kg: float = 8.0
    rotor_mass_frac: float = 0.35
    rotor_radius_m: float = 0.04
    balance_grade_mm_s: float = 6.3
    belt_stiffness_n_per_m: float = 20_000.0
    #: where on the deck a belt motor sits, as fractions of half extents
    at: tuple = (0.6, 0.3)


@dataclass(frozen=True)
class Drum:
    """A horizontal drum behind the door: the washer's and the dryer's."""
    radius_m: float
    length_m: float
    mass_kg: float
    rated_rpm: float
    #: suspended (springs and dampers, a washer) | rollers (a dryer)
    mount: str = "suspended"
    suspension_n_per_m: float = 8000.0
    suspension_damping: float = 0.25
    balance_grade_mm_s: float = 6.3
    runs_in: tuple = ("*",)
    load_kg: float = 8.0
    material: str = "stainless-sheet"
    #: the motor it is specced with; None is a drum something else turns
    drive: DrumDrive | None = None


@dataclass(frozen=True)
class CabinetSpec:
    identity: str
    inside_m: tuple                         # (W, H, D)
    working_pressure_pa: float = 0.0
    construction: str = "welded-plate"      # welded-plate | screwed-sheet
    sheet_thickness_m: float = 0.0008
    stiffener_pitch_m: float = 0.40
    racks: tuple = ()
    top: Compartment | None = None
    bottom: Compartment | None = None
    door: Door = Door()
    drum: Drum | None = None
    feet_per_side: int = 2
    #: extra wall ports the machine wants: {wall: {name: (face, across)}}
    wall_ports: dict = field(default_factory=dict)
    label: str = "cabinet"

    @property
    def is_pressure(self) -> bool:
        return self.working_pressure_pa > 0.0 and self.construction == "welded-plate"

    def wall_t(self) -> float:
        if self.construction == "welded-plate":
            return wall_thickness_m(self.working_pressure_pa, self.stiffener_pitch_m)
        return self.sheet_thickness_m

    def glass_t(self, span_m: float) -> float:
        return glass_thickness_m(self.working_pressure_pa, span_m)


# ---------------------------------------------------------------------
# what a build hands back to the machine module
# ---------------------------------------------------------------------

@dataclass
class Built:
    """The graph plus every named place on it, for the extras hook."""
    graph: ProductionGraph
    spec: CabinetSpec
    wall_nodes: dict                # wall -> {port: node identity}
    skid_nodes: dict
    compartment_nodes: dict         # "top"/"bottom" -> {port: node}
    plant_nodes: dict               # plant item name -> {port: node}
    door_nodes: dict
    drum_nodes: dict
    t: float

    def wn(self, wall: str, port: str) -> str:
        return self.wall_nodes[wall][port]

    def wall(self, name: str) -> Prism:
        return self._walls[name]

    def port(self, wall: str, name: str, closure: str, bore_m: float, **kw) -> str:
        """A service port through a wall: the nozzle node, welded (or
        screwed) to the wall's wrench point."""
        g = self.graph
        pos = next(n["reference_position"] for n in g.nodes
                   if n["identity"] == self.wn(wall, name))
        ident = f"{self.spec.identity}.port.{name}"
        g.node(ident, pos, "vessel-port", port_kind=name, closure=closure,
               bore_m=bore_m, in_view="plant", **kw)
        g.edge(f"{ident}.seat", self.wn(wall, name), ident,
               _seam_constraint(self.spec), radius=0.004, part_role="port-seat",
               **_seam_kwargs(self.spec))
        return ident

    def tag(self, prefixes, claim: str) -> None:
        """Mark every member under the prefixes as belonging in the claim."""
        for e in self.graph.edges:
            if any(e["identity"].startswith(px) for px in prefixes):
                e.setdefault("clears", claim)

    def claim(self, c: Claim, owner: str) -> None:
        self.graph.clear_volumes.append(c.as_clear_volume(owner))


def _seam_constraint(spec: CabinetSpec) -> str:
    return "shell-attachment-weld" if spec.construction == "welded-plate" else "screwed-seam"


def _seam_kwargs(spec: CabinetSpec) -> dict:
    if spec.construction == "welded-plate":
        return {}
    seam = Seam(fastener="sheet-metal-screw", per_inch=SHEET_SEAM.per_inch,
                sheet_thickness_m=spec.sheet_thickness_m,
                fastener_diameter_m=SHEET_SEAM.fastener_diameter_m)
    return {"seam": seam, "section": SheetSection(0.05, spec.sheet_thickness_m)}


def _rib_mass(span_m: float, run_m: float, pitch_m: float) -> float:
    count = max(1, int(round(run_m / pitch_m)) - 1)
    return count * span_m * RIB_SECTION.mass_per_m_kg


def rack_claim(spec: CabinetSpec, i: int, r: Rack) -> Claim:
    W, H, D = spec.inside_m
    h = r.half_extent_m or (W / 2.0 - 0.05, 0.015, D / 2.0 - 0.05)
    return Claim(identity=f"{spec.identity}.rack_{i}",
                 min=(-h[0], r.y_m - 0.02, -h[2]),
                 max=(h[0], min(r.y_m + r.headroom_m, H - 0.02), h[2]),
                 note=f"rack level {i} and the headroom above it")


# ---------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------

def _box_walls(g, spec, prefix, centre_y, height, t, extra_ports, *, is_chamber):
    """Five (or six) walls of a box, welded or screwed along their edges.

    Returns {wall: (Prism, {port: node})}. A chamber has no front wall
    -- the door is the front; a compartment has one."""
    W, _, D = spec.inside_m
    hw, hd = W / 2.0, D / 2.0
    hh = height / 2.0
    walls = {
        "floor":   ((0.0, centre_y - hh - t / 2.0, 0.0), (hw + t, t / 2.0, hd + t), (W, D)),
        "ceiling": ((0.0, centre_y + hh + t / 2.0, 0.0), (hw + t, t / 2.0, hd + t), (W, D)),
        "back":    ((0.0, centre_y, -hd - t / 2.0), (hw + t, hh, t / 2.0), (W, height)),
        "left":    ((-hw - t / 2.0, centre_y, 0.0), (t / 2.0, hh, hd), (D, height)),
        "right":   ((+hw + t / 2.0, centre_y, 0.0), (t / 2.0, hh, hd), (D, height)),
    }
    if not is_chamber:
        walls["front"] = ((0.0, centre_y, +hd + t / 2.0), (hw + t, hh, t / 2.0), (W, height))
    faces = {"floor": (W + 2 * t) * (D + 2 * t), "ceiling": (W + 2 * t) * (D + 2 * t),
             "back": (W + 2 * t) * height, "front": (W + 2 * t) * height,
             "left": D * height, "right": D * height}
    out = {}
    for name, (c, half, (span, run)) in walls.items():
        mass = faces[name] * t * STEEL_DENSITY_KG_M3
        if spec.construction == "welded-plate" and spec.working_pressure_pa > 0.0:
            mass += _rib_mass(span, run, spec.stiffener_pitch_m)
        kind = "pressure-wall" if (is_chamber and spec.is_pressure) else "cabinet-panel"
        wall = Prism(identity=f"{prefix}.wall.{name}", kind=kind, centre=c, shape="box",
                     half_extent=half,
                     material="boiler-plate" if spec.construction == "welded-plate" else "steel-sheet",
                     mass_kg=round(mass, 2),
                     attributes={"in_view": "plant", "plate_thickness_m": t,
                                 "construction": spec.construction,
                                 "mass_is_derived": True,
                                 **({"rib_section": RIB_SECTION.designation,
                                     "rib_pitch_m": spec.stiffener_pitch_m,
                                     "working_pressure_pa": spec.working_pressure_pa}
                                    if kind == "pressure-wall" else {})})
        ports = {"floor": {"e_back": ("-z", (0, 0)), "e_left": ("-x", (0, 0)),
                           "e_right": ("+x", (0, 0)), "e_front": ("+z", (0, 0))},
                 "ceiling": {"e_back": ("-z", (0, 0)), "e_left": ("-x", (0, 0)),
                             "e_right": ("+x", (0, 0)), "e_front": ("+z", (0, 0))},
                 "back": {"e_floor": ("-y", (0, 0)), "e_ceiling": ("+y", (0, 0)),
                          "e_left": ("-x", (0, 0)), "e_right": ("+x", (0, 0))},
                 "front": {"e_floor": ("-y", (0, 0)), "e_ceiling": ("+y", (0, 0)),
                           "e_left": ("-x", (0, 0)), "e_right": ("+x", (0, 0))},
                 "left": {"e_floor": ("-y", (0, 0)), "e_ceiling": ("+y", (0, 0)),
                          "e_back": ("-z", (0, 0)), "e_front": ("+z", (0, 0))},
                 "right": {"e_floor": ("-y", (0, 0)), "e_ceiling": ("+y", (0, 0)),
                           "e_back": ("-z", (0, 0)), "e_front": ("+z", (0, 0))}}[name]
        ports = dict(ports)
        ports.update(extra_ports.get(name, {}))
        pts = {pn: wall.face_point(face, across=across) for pn, (face, across) in ports.items()}
        out[name] = (wall, emit_prism(g, wall, pts, assembly=spec.label))
    seams = [("floor", "e_back", "back", "e_floor"), ("floor", "e_left", "left", "e_floor"),
             ("floor", "e_right", "right", "e_floor"),
             ("ceiling", "e_back", "back", "e_ceiling"), ("ceiling", "e_left", "left", "e_ceiling"),
             ("ceiling", "e_right", "right", "e_ceiling"),
             ("back", "e_left", "left", "e_back"), ("back", "e_right", "right", "e_back")]
    if not is_chamber:
        seams += [("floor", "e_front", "front", "e_floor"), ("ceiling", "e_front", "front", "e_ceiling"),
                  ("front", "e_left", "left", "e_front"), ("front", "e_right", "right", "e_front")]
    for a, ap_, b, bp in seams:
        g.edge(f"{prefix}.seam.{a}_{b}", out[a][1][ap_], out[b][1][bp],
               _seam_constraint(spec), radius=max(t / 2.0, 0.003), part_role="wall-seam",
               load_path="the-case-is-the-frame", **_seam_kwargs(spec))
    return out


def build(spec: CabinetSpec, *, extras=None) -> Built:
    """The box, its door, its racks, its compartments, its drum, its skid.

    `extras(built)` is the machine's own hook, run last, with every
    named place on the graph available by name."""
    g = ProductionGraph(identity=spec.identity)
    g.assembly = spec.label
    ident = spec.identity
    W, H, D = spec.inside_m
    hw, hh, hd = W / 2.0, H / 2.0, D / 2.0
    t = spec.wall_t()
    light = spec.construction == "screwed-sheet"
    door_section = LIGHT_DOOR_SECTION if light else DOOR_SECTION
    skid_section = LIGHT_SKID_SECTION if light else SKID_SECTION
    n_side = max(2, int(spec.feet_per_side))
    foot_fracs = [(-1.0 + 2.0 * i / (n_side - 1)) * 0.85 for i in range(n_side)]

    # ---- the chamber's own ports ---------------------------------------
    chamber_ports = {w: dict(p) for w, p in spec.wall_ports.items()}
    chamber_ports.setdefault("left", {}).update(
        {"hinge_top": ("+z", (0.9, 0.0)), "hinge_bottom": ("+z", (-0.9, 0.0))})
    chamber_ports.setdefault("right", {}).update(
        {f"dog_{i}": ("+z", ((-1.0 + 2.0 * (i + 0.5) / spec.door.dogs) * 0.9, 0.0))
         for i in range(spec.door.dogs)})
    for i, r in enumerate(spec.racks):
        chamber_ports["left"][f"rail_{i}"] = ("+x", ((r.y_m - hh) / hh, 0.0))
        chamber_ports["right"][f"rail_{i}"] = ("-x", ((r.y_m - hh) / hh, 0.0))
    chamber_ports.setdefault("floor", {}).update(
        {f"skid_{i}_{j}": ("-y", (fx, fz)) for i, fx in enumerate((-0.85, 0.85))
         for j, fz in enumerate(foot_fracs)})
    if spec.drum is not None:
        chamber_ports.setdefault("back", {})["drum_bearing"] = ("+z", (0.0, 0.0))
        chamber_ports.setdefault("ceiling", {}).update(
            {"drum_spring_l": ("-y", (-0.6, 0.0)), "drum_spring_r": ("-y", (0.6, 0.0))})
        chamber_ports["floor"].update(
            {"drum_damper_l": ("+y", (-0.6, 0.3)), "drum_damper_r": ("+y", (0.6, 0.3))})
    if spec.top is not None:
        chamber_ports.setdefault("ceiling", {}).update(
            {f"top_{k}": ("+y", fr) for k, fr in
             (("a", (-0.85, -0.85)), ("b", (0.85, -0.85)), ("c", (0.85, 0.85)), ("d", (-0.85, 0.85)))})
    if spec.bottom is not None:
        chamber_ports["floor"].update(
            {f"bottom_{k}": ("-y", fr) for k, fr in
             (("a", (-0.85, -0.85)), ("b", (0.85, -0.85)), ("c", (0.85, 0.85)), ("d", (-0.85, 0.85)))})

    walls = _box_walls(g, spec, ident, hh, H, t, chamber_ports, is_chamber=True)
    wall_nodes = {k: v[1] for k, v in walls.items()}

    def wn(wall, port):
        return wall_nodes[wall][port]

    # ---- compartments above and below ------------------------------------
    comp_nodes, plant_nodes = {}, {}
    lowest_y = -t                       # underside of the chamber floor
    for where, comp in (("top", spec.top), ("bottom", spec.bottom)):
        if comp is None:
            continue
        ct = t if not light else spec.sheet_thickness_m
        if where == "top":
            cy = H + t + ct + comp.height_m / 2.0
        else:
            cy = -t - ct - comp.height_m / 2.0
            lowest_y = cy - comp.height_m / 2.0 - ct
        # PLANT STANDS ON THE COMPARTMENT'S OWN FLOOR, in either box. A top
        # box joins the chamber by its floor's underside, a bottom box by
        # its ceiling's top; the plant deck is the floor in both.
        corners = (("a", (-0.85, -0.85)), ("b", (0.85, -0.85)),
                   ("c", (0.85, 0.85)), ("d", (-0.85, 0.85)))
        deck_ports = {"floor": {f"deck_{p.name}": ("+y", tuple(p.at)) for p in comp.plant}}
        if where == "top":
            deck_ports["floor"].update({f"join_{k}": ("-y", fr) for k, fr in corners})
        else:
            deck_ports["ceiling"] = {f"join_{k}": ("+y", fr) for k, fr in corners}
        if where == "bottom":
            deck_ports["floor"].update(
                {f"skid_{i}_{j}": ("-y", (fx, fz)) for i, fx in enumerate((-0.85, 0.85))
                 for j, fz in enumerate(foot_fracs)})
        cw = _box_walls(g, spec, f"{ident}.{where}", cy, comp.height_m, ct, deck_ports,
                        is_chamber=False)
        comp_nodes[where] = {w: v[1] for w, v in cw.items()}
        for k in "abcd":
            a = wn("ceiling", f"top_{k}") if where == "top" else wn("floor", f"bottom_{k}")
            b = comp_nodes[where]["floor" if where == "top" else "ceiling"][f"join_{k}"]
            g.edge(f"{ident}.{where}.join.{k}", a, b, _seam_constraint(spec),
                   radius=max(ct / 2.0, 0.003), part_role="compartment-seam",
                   **_seam_kwargs(spec))
        # the compartment's interior is a claimed volume, occupied by its plant
        fl = comp_nodes[where]["floor"]
        floor_y = cy - comp.height_m / 2.0
        g.clear_volumes.append(Claim(identity=f"{ident}.{where}_bay",
                                     min=(-hw, floor_y, -hd), max=(hw, floor_y + comp.height_m, hd),
                                     note=f"the {where} {comp.label} compartment").as_clear_volume(
            f"{ident}.{where}"))
        for p in comp.plant:
            attrs = {"in_view": "plant", "compartment": where, **(p.rotor or {})}
            body = Prism(identity=f"{ident}.{where}.{p.name}", kind=p.kind,
                         centre=(p.at[0] * hw, floor_y + p.half_extent_m[1] + 0.01,
                                 p.at[1] * hd),
                         shape="box", half_extent=tuple(p.half_extent_m),
                         material=p.material, mass_kg=p.mass_kg, attributes=attrs)
            pn = emit_prism(g, body, {"base": body.face_point("-y")}, assembly=spec.label)
            plant_nodes[p.name] = pn
            deck = fl[f"deck_{p.name}"]
            if p.mount == "isolated":
                g.edge(f"{ident}.{where}.{p.name}.mount", pn["base"], deck,
                       "engine-mount-isolator", radius=0.010, part_role="plant-isolator")
            elif p.mount == "suspended":
                g.edge(f"{ident}.{where}.{p.name}.mount", pn["base"], deck, "spring-damper",
                       radius=0.008, part_role="plant-suspension",
                       suspension={"linear_stiffness_n_per_m": p.suspension_n_per_m})
            else:
                g.edge(f"{ident}.{where}.{p.name}.mount", pn["base"], deck,
                       "bolted-flange-mount", radius=0.008, part_role="plant-feet")
        for e in g.edges:
            if e["identity"].startswith(f"{ident}.{where}.") and ".wall." not in e["identity"] \
                    and ".seam." not in e["identity"] and ".join." not in e["identity"]:
                e.setdefault("clears", f"{ident}.{where}_bay")

    # ---- the door ----------------------------------------------------------
    door = spec.door
    d_prefix = f"{ident}.door"
    d_nodes = {}
    door_t = door_section.side_m
    if door.kind == "glazed":
        mullions = max(1, int(round(W / spec.stiffener_pitch_m)) - 1)
        frame_len = 2.0 * (W + H) + mullions * H
        mass = frame_len * door_section.mass_per_m_kg
    else:
        mullions = 0
        mass = (W + 2 * door_t) * (H + 2 * door_t) * t * STEEL_DENSITY_KG_M3 \
            + 2.0 * (W + H) * door_section.mass_per_m_kg
    dprism = Prism(identity=d_prefix, kind="glazed-door" if door.kind == "glazed" else "solid-door",
                   centre=(0.0, hh, hd + door_t / 2.0), shape="box",
                   half_extent=(hw + door_t, hh + door_t, door_t / 2.0),
                   material="steel-plate", mass_kg=round(mass, 2),
                   attributes={"in_view": "plant", "frame_section": door_section.designation,
                               "mullions": mullions, "hinge": door.hinge, "mass_is_derived": True})
    d_ports = {"hinge_top": dprism.face_point("-x", across=(0.9, 0.0)),
               "hinge_bottom": dprism.face_point("-x", across=(-0.9, 0.0)),
               "seal": dprism.face_point("-z"),
               **{f"dog_{i}": dprism.face_point("+x", across=((-1.0 + 2.0 * (i + 0.5) / door.dogs) * 0.9, 0.0))
                  for i in range(door.dogs)}}
    if door.kind == "glazed":
        d_ports.update({f"pane_{i}": dprism.face_point("-z", across=(((i + 0.5) / (mullions + 1)) * 2.0 - 1.0, 0.0))
                        for i in range(mullions + 1)})
    if door.window.kind == "porthole":
        d_ports["porthole"] = dprism.face_point("-z", across=(0.0, 0.1))
    d_nodes = emit_prism(g, dprism, d_ports, assembly=spec.label)
    # THE HINGE CARRIES THE DOOR OPEN. A piano hinge turns about its own
    # line and nothing else; shut, the dogs take the pressure.
    for side in ("hinge_top", "hinge_bottom"):
        g.edge(f"{d_prefix}.{side}", d_nodes[side], wn("left", side), door.hinge,
               radius=0.010, part_role="door-hinge")
    for i in range(door.dogs):
        g.edge(f"{d_prefix}.dog_{i}", d_nodes[f"dog_{i}"], wn("right", f"dog_{i}"),
               "bolted-flange-mount", radius=0.012, part_role="door-dog")
    g.node(f"{ident}.port.door_face", d_ports["seal"].position, "vessel-port",
           port_kind="door_face", closure=f"{door.kind}-clamped-door", bore_m=W,
           normal=(0.0, 0.0, 1.0), mating=True, in_view="plant")
    g.edge(f"{d_prefix}.seal", d_nodes["seal"], f"{ident}.port.door_face",
           "port-face-seal", radius=0.006, part_role="door-gasket")
    for wall in ("left", "right"):
        g.edge(f"{ident}.door_ring.{wall}", f"{ident}.port.door_face", wn(wall, "e_front"),
               "rigid-distance", radius=0.010, part_role="door-opening-flange")
    glass_nodes = {}
    if door.kind == "glazed":
        pane_w = W / (mullions + 1)
        tg = spec.glass_t(pane_w)
        for i in range(mullions + 1):
            cx = (i + 0.5) * pane_w - hw
            pane = Prism(identity=f"{ident}.pane.{i}", kind="pressure-window",
                         centre=(cx, hh, hd + tg / 2.0), shape="box",
                         half_extent=(pane_w / 2.0, hh, tg / 2.0), material="laminated-glass",
                         mass_kg=round(pane_w * H * tg * GLASS_DENSITY_KG_M3, 2),
                         attributes={"in_view": "plant", "glass_thickness_m": tg,
                                     "span_m": pane_w, "mass_is_derived": True})
            pn = emit_prism(g, pane, {"retainer": pane.face_point("+z")}, assembly=spec.label)
            glass_nodes[i] = pn
            g.edge(f"{ident}.pane.{i}.retainer", pn["retainer"], d_nodes[f"pane_{i}"],
                   "bolted-flange-mount", radius=0.006, part_role="glazing-retainer")
    elif door.window.kind == "porthole":
        r = door.window.porthole_radius_m
        tg = spec.glass_t(2.0 * r)
        pane = Prism(identity=f"{ident}.pane.porthole", kind="pressure-window",
                     centre=(0.0, hh + 0.1 * hh, hd + door_t / 2.0), shape="cylinder",
                     axis=(0.0, 0.0, 1.0), radius=r, length=tg, material="laminated-glass",
                     mass_kg=round(math.pi * r * r * tg * GLASS_DENSITY_KG_M3, 2),
                     attributes={"in_view": "plant", "glass_thickness_m": tg,
                                 "span_m": 2.0 * r, "mass_is_derived": True})
        pn = emit_prism(g, pane, {"retainer": pane.end_point("+")}, assembly=spec.label)
        glass_nodes["porthole"] = pn
        g.edge(f"{ident}.pane.porthole.retainer", pn["retainer"], d_nodes["porthole"],
               "bolted-flange-mount", radius=0.006, part_role="glazing-retainer")
    if "glove-ports" in door.window.accessories:
        host = (glass_nodes[mullions // 2]["retainer"] if door.kind == "glazed"
                else d_nodes["seal"])
        for side, x_off in (("left", -0.12), ("right", +0.12)):
            g.node(f"{ident}.port.glove_{side}", (x_off, 0.7 * H, hd + door_t),
                   "vessel-port", port_kind=f"glove_{side}", closure="glove-sleeve",
                   bore_m=0.20, normal=(0.0, 0.0, 1.0), in_view="plant")
            g.edge(f"{ident}.port.glove_{side}.seat", host, f"{ident}.port.glove_{side}",
                   "bolted-flange-mount", radius=0.004, part_role="glove-port-ring")

    # ---- racks -----------------------------------------------------------------
    for i, r in enumerate(spec.racks):
        for side in ("left", "right"):
            pos = walls[side][0].face_point("+x" if side == "left" else "-x",
                                            across=((r.y_m - hh) / hh, 0.0)).position
            g.node(f"{ident}.rail.{side}.{i}", pos, "chamber-rail", in_view="plant",
                   on_inner_surface=True)
            g.edge(f"{ident}.rail.{side}.{i}.weld", wn(side, f"rail_{i}"),
                   f"{ident}.rail.{side}.{i}", _seam_constraint(spec), radius=0.006,
                   part_role="rail-foot", **_seam_kwargs(spec))
        h = r.half_extent_m or (hw - 0.05, 0.015, hd - 0.05)
        rack = Prism(identity=f"{ident}.rack.{i}", kind="load-rack", centre=(0.0, r.y_m, 0.0),
                     shape="box", half_extent=tuple(h), material=r.material, mass_kg=r.mass_kg,
                     attributes={"in_view": "plant", "inside": True, "level": i})
        rn = emit_prism(g, rack, {"rail_l": rack.face_point("-x"), "rail_r": rack.face_point("+x")},
                        assembly=spec.label)
        for side, port in (("left", "rail_l"), ("right", "rail_r")):
            g.edge(f"{ident}.rack.{i}.mount_{side[0]}", rn[port], f"{ident}.rail.{side}.{i}",
                   "bolted-flange-mount", radius=0.008, part_role="rack-mount")
        claim = rack_claim(spec, i, r)
        g.clear_volumes.append(claim.as_clear_volume(f"{ident}.racks"))
        for e in g.edges:
            if e["identity"].startswith(f"{ident}.rack.{i}.") or \
                    re.match(rf"{re.escape(ident)}\.rail\.(left|right)\.{i}\.", e["identity"]):
                e.setdefault("clears", claim.identity)

    # ---- the drum behind the door ------------------------------------------------
    drum_nodes = {}
    if spec.drum is not None:
        dr = spec.drum
        centre = (0.0, hh, hd - 0.05 - dr.length_m / 2.0)
        drum = Prism(identity=f"{ident}.drum", kind="rotating-drum", centre=centre,
                     shape="cylinder", axis=(0.0, 0.0, 1.0), radius=dr.radius_m,
                     length=dr.length_m, material=dr.material, mass_kg=dr.mass_kg,
                     attributes={"in_view": "plant", "inside": True,
                                 "rotor_axis": (0.0, 0.0, 1.0), "rotor_rated_rpm": dr.rated_rpm,
                                 "rotor_shape": "thin-ring", "rotor_mass_kg": dr.mass_kg,
                                 "rotor_radius_m": dr.radius_m,
                                 "balance_grade_mm_s": dr.balance_grade_mm_s,
                                 "runs_in": tuple(dr.runs_in), "drum_mount": dr.mount})
        ports = {"bearing": drum.end_point("-"),
                 "spring_l": drum.side_point(math.radians(300.0)),
                 "spring_r": drum.side_point(math.radians(60.0)),
                 "damper_l": drum.side_point(math.radians(210.0)),
                 "damper_r": drum.side_point(math.radians(150.0))}
        dn = emit_prism(g, drum, ports, assembly=spec.label)
        drum_nodes = dn
        if dr.mount == "suspended":
            # THE TUB HANGS ON SPRINGS AND STANDS ON DAMPERS. That is the
            # whole reason a washer does not walk across the floor at a
            # thousand rpm with a wet towel on one side, and it is a
            # declared stiffness, not a bushing.
            for side in ("l", "r"):
                g.edge(f"{ident}.drum.spring_{side}", dn[f"spring_{side}"],
                       wn("ceiling", f"drum_spring_{side}"), "spring-damper", radius=0.006,
                       part_role="tub-spring",
                       suspension={"linear_stiffness_n_per_m": dr.suspension_n_per_m,
                                   "damping_ratio": dr.suspension_damping})
                g.edge(f"{ident}.drum.damper_{side}", dn[f"damper_{side}"],
                       wn("floor", f"drum_damper_{side}"), "spring-damper", radius=0.006,
                       part_role="tub-damper",
                       suspension={"linear_stiffness_n_per_m": dr.suspension_n_per_m * 0.5,
                                   "damping_ratio": dr.suspension_damping})
        else:
            # on rollers and a rear bearing: bolted, because a dryer's drum
            # is light and slow and its cabinet is the bearing housing
            g.edge(f"{ident}.drum.bearing", dn["bearing"], wn("back", "drum_bearing"),
                   "bolted-flange-mount", radius=0.010, part_role="drum-rear-bearing")
            for side in ("l", "r"):
                g.edge(f"{ident}.drum.roller_{side}", dn[f"damper_{side}"],
                       wn("floor", f"drum_damper_{side}"), "bolted-flange-mount",
                       radius=0.006, part_role="drum-roller")
        if dr.drive is not None:
            dv = dr.drive
            rotor = {"rotor_axis": (0.0, 0.0, 1.0), "rotor_rated_rpm": dv.rated_rpm,
                     "rotor_shape": "solid-disc", "rotor_mass_kg": dv.mass_kg * dv.rotor_mass_frac,
                     "rotor_radius_m": dv.rotor_radius_m,
                     "balance_grade_mm_s": dv.balance_grade_mm_s, "runs_in": tuple(dr.runs_in)}
            if dv.kind == "direct":
                # on the back of the tub, moving with it
                mc = (centre[0], centre[1], centre[2] - dr.length_m / 2.0 - 0.06)
                motor = Prism(identity=f"{ident}.drum.motor", kind="drum-motor", centre=mc,
                              shape="cylinder", axis=(0.0, 0.0, 1.0), radius=0.10, length=0.10,
                              material="cast-iron", mass_kg=dv.mass_kg,
                              attributes={"in_view": "plant", "drive": "direct", **rotor})
                mn = emit_prism(g, motor, {"flange": motor.end_point("+")}, assembly=spec.label)
                g.edge(f"{ident}.drum.motor.flange", mn["flange"], dn["bearing"],
                       "bolted-flange-mount", radius=0.010, part_role="direct-drive-flange")
            else:
                # on the deck under the drum, coupled by a belt of real stiffness
                deck_wall = wn("floor", "drum_damper_r")
                mc = (dv.at[0] * hw, 0.10, dv.at[1] * hd)
                motor = Prism(identity=f"{ident}.drum.motor", kind="drum-motor", centre=mc,
                              shape="cylinder", axis=(0.0, 0.0, 1.0), radius=0.07, length=0.16,
                              material="cast-iron", mass_kg=dv.mass_kg,
                              attributes={"in_view": "plant", "drive": "belt", **rotor})
                mn = emit_prism(g, motor, {"base": motor.side_point(math.radians(180.0)),
                                           "pulley": motor.end_point("-")}, assembly=spec.label)
                g.edge(f"{ident}.drum.motor.mount", mn["base"], deck_wall,
                       "bolted-flange-mount", radius=0.008, part_role="motor-feet")
                g.edge(f"{ident}.drum.belt", mn["pulley"], dn["bearing"], "spring-damper",
                       radius=0.004, part_role="drive-belt",
                       suspension={"linear_stiffness_n_per_m": dv.belt_stiffness_n_per_m,
                                   "damping_ratio": 0.05})
            drum_nodes["motor"] = mn
        # a suspended tub swings and wants room; a drum on rollers does
        # not, and a dryer's drum very nearly fills its cabinet
        swing = 0.03 if dr.mount == "suspended" else 0.005
        claim = Claim(identity=f"{ident}.drum_bay",
                      min=(-dr.radius_m - swing, hh - dr.radius_m - swing,
                           centre[2] - dr.length_m / 2.0 - 0.02),
                      max=(dr.radius_m + swing, hh + dr.radius_m + swing, hd),
                      note="the drum and the room it needs to swing")
        g.clear_volumes.append(claim.as_clear_volume(f"{ident}.drum"))
        for e in g.edges:
            if e["identity"].startswith(f"{ident}.drum"):
                e.setdefault("clears", claim.identity)

    # ---- the skid and the feet --------------------------------------------------------
    skid_h = 0.15 if not light else 0.06
    skid = Prism(identity=f"{ident}.skid", kind="machine-frame",
                 centre=(0.0, lowest_y - skid_h / 2.0, 0.0), shape="box",
                 half_extent=(hw + t, skid_h / 2.0, hd + t), material=skid_section.material,
                 mass_kg=round((4.0 * (W + D) + (n_side - 2) * D) * skid_section.mass_per_m_kg, 2),
                 attributes={"in_view": "plant", "section": skid_section.designation,
                             "mass_is_derived": True})
    s_ports = {}
    for i, fx in enumerate((-0.85, 0.85)):
        for j, fz in enumerate(foot_fracs):
            s_ports[f"top_{i}_{j}"] = skid.face_point("+y", across=(fx, fz))
            s_ports[f"foot_{i}_{j}"] = skid.face_point("-y", across=(fx, fz))
    s_ports["side_mount"] = skid.face_point("+x", across=(0.0, -0.5))
    s_nodes = emit_prism(g, skid, s_ports, assembly=spec.label,
                         port_kwargs={"bolt_radius_m": 0.012 if not light else 0.006})
    under = comp_nodes["bottom"]["floor"] if spec.bottom is not None else wall_nodes["floor"]
    for i in range(2):
        for j in range(n_side):
            node = next(n for n in g.nodes if n["identity"] == s_nodes[f"foot_{i}_{j}"])
            node.update(port_role="structural-mount", joint="bolted-flange", outward=(0.0, -1.0, 0.0))
            g.edge(f"{ident}.skid.weld_{i}_{j}", s_nodes[f"top_{i}_{j}"], under[f"skid_{i}_{j}"],
                   _seam_constraint(spec), radius=0.008, part_role="skid-weld", **_seam_kwargs(spec))

    built = Built(graph=g, spec=spec, wall_nodes=wall_nodes, skid_nodes=s_nodes,
                  compartment_nodes=comp_nodes, plant_nodes=plant_nodes,
                  door_nodes=d_nodes, drum_nodes=drum_nodes, t=t)
    built._walls = {k: v[0] for k, v in walls.items()}
    if extras is not None:
        extras(built)
    return built


def charges(spec: CabinetSpec, *, racks_loaded: bool = True, drum_loaded: bool = True) -> dict:
    """What the cabinet carries when loaded, per body, for a cycle state."""
    out = {}
    if racks_loaded:
        for i, r in enumerate(spec.racks):
            out[f"{spec.identity}.rack.{i}"] = r.load_kg
    if drum_loaded and spec.drum is not None:
        out[f"{spec.identity}.drum"] = spec.drum.load_kg
    return out


def describe(built: Built) -> str:
    spec = built.spec
    doc = built.graph.as_document()
    W, H, D = spec.inside_m
    total = sum(float(n.get("mass_kg", 0.0)) for n in doc["nodes"] if not n.get("wrench_point"))
    walls = sum(float(n["mass_kg"]) for n in doc["nodes"]
                if n.get("kind") in ("pressure-wall", "cabinet-panel"))
    glass = sum(float(n["mass_kg"]) for n in doc["nodes"] if n.get("kind") == "pressure-window")
    feet = sum(1 for n in doc["nodes"] if n.get("port_role") == "structural-mount")
    lines = [f"{spec.identity}: {W:.2f} x {H:.2f} x {D:.2f} m inside, {spec.construction}"
             + (f", {spec.working_pressure_pa / 1e5:.1f} bar" if spec.working_pressure_pa else ""),
             f"  walls {built.t * 1000:.1f} mm: {walls:.0f} kg; glass {glass:.0f} kg; "
             f"{len(spec.racks)} racks; door {spec.door.kind} on a {spec.door.hinge}"
             + (f", {spec.door.window.kind} window" if spec.door.window.kind != "none" else "")]
    for where, comp in (("top", spec.top), ("bottom", spec.bottom)):
        if comp is not None:
            lines.append(f"  {where} {comp.label}: {comp.height_m * 1000:.0f} mm, "
                         f"{', '.join(p.name for p in comp.plant) or 'empty'}")
    if spec.drum is not None:
        lines.append(f"  drum r {spec.drum.radius_m:.2f} m x {spec.drum.length_m:.2f} m, "
                     f"{spec.drum.rated_rpm:.0f} rpm, {spec.drum.mount}")
    lines.append(f"  {total:.0f} kg all told on {feet} feet")
    return "\n".join(lines)
