"""Per-KIND cylinder port layouts and cylinder meshes -- the physical
head-end of every cylinder this toy can run, laid out from the same
cylinder sites engine_geometry already places and the same working-
fluid / network declarations the conversion layer already reads.

A cylinder is one of four real kinds here, and each has a real,
different set of holes in it:

  spark-piston        intake valve port, exhaust valve port, spark plug
                      boss; a port-injector boss (liquid EFI or a gas
                      injector) or a direct-injector boss; a carburetor
                      or mixer feeds the intake port upstream.
  compression-piston  intake, exhaust, a DIRECT injector boss dead
                      centre (or in the pre-chamber) and a glow-plug
                      boss beside it -- no plug.
  atmospheric         the Otto-Langen free piston: an OPEN-TOP bore (the
                      piston flies free up it; the rack rides in guides
                      above), and at the bottom the slide valve's own
                      three real openings -- a gas port, an air port, and
                      the flame-transfer port the outside pilot flame is
                      carried through -- plus the exhaust port. No head.
  expander            a cutoff expander: a valve chest on the side of the
                      bore with an admission and an exhaust passage to
                      the HEAD end, the same pair to the CRANK end when
                      double-acting (plus a rod gland in the crank-end
                      cover), a drain cock at each working end, and a
                      lubricator boss on the chest.

Every port is placed on the cylinder's own real geometry: on the head
(bore axis end), on the side wall at a stated height along the stroke,
or on the valve chest -- radius from the real bore (a valve throat is
a real fraction of bore; a plug/injector boss is a small fixed hole),
and pointing along a real outward direction. Meshes are built from the
same tube/cuboid generators every other part uses (mesh_primitives),
tagged with the per-cylinder thermal group ("block_cyl_N") the live
renderer already colours by block temperature.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mesh_primitives import tube_mesh, cuboid_mesh, capped_tube_mesh, prism_mesh

Vec3 = tuple[float, float, float]

SPARK_PISTON = "spark-piston"
COMPRESSION_PISTON = "compression-piston"
ATMOSPHERIC = "atmospheric"
EXPANDER = "expander"
ROTARY = "rotary"

# Real Wankel proportions (the 13B's: R = 105 mm generating radius,
# e = 15 mm eccentricity, 80 mm rotor width, 654 cc per rotor) -- a
# rotor housing is scaled off these by displacement per rotor, since
# the chamber volume goes as R * e * width and all three scale
# together on a geometrically similar rotor.
WANKEL_R_M, WANKEL_E_M, WANKEL_WIDTH_M, WANKEL_CC_PER_ROTOR = 0.105, 0.015, 0.080, 654.0

# real, disclosed proportions
INTAKE_VALVE_FRAC_OF_BORE = 0.45        # a four-valve head's intake throat pair, as one equivalent hole
EXHAUST_VALVE_FRAC_OF_BORE = 0.38
PLUG_BOSS_RADIUS_M = 0.007              # M14 plug
INJECTOR_BOSS_RADIUS_M = 0.008
GLOW_PLUG_RADIUS_M = 0.005
DRAIN_COCK_RADIUS_M = 0.006
GAS_PORT_FRAC_OF_BORE = 0.12            # Otto-Langen slide-valve gas opening
AIR_PORT_FRAC_OF_BORE = 0.30
FLAME_PORT_RADIUS_M = 0.006
EXPANDER_PASSAGE_FRAC_OF_BORE = 0.22
WALL_FRAC_OF_BORE = 0.10                # cylinder wall thickness as a fraction of bore


@dataclass(frozen=True)
class PortSpec:
    name: str                 # "intake_port", "spark_plug", "head_admission", ...
    port_kind: str            # what real hole this is
    position: Vec3            # absolute, in the drivetrain graph frame
    direction: Vec3           # outward unit vector the port stub points along
    radius_m: float
    fluid_role: str           # "intake" | "exhaust" | "ignition" | "fuel" | "admission" | "drain" | "lubrication" | "rod"


@dataclass(frozen=True)
class CylinderGeometry:
    number: int
    kind: str
    base: Vec3                # crank-end centre of the bore
    axis: Vec3                # unit vector, crank end -> head end
    bore_m: float
    length_m: float           # crank-end cover to head-end cover
    double_acting: bool = False
    open_top: bool = False
    # the running gear this bore drives: the crank centre this bore's
    # axis passes through, its throw radius (half stroke), the rod
    # length, and the throw's own phase on the crank -- what places the
    # piston, rod and crankpin for any crank angle
    crank_centre: Vec3 = (0.0, 0.0, 0.0)
    crank_radius_m: float = 0.0
    rod_length_m: float = 0.0
    throw_angle_deg: float = 0.0
    crosshead: bool = False   # double-acting expander: piston rod through a gland to a crosshead, then the con rod
    # passive cooling: a finned outer wall (air-cooled) or a plain
    # water-jacketed one
    finned: bool = False
    # two-stroke wall porting: no poppet valves at all (transfer +
    # exhaust ports in the wall), or a uniflow (wall scavenge ports,
    # exhaust valves in the head)
    wall_ports: str = "none"  # "none" | "loop" | "uniflow"
    valves_per_cylinder: int = 2
    # where the cam lives (head_mesh.py): "pushrod" | "sohc" | "dohc" |
    # "none" -- derived until the catalogue declares it (see
    # derive_valvetrain)
    valvetrain: str = "pushrod"
    # how the outside of the cylinder sheds heat: "fins" (air), "jacket"
    # (pumped water), "hopper" (an open water hopper on a slow stationary
    # engine: jacket, no pump, water boils off and is topped up), "none"
    cooling: str = "jacket"
    # a hit-and-miss / stationary single carries a flywheel on BOTH ends
    twin_flywheels: bool = False
    # rotary only: eccentricity (the rotor's orbit radius); crank_radius_m
    # is unused, the eccentric shaft is the crank
    eccentricity_m: float = 0.0


def _unit(v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


def _perp(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = _unit(np.cross(axis, helper))
    v = _unit(np.cross(axis, u))
    return u, v


def cylinder_kind_of(engine) -> str:
    if engine.kind == "expander":
        return EXPANDER
    if getattr(engine.architecture, "rotary", False):
        return ROTARY
    if engine.kind == "atmospheric":
        return ATMOSPHERIC
    if getattr(engine, "compression_ignition", False):
        return COMPRESSION_PISTON
    return SPARK_PISTON


def _admission_kind(engine) -> str:
    net = getattr(engine, "fuel_network", None)
    if net is not None:
        return net.admission.kind
    if getattr(engine, "compression_ignition", False):
        return "liquid-injector"
    carb = getattr(engine, "carburetor", None)
    if carb is not None and carb.is_carbureted:
        return "liquid-carburetor"
    return "liquid-injector"


def is_air_cooled(engine) -> bool:
    """No water pump on the engine and no separate coolant pump: the
    block sheds heat straight to air, which is what fins are for."""
    acc = engine.accessories
    return not acc.water_pump and not getattr(acc, "coolant_pump", False)


def throw_angles_deg(engine) -> dict[int, float]:
    """Each cylinder's crank-throw phase from the declared firing order:
    evenly spaced firing over one cycle (720 deg four-stroke, 360 two-
    stroke), the throw sitting where the piston is at TDC as that
    cylinder fires -- the same evenly-spaced firing engine_cycle_sim
    already runs. A bank angle shows up in the sites, not here."""
    arch = engine.architecture
    order = list(arch.firing_order) if arch.firing_order else list(range(1, max(1, arch.cylinders) + 1))
    cycle = 360.0 if arch.two_stroke else 720.0
    step = cycle / max(len(order), 1)
    return {cyl: (slot * step) % 360.0 for slot, cyl in enumerate(order)}


def derive_valvetrain(engine) -> str:
    """Disclosed stand-in until EngineArchitecture declares it: 3+ valves
    per cylinder -> twin cams; a two-valve head revving past 7000 ->
    single overhead cam; any other two-valve head -> cam in the block
    with pushrods (which is what every catalogue engine whose production
    graph puts `powertrain.camshaft` beside the crank actually is). A
    two-stroke with no poppet valves has none."""
    arch = engine.architecture
    if getattr(arch, "valvetrain", None):
        return arch.valvetrain
    if arch.two_stroke and not arch.has_poppet_valves:
        return "none"
    valves = int(getattr(engine.lifter_spring, "valves_per_cylinder", 2) or 2)
    if valves >= 3:
        return "dohc"
    if engine.redline_rpm >= 7000.0:
        return "sohc"
    return "pushrod"


def cylinder_geometries(engine) -> list[CylinderGeometry]:
    """Real bore/stroke geometry for every cylinder of any kind, plus
    the running gear behind it. Piston engines use engine_geometry's
    own sites (the bore axis runs from the crank centre out through
    the site; the bore's foot sits where the piston pin is at bottom
    dead centre); the Otto-Langen and expander kinds, which carry
    cylinders=0 in the architecture, lay their bores out from their
    own specs."""
    import engine_geometry
    kind = cylinder_kind_of(engine)
    arch = engine.architecture
    out: list[CylinderGeometry] = []
    finned = is_air_cooled(engine)
    if kind == ATMOSPHERIC:
        spec = engine.atmospheric
        out.append(CylinderGeometry(number=1, kind=kind, base=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0),
                                    bore_m=spec.bore_m, length_m=spec.stroke_m + spec.ignition_height_m,
                                    open_top=True, finned=False, crank_centre=(0.0, spec.stroke_m + spec.ignition_height_m
                                                                              + spec.pinion_radius_m * 1.5, 0.0),
                                    crank_radius_m=spec.pinion_radius_m))
        return out
    if kind == EXPANDER:
        spec = engine.expander
        bore, stroke = spec.bore_m, spec.stroke_m
        pitch = bore * 1.6
        n = max(1, spec.cylinders)
        rod = stroke * 2.2
        r = stroke / 2.0
        # the crank sits below the crank-end cover by the rod (and, if
        # double-acting, the piston rod + crosshead) at its longest
        gap = rod + r + (stroke * 0.9 if spec.double_acting else 0.0)
        for i in range(n):
            x = (i - (n - 1) / 2.0) * pitch
            base_y = gap + bore * 0.15
            out.append(CylinderGeometry(number=i + 1, kind=kind, base=(x, base_y, 0.0), axis=(0.0, 1.0, 0.0),
                                        bore_m=bore, length_m=stroke * 1.12, double_acting=spec.double_acting,
                                        crank_centre=(x, 0.0, 0.0), crank_radius_m=r, rod_length_m=rod,
                                        throw_angle_deg=(i * 360.0 / n) % 360.0, crosshead=spec.double_acting,
                                        finned=False))
        return out
    if kind == ROTARY:
        n = max(1, arch.cylinders)
        scale = (engine.displacement_l * 1000.0 / n / WANKEL_CC_PER_ROTOR) ** (1.0 / 3.0)
        R, e, w = WANKEL_R_M * scale, WANKEL_E_M * scale, WANKEL_WIDTH_M * scale
        pitch = w * 1.35
        for i in range(n):
            x = (i - (n - 1) / 2.0) * pitch
            # the housing "bore" stands in as the trochoid's outer radius;
            # the geometry is a drum along the eccentric shaft (x), so
            # base/axis describe the housing's axial extent
            out.append(CylinderGeometry(number=i + 1, kind=kind, base=(x - w / 2.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
                                        bore_m=2.0 * (R + e), length_m=w, crank_centre=(x, 0.0, 0.0),
                                        crank_radius_m=0.0, rod_length_m=0.0, eccentricity_m=e,
                                        throw_angle_deg=(180.0 * i) % 360.0, finned=False, cooling="jacket",
                                        valvetrain="none"))
        return out
    sites = engine_geometry.cylinder_sites(engine)
    bore = arch.bore_m if arch.bore_m > 0.0 else 0.08
    stroke = arch.stroke_m if arch.stroke_m > 0.0 else 0.08
    rod = arch.rod_length_m if arch.rod_length_m > 0.0 else stroke * 1.75
    r = stroke / 2.0
    throws = throw_angles_deg(engine)
    if arch.two_stroke:
        wall_ports = "uniflow" if arch.has_poppet_valves else "loop"
    else:
        wall_ports = "none"
    valves = int(getattr(engine.lifter_spring, "valves_per_cylinder", 2) or 2)
    valvetrain = derive_valvetrain(engine)
    layout_name = arch.layout.lower()
    hit_and_miss = "hit-and-miss" in layout_name
    cooling = "hopper" if hit_and_miss else ("fins" if finned else "jacket")
    for site in sites:
        pos = np.array(site.position, dtype=np.float64)
        radial = pos.copy(); radial[0] = 0.0
        axis = _unit(radial) if np.linalg.norm(radial) > 1e-9 else np.array([0.0, 1.0, 0.0])
        crank_c = np.array([pos[0], 0.0, 0.0])
        # the bore's foot: just below the piston skirt at BDC
        base = crank_c + axis * (rod - r - bore * 0.35)
        out.append(CylinderGeometry(number=site.number, kind=kind, base=tuple(base), axis=tuple(axis),
                                    bore_m=bore, length_m=stroke + bore * 0.75,
                                    crank_centre=tuple(crank_c), crank_radius_m=r, rod_length_m=rod,
                                    throw_angle_deg=throws.get(site.number, 0.0), finned=finned,
                                    wall_ports=wall_ports, valves_per_cylinder=valves, valvetrain=valvetrain,
                                    cooling=cooling, twin_flywheels=hit_and_miss))
    return out


def cylinder_port_layout(engine) -> list[tuple[CylinderGeometry, list[PortSpec]]]:
    """Every real port of every cylinder, by kind (see module docstring)."""
    kind = cylinder_kind_of(engine)
    admission = _admission_kind(engine)
    result = []
    for g in cylinder_geometries(engine):
        axis = _unit(g.axis)
        u, v = _perp(axis)
        base = np.array(g.base)
        # a tilted bank (V/W) breathes in from the valley and out
        # outboard: point +u (the intake side) toward the engine's
        # centre plane; an upright inline bank keeps intake at -z
        if abs(float(base[2])) > 1e-6 and float(u[2]) * float(base[2]) > 0.0:
            u = -u
        head = base + axis * g.length_m
        r = g.bore_m / 2.0
        wall = g.bore_m * WALL_FRAC_OF_BORE
        ports: list[PortSpec] = []

        def P(name, port_kind, pos, direction, radius, role):
            ports.append(PortSpec(name=name, port_kind=port_kind, position=tuple(float(x) for x in pos),
                                  direction=tuple(float(x) for x in _unit(direction)), radius_m=radius, fluid_role=role))

        if kind == ROTARY:
            # a Wankel housing: side intake ports in the end plate, a
            # peripheral exhaust port in the rim, leading and trailing
            # plugs in the rim ahead of it -- the 13B's own arrangement
            R = g.bore_m / 2.0
            xc = np.array(g.crank_centre)
            rim = lambda ang: xc + np.array([0.0, math.cos(ang), math.sin(ang)]) * R
            rim_dir = lambda ang: np.array([0.0, math.cos(ang), math.sin(ang)])
            front = xc - np.array([g.length_m / 2.0 + wall, 0.0, 0.0])
            P("intake_port", "side-intake-port", front + np.array([0.0, -R * 0.55, -R * 0.35]), (-1.0, 0.0, 0.0),
              R * 0.17, "intake")
            P("intake_port_2", "side-intake-port", front + np.array([0.0, -R * 0.55, R * 0.35]), (-1.0, 0.0, 0.0),
              R * 0.17, "intake")
            P("exhaust_port", "peripheral-exhaust-port", rim(math.radians(200.0)), rim_dir(math.radians(200.0)),
              R * 0.16, "exhaust")
            P("leading_plug", "spark-plug-boss", rim(math.radians(-20.0)), rim_dir(math.radians(-20.0)),
              PLUG_BOSS_RADIUS_M, "ignition")
            P("trailing_plug", "spark-plug-boss", rim(math.radians(10.0)), rim_dir(math.radians(10.0)),
              PLUG_BOSS_RADIUS_M, "ignition")
        elif kind in (SPARK_PISTON, COMPRESSION_PISTON):
            n_valves = max(1, g.valves_per_cylinder)
            n_in = (n_valves + 1) // 2 if n_valves > 1 else 1
            n_ex = n_valves - n_in if n_valves > 1 else 1
            if g.wall_ports == "loop":
                # a loop-scavenged two-stroke: no valves at all -- two
                # transfer ports low in the wall on the intake side, one
                # exhaust port opposite, uncovered by the piston itself
                port_h = base + axis * (g.length_m * 0.30)
                for i, side in enumerate((-1.0, 1.0)):
                    P(f"transfer_port_{i + 1}" if i else "intake_port", "transfer-port",
                      port_h + u * (r + wall) + v * (side * r * 0.45), u, r * 0.22, "intake")
                P("exhaust_port", "exhaust-wall-port", port_h + axis * (g.length_m * 0.06) - u * (r + wall),
                  -u, r * 0.32, "exhaust")
                n_in = 0
            elif g.wall_ports == "uniflow":
                # uniflow: a ring of scavenge ports low in the wall, the
                # exhaust through valves in the head
                port_h = base + axis * (g.length_m * 0.22)
                for i in range(4):
                    ang = math.radians(45.0 + 90.0 * i)
                    dirn = u * math.cos(ang) + v * math.sin(ang)
                    P(f"scavenge_port_{i + 1}" if i else "intake_port", "scavenge-port",
                      port_h + dirn * (r + wall), dirn, r * 0.16, "intake")
                n_in = 0
                n_ex = n_valves
            # poppet valves in the head, canted to either side of the axis
            for i in range(n_in):
                off = (i - (n_in - 1) / 2.0) * (r * 0.9 / max(n_in, 1))
                P("intake_port" if i == 0 else f"intake_port_{i + 1}", "intake-valve-port",
                  head + u * (r * 0.45) + v * off, u * 0.35 + axis,
                  r * INTAKE_VALVE_FRAC_OF_BORE / math.sqrt(n_in), "intake")
            if g.wall_ports != "loop":
                for i in range(n_ex):
                    off = (i - (n_ex - 1) / 2.0) * (r * 0.9 / max(n_ex, 1))
                    P("exhaust_port" if i == 0 else f"exhaust_port_{i + 1}", "exhaust-valve-port",
                      head - u * (r * 0.45) + v * off, -u * 0.35 + axis,
                      r * EXHAUST_VALVE_FRAC_OF_BORE / math.sqrt(n_ex), "exhaust")
            if kind == SPARK_PISTON:
                P("spark_plug", "spark-plug-boss", head + v * (r * 0.15), axis, PLUG_BOSS_RADIUS_M, "ignition")
                if admission in ("liquid-injector", "gas-injector"):
                    # a port injector sits in the intake tract just upstream of the valve
                    P("port_injector", "gas-injector-boss" if admission == "gas-injector" else "port-injector-boss",
                      head + u * (r * 0.9) + axis * (wall * 2.0), u, INJECTOR_BOSS_RADIUS_M, "fuel")
                # a carburetor/mixer feeds the intake port from the runner: no boss on the cylinder itself
            else:
                P("direct_injector", "direct-injector-boss", head, axis, INJECTOR_BOSS_RADIUS_M, "fuel")
                P("glow_plug", "glow-plug-boss", head - v * (r * 0.3), axis, GLOW_PLUG_RADIUS_M, "ignition")
        elif kind == ATMOSPHERIC:
            # slide valve at the bottom of the open bore: gas, air and the flame-transfer port side by side
            foot = base + axis * (wall * 1.5)
            P("gas_port", "slide-valve-gas-port", foot + u * (r + wall), u, r * GAS_PORT_FRAC_OF_BORE, "fuel")
            P("air_port", "slide-valve-air-port", foot + u * (r + wall) + v * (r * 0.45), u,
              r * AIR_PORT_FRAC_OF_BORE, "intake")
            P("flame_port", "flame-transfer-port", foot + u * (r + wall) - v * (r * 0.45), u,
              FLAME_PORT_RADIUS_M, "ignition")
            P("exhaust_port", "slide-valve-exhaust-port", foot - u * (r + wall), -u, r * 0.35, "exhaust")
        else:  # EXPANDER
            chest_offset = u * (r + wall + g.bore_m * 0.35)
            head_end = head - axis * (wall * 1.5)
            crank_end = base + axis * (wall * 1.5)
            P("head_admission", "expander-admission-passage", head_end + chest_offset, u,
              r * EXPANDER_PASSAGE_FRAC_OF_BORE, "admission")
            P("head_exhaust", "expander-exhaust-passage", head_end + chest_offset + v * (r * 0.5), v,
              r * EXPANDER_PASSAGE_FRAC_OF_BORE, "exhaust")
            P("head_drain_cock", "drain-cock", head_end - v * (r + wall), -v, DRAIN_COCK_RADIUS_M, "drain")
            if g.double_acting:
                P("crank_admission", "expander-admission-passage", crank_end + chest_offset, u,
                  r * EXPANDER_PASSAGE_FRAC_OF_BORE, "admission")
                P("crank_exhaust", "expander-exhaust-passage", crank_end + chest_offset + v * (r * 0.5), v,
                  r * EXPANDER_PASSAGE_FRAC_OF_BORE, "exhaust")
                P("crank_drain_cock", "drain-cock", crank_end - v * (r + wall), -v, DRAIN_COCK_RADIUS_M, "drain")
                P("rod_gland", "piston-rod-gland", base, -axis, r * 0.18, "rod")
            P("lubricator", "lubricator-boss", (head_end + crank_end) / 2.0 + chest_offset + axis * 0.0 - v * (r * 0.5),
              -v, GLOW_PLUG_RADIUS_M, "lubrication")
        result.append((g, ports))
    return result


# ---------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------

def _tube(a, b, radius, sides=16):
    return capped_tube_mesh(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), radius, sides=sides)


def serialize_layout(layout) -> list[dict]:
    """The layout as plain dicts -- what drivetrain_graph stores on the
    graph document (graph["cylinder_layout"]) so vehicle_mesh can build
    the cylinder bodies from the graph alone, the way it builds
    everything else."""
    import dataclasses
    return [{"geometry": dataclasses.asdict(g), "ports": [dataclasses.asdict(p) for p in ports]}
            for g, ports in layout]


def deserialize_layout(data: list[dict]):
    return [(CylinderGeometry(**d["geometry"]), [PortSpec(**pd) for pd in d["ports"]]) for d in data]


def build_cylinder_solid_parts(engine, piston_travel_frac: float | dict[int, float] = 0.5) -> list:
    """SolidParts (vehicle_mesh.SolidPart) for every cylinder: the bore
    wall, the head-end cover (none for an open-top atmospheric bore),
    the piston at its travel, per-kind extras (valve chest, rod gland,
    rack guide), and a stub tube for every real port from
    cylinder_port_layout. Tagged block_cyl_N so the live renderer's
    per-cylinder block temperatures colour them."""
    return build_parts_from_layout(cylinder_port_layout(engine), piston_travel_frac)


FIN_PITCH_FRAC_OF_BORE = 0.09
FIN_DEPTH_FRAC_OF_BORE = 0.22
FIN_THICKNESS_FRAC_OF_BORE = 0.025
JACKET_FRAC_OF_BORE = 0.12


def piston_pin_distance_m(crank_radius_m: float, rod_length_m: float, crank_angle_deg: float) -> float:
    """Slider-crank: pin distance from the crank centre along the bore
    axis at this crank angle (0 = TDC)."""
    th = math.radians(crank_angle_deg)
    r, l = crank_radius_m, max(rod_length_m, crank_radius_m + 1e-6)
    return r * math.cos(th) + math.sqrt(max(l * l - (r * math.sin(th)) ** 2, 0.0))


def build_parts_from_layout(layout, piston_travel_frac: float | dict[int, float] = 0.5,
                            crank_angle_deg: float = 0.0, covers_off: bool = False, engine=None,
                            moving_only: bool = False, spring_style: str = "helix") -> list:
    """`crank_angle_deg` is the crank's own angle (the sim's state.
    crank_angle_deg); each cylinder's piston, rod and crankpin follow
    from it through that cylinder's own throw phase. piston_travel_frac
    is only used for kinds with no crank (the free piston)."""
    from vehicle_mesh import SolidPart
    parts: list = []
    for g, ports in layout:
        axis = _unit(g.axis)
        u, v = _perp(axis)
        base = np.array(g.base)
        head = base + axis * g.length_m
        r = g.bore_m / 2.0
        wall = g.bore_m * WALL_FRAC_OF_BORE
        grp = f"block_cyl_{g.number}"
        tag = f"cyl{g.number}"
        if g.kind == ROTARY:
            # housing drum, end plate, eccentric shaft, and the rotor as a
            # triangular prism riding the eccentric at a third of the
            # shaft's angle (the real 3:1 phasing)
            R = g.bore_m / 2.0; e = g.eccentricity_m; w = g.length_m
            xc = np.array(g.crank_centre)
            ax = np.array([1.0, 0.0, 0.0])
            if not moving_only:
                vtx, nrm = tube_mesh(xc - ax * (w / 2.0), xc + ax * (w / 2.0), R + wall, sides=28)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_rotor_housing"))
                vtx, nrm = _tube(xc - ax * (w / 2.0 + wall), xc - ax * (w / 2.0), R + wall * 1.4, sides=28)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_side_plate"))
            shaft_ang = math.radians(crank_angle_deg + g.throw_angle_deg)
            ecc = xc + np.array([0.0, math.cos(shaft_ang), math.sin(shaft_ang)]) * e
            rot_ang = shaft_ang / 3.0
            tri = [((R - e) * math.cos(rot_ang + k * 2.0 * math.pi / 3.0), (R - e) * math.sin(rot_ang + k * 2.0 * math.pi / 3.0))
                   for k in range(3)]
            vtx, nrm = prism_mesh(tri, ecc, ax, w * 0.46, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_rotor"))
            vtx, nrm = _tube(xc - ax * (w * 0.9), xc + ax * (w * 0.9), e * 1.1, sides=12)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_eccentric_shaft"))
            if not moving_only:
                for p in ports:
                    d = np.array(p.direction); pos = np.array(p.position)
                    length = max(0.02, p.radius_m * 2.5)
                    vtx, nrm = _tube(pos - d * (length * 0.3), pos + d * length, p.radius_m, sides=10)
                    grp_p = {"intake": "intake", "exhaust": "exhaust"}.get(p.fluid_role, grp)
                    parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp_p, name=f"{tag}_{p.name}"))
            continue
        static_parts: list = []
        _append = static_parts.append
        # bore wall (the liner)
        vtx, nrm = tube_mesh(base, head, r + wall, sides=20)
        _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_bore"))
        # outer wall: a stack of cooling fins (air-cooled) or a plain
        # water jacket -- the same real casting choice, passive vs pumped
        if g.finned:
            pitch = g.bore_m * FIN_PITCH_FRAC_OF_BORE
            thick = g.bore_m * FIN_THICKNESS_FRAC_OF_BORE
            n_fins = max(2, int((g.length_m - wall) / pitch))
            for i in range(n_fins):
                p_fin = base + axis * (wall * 0.5 + i * pitch)
                vtx, nrm = _tube(p_fin, p_fin + axis * thick, r + wall + g.bore_m * FIN_DEPTH_FRAC_OF_BORE, sides=24)
                _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_fin_{i + 1}"))
        elif g.kind in (SPARK_PISTON, COMPRESSION_PISTON):
            vtx, nrm = _tube(base, head - axis * (wall * 0.5), r + wall + g.bore_m * JACKET_FRAC_OF_BORE, sides=20)
            _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_water_jacket"))
            if g.cooling == "hopper":
                # an open hopper on top of the jacket: the water sits in
                # it, boils, and gets topped up -- no pump, no radiator
                u_, v_ = _perp(axis)
                hc = head - axis * (g.length_m * 0.15)
                vtx, nrm = cuboid_mesh(hc, np.abs(axis) * (g.length_m * 0.30) + np.abs(u_) * (r + wall * 3.5) + np.abs(v_) * (r + wall * 3.5))
                _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_water_hopper"))
        # crank-end cover (a crankcase deck for piston kinds, a real bottom cover for an expander)
        vtx, nrm = _tube(base - axis * wall, base, r + wall * 1.6, sides=20)
        _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_crank_cover"))
        if not g.open_top:
            vtx, nrm = _tube(head, head + axis * (wall * 1.5), r + wall * 1.6, sides=20)
            _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_head"))
        # piston, rod and crank
        crank_c = np.array(g.crank_centre)
        if g.crank_radius_m > 0.0 and g.rod_length_m > 0.0 and g.kind != ATMOSPHERIC:
            theta = crank_angle_deg + g.throw_angle_deg
            th = math.radians(theta)
            # the throw turns in the plane of the bore axis and its
            # in-plane perpendicular v (the crank runs along u = x)
            crank_axis = np.array([1.0, 0.0, 0.0])
            side = _unit(np.cross(crank_axis, axis))
            pin = crank_c + axis * (g.crank_radius_m * math.cos(th)) + side * (g.crank_radius_m * math.sin(th))
            d = piston_pin_distance_m(g.crank_radius_m, g.rod_length_m, theta)
            if g.crosshead:
                # double-acting: piston rod through the gland to a
                # crosshead, then the con rod down to the crankpin
                stroke_len = g.crank_radius_m * 2.0
                xhead = crank_c + axis * (d + 0.0)
                piston_c = base + axis * (wall + (d - (g.rod_length_m - g.crank_radius_m)) )
                vtx, nrm = _tube(xhead - axis * (g.bore_m * 0.12), xhead + axis * (g.bore_m * 0.12), r * 0.28, sides=10)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_crosshead"))
                vtx, nrm = _tube(xhead, piston_c, r * 0.16, sides=10)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_piston_rod"))
                vtx, nrm = _tube(pin, xhead, r * 0.12, sides=10)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_con_rod"))
                p0 = piston_c
            else:
                p0 = crank_c + axis * d
                vtx, nrm = _tube(pin, p0, r * 0.14, sides=10)
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_con_rod"))
            vtx, nrm = _tube(p0 - axis * (r * 0.35), p0 + axis * (r * 0.45), r * 0.96, sides=20)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_piston"))
        else:
            frac = piston_travel_frac.get(g.number, 0.5) if isinstance(piston_travel_frac, dict) else piston_travel_frac
            stroke_span = g.length_m - wall * 2.0
            p0 = base + axis * (wall + stroke_span * max(0.0, min(1.0, frac)) * 0.8)
            vtx, nrm = _tube(p0, p0 + axis * (r * 0.6), r * 0.96, sides=20)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_piston"))
        if g.kind == ATMOSPHERIC:
            # The Otto-Langen's "crank": the rack drives a pinion, and
            # the pinion turns the flywheel shaft ONLY through a real
            # freewheel (ratchet / overrunning clutch) -- engaged on the
            # power stroke (atmosphere driving the piston down), slipping
            # while the piston flies up -- so the flywheel keeps its
            # speed between strokes. That freewheel is exactly what
            # otto_langen.OttoLangenCylinder.step integrates; here it is
            # the drum between the pinion and the shaft. The shaft is
            # held in a bearing at each end of the column top, free to
            # rotate, and carries the big flywheel outboard.
            pin_c = np.array(g.crank_centre)
            ax = np.array([1.0, 0.0, 0.0])
            r_pin = g.crank_radius_m
            vtx, nrm = _tube(pin_c - ax * (g.bore_m * 0.12), pin_c + ax * (g.bore_m * 0.12), r_pin, sides=24)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_pinion"))
            vtx, nrm = _tube(pin_c + ax * (g.bore_m * 0.12), pin_c + ax * (g.bore_m * 0.40), r_pin * 1.35, sides=20)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_freewheel_drum"))
            shaft_a = pin_c - ax * (g.bore_m * 0.9)
            shaft_b = pin_c + ax * (g.bore_m * 1.9)
            vtx, nrm = _tube(shaft_a, shaft_b, r_pin * 0.35, sides=12)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_flywheel_shaft"))
            for k_b, bx in enumerate((pin_c - ax * (g.bore_m * 0.7), pin_c + ax * (g.bore_m * 1.0))):
                vtx, nrm = cuboid_mesh(bx - np.array([0.0, r_pin * 0.9, 0.0]),
                                       np.array([g.bore_m * 0.12, r_pin * 1.1, r_pin * 0.9]))
                parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_shaft_bearing_{k_b + 1}"))
            fly_c = pin_c + ax * (g.bore_m * 1.5)
            # the wheel is sized off the column, ~a third of its height
            # in radius -- the real Otto-Langen proportion in period
            # drawings, the same on the big and the workshop machine
            vtx, nrm = _tube(fly_c - ax * (g.bore_m * 0.12), fly_c + ax * (g.bore_m * 0.12), g.length_m * 0.33, sides=40)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name="flywheel"))
            # the rack rising out of the open bore, and its guide above
            vtx, nrm = _tube(p0 + axis * (r * 0.6), head + axis * (g.length_m * 0.4), r * 0.12, sides=8)
            parts.append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_rack"))
            vtx, nrm = cuboid_mesh(head + axis * (g.length_m * 0.25), np.array([r * 0.35, r * 0.35, r * 0.35]))
            _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=None, name=f"{tag}_rack_guide"))
        elif g.kind == EXPANDER:
            # the valve chest alongside the bore
            centre = (base + head) / 2.0 + u * (r + wall + g.bore_m * 0.35)
            half = np.abs(axis) * (g.length_m * 0.42) + np.abs(u) * (g.bore_m * 0.30) + np.abs(v) * (g.bore_m * 0.30)
            vtx, nrm = cuboid_mesh(centre, np.maximum(half, 0.01))
            _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp, name=f"{tag}_valve_chest"))
        # port stubs
        for p in ports:
            d = np.array(p.direction)
            pos = np.array(p.position)
            length = max(0.02, p.radius_m * 2.5)
            vtx, nrm = _tube(pos - d * (length * 0.3), pos + d * length, p.radius_m, sides=10)
            grp_p = {"intake": "intake", "exhaust": "exhaust", "admission": "intake"}.get(p.fluid_role, grp)
            _append(SolidPart(vertices=vtx, normals=nrm, thermal_group=grp_p, name=f"{tag}_{p.name}"))
        if not moving_only:
            parts.extend(static_parts)
    # the crank train and crankcase are ONE part each across the whole
    # layout (crank_mesh.py), not something any single cylinder owns
    from crank_mesh import build_crank_train_parts, build_crankshaft_parts
    from head_mesh import build_head_parts
    if moving_only:
        parts.extend(build_crankshaft_parts(layout, crank_angle_deg))
        if covers_off:
            from valvetrain_parts import build_valvetrain_parts
            parts.extend(build_valvetrain_parts(layout, engine, crank_angle_deg, spring_style=spring_style))
        return parts
    parts.extend(build_crank_train_parts(layout, crank_angle_deg))
    # heads, cam case, valve covers and the valley/top cover, per bank
    parts.extend(build_head_parts(layout, covers_off=covers_off, engine=engine, crank_angle_deg=crank_angle_deg,
                                  spring_style=spring_style))
    return parts


def port_layout_summary(engine) -> list[str]:
    out = [f"{engine.identity}: {cylinder_kind_of(engine)}"]
    for g, ports in cylinder_port_layout(engine):
        out.append(f"  cylinder {g.number}: bore {g.bore_m*1000:.0f} mm, length {g.length_m*1000:.0f} mm"
                   + (", double-acting" if g.double_acting else "") + (", open top" if g.open_top else ""))
        for p in ports:
            out.append(f"    {p.name:18s} {p.port_kind:28s} r={p.radius_m*1000:4.1f} mm  {p.fluid_role}")
    return out
