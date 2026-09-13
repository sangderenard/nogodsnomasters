"""The real modules a shaft gas turbine is actually made of.

WHY THIS EXISTS. A turbine in this catalogue was getting a PISTON
ENGINE'S package: `drivetrain_graph` builds the block geometry from
`displacement_l`, and a turbine's displacement is a placeholder because
its real cycle lives in `TurbineSpec`. So asking an AGT1500 how big it
was returned 0.89 x 0.66 x 0.79 m -- the size of a 12 litre V-block --
for an engine that is really about 1.6 m long and weighs 1134 kg. Every
decision made against that number, and the first one was "does it fit
in the bay", would have been wrong by a factor of four in volume.

A turbine is not a block with holes in it. It is a string of modules on
one axis, and the two that make it BULKY are the ones a piston engine
does not have at all:

  RECUPERATORS. The AGT1500 is a regenerated engine: two rotary heat
  exchangers take heat out of the exhaust and put it back into the air
  leaving the compressor. That is what lets a turbine of this class run
  a modest pressure ratio and still be worth fuelling. They are two
  large drums either side of the core and they are most of why the
  engine is as wide as it is. Delete them and you have a smaller engine
  that drinks a third more fuel.

  THE REDUCTION GEARBOX. The power turbine runs at tens of thousands of
  rpm and the output shaft has to turn at a few thousand. That gearbox
  is a heavy, oil-cooled lump and it is not optional.

Everything here is declared with its own real dimensions and its own
mass, so the package prism is measured from the parts rather than
inferred from a number that was never about this engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TurbineModule:
    """One module of the engine, with its real size and where it sits
    along the engine's own axis."""
    identity: str
    label: str
    role: str
    #: centre, in the engine's own frame: x along the axis, y up, z across
    position: tuple
    #: half extents of the box it actually occupies
    half_extent: tuple
    mass_kg: float
    material: str = "cast-iron"
    note: str = ""
    attributes: dict = field(default_factory=dict)


def agt1500_modules() -> list[TurbineModule]:
    """The AGT1500's own module set.

    Laid out along x, which is the engine's axis in this project's
    drivetrain frame. Masses sum to the catalogue's declared 1134 kg
    dry, and the extents are the real package: about 1.63 m long,
    1.00 m wide over the recuperators, 0.99 m tall.
    """
    return [
        TurbineModule(
            "turbine.inlet_plenum", "inlet plenum and barrier filter",
            "air-inlet", (-0.72, 0.10, 0.0), (0.10, 0.30, 0.34), 62.0,
            "pressed-steel",
            note="a turbine swallows 5.6 kg/s of air and it must be CLEAN "
                 "-- ingested sand is what actually kills these engines, "
                 "so the filter is large, serviceable and in the way",
            attributes=dict(air_flow_kg_s=5.6, filter_area_m2=4.2)),
        TurbineModule(
            "turbine.compressor", "axial-centrifugal compressor spool",
            "compressor", (-0.44, 0.10, 0.0), (0.18, 0.22, 0.22), 141.0,
            "steel-shaft",
            note="stages of axial blading feeding a centrifugal last "
                 "stage: the axial part makes the pressure efficiently "
                 "and the centrifugal stage makes the rest of it in a "
                 "short axial length, which is why nearly every engine "
                 "of this size is built this way",
            attributes=dict(pressure_ratio=5.5, stages=6)),
        TurbineModule(
            "turbine.recuperator.left", "rotary recuperator, left",
            "heat-exchanger", (-0.06, 0.16, -0.34), (0.30, 0.30, 0.16), 168.0,
            "hardened-steel",
            note="a ceramic matrix drum turning slowly through both gas "
                 "paths: it picks up heat in the exhaust and gives it "
                 "back to the compressor discharge. Most of the width "
                 "of this engine is these two drums",
            attributes=dict(effectiveness=0.72, matrix="ceramic")),
        TurbineModule(
            "turbine.recuperator.right", "rotary recuperator, right",
            "heat-exchanger", (-0.06, 0.16, 0.34), (0.30, 0.30, 0.16), 168.0,
            "hardened-steel",
            note="the other one",
            attributes=dict(effectiveness=0.72, matrix="ceramic")),
        TurbineModule(
            "turbine.combustor", "reverse-flow annular combustor",
            "combustor", (-0.04, 0.30, 0.0), (0.17, 0.17, 0.19), 74.0,
            "hardened-steel",
            note="one continuous flame. There is no compression stroke "
                 "and no octane requirement, which is the whole reason "
                 "this engine does not care what is in the drum",
            attributes=dict(exit_temperature_k=1400.0, reverse_flow=True)),
        TurbineModule(
            "turbine.gas_generator_turbine", "gas generator turbine",
            "turbine-stage", (0.20, 0.10, 0.0), (0.12, 0.20, 0.20), 96.0,
            "steel-shaft",
            note="takes exactly enough power out of the gas to drive the "
                 "compressor and nothing else -- everything left over "
                 "goes downstream to the power turbine",
            attributes=dict(stages=2, drives="turbine.compressor")),
        TurbineModule(
            "turbine.power_turbine", "free power turbine",
            "turbine-stage", (0.42, 0.10, 0.0), (0.11, 0.21, 0.21), 88.0,
            "steel-shaft",
            note="on its OWN shaft, mechanically free of the gas "
                 "generator. That is why a turbine can sit at zero "
                 "output shaft speed against full load without stalling "
                 "-- the core keeps spinning regardless",
            attributes=dict(stages=2, free_shaft=True)),
        TurbineModule(
            "turbine.reduction_gearbox", "reduction gearbox",
            "gearbox", (0.66, 0.02, 0.0), (0.15, 0.24, 0.24), 213.0,
            "cast-iron",
            note="tens of thousands of rpm down to the low thousands. "
                 "Heavy, oil-cooled, and a third of the dry weight of "
                 "the engine once you count its oil",
            attributes=dict(ratio=4.2)),
        TurbineModule(
            "turbine.exhaust_duct", "exhaust duct",
            "exhaust", (0.40, 0.52, 0.0), (0.26, 0.16, 0.26), 54.0,
            "pressed-steel",
            note="it leaves hot and it leaves in bulk -- 5.6 kg/s of it "
                 "-- and where that plume goes is a signature problem, "
                 "not just a packaging one",
            attributes=dict(flow_kg_s=5.6, exit_temperature_k=780.0)),
        TurbineModule(
            "turbine.oil_tank", "lubrication tank and cooler",
            "lubrication", (0.60, 0.42, -0.30), (0.13, 0.14, 0.11), 44.0,
            "pressed-steel",
            note="bearings at those speeds are fed, not splashed",
            attributes=dict(oil_l=26.0)),
        TurbineModule(
            "turbine.accessory_drive", "accessory gearbox",
            "accessory-drive", (0.60, 0.42, 0.30), (0.12, 0.14, 0.11), 26.0,
            "cast-iron",
            note="starter, fuel pump, oil pumps, alternator -- all of it "
                 "hangs off the gas generator shaft",
            attributes=dict(drives=("starter", "fuel-pump", "oil-pump",
                                    "alternator"))),
    ]


MODULE_SETS = {"agt1500-abrams-turbine": agt1500_modules}


def modules_for(engine) -> list[TurbineModule]:
    """The module set an engine DECLARES, by its own identity.

    Returns an empty list for a turbine that has not had its modules
    drawn yet, which is honest: the caller then knows it is looking at
    an engine whose real package is not described, rather than being
    handed a piston engine's block and told it is a turbine."""
    maker = MODULE_SETS.get(getattr(engine, "identity", None))
    return maker() if maker else []


def envelope(modules: list[TurbineModule]) -> dict:
    """The package the modules actually occupy."""
    if not modules:
        return {"size_m": (0.0, 0.0, 0.0), "volume_m3": 0.0, "mass_kg": 0.0}
    lo = [min(m.position[i] - m.half_extent[i] for m in modules) for i in range(3)]
    hi = [max(m.position[i] + m.half_extent[i] for m in modules) for i in range(3)]
    size = tuple(hi[i] - lo[i] for i in range(3))
    return {"min_corner": tuple(lo), "max_corner": tuple(hi), "size_m": size,
            "volume_m3": size[0] * size[1] * size[2],
            "mass_kg": sum(m.mass_kg for m in modules)}


def emit_modules(g, engine, *, at=(0.0, 0.0, 0.0), assembly: str = "powerplant",
                 motion_group: str = "frame", carried_by: list = ()) -> list[str]:
    """Write one turbine's real modules into a production graph."""
    mods = modules_for(engine)
    if not mods:
        raise KeyError(
            f"{getattr(engine, 'identity', engine)!r} has no declared turbine "
            f"modules -- add them to turbine_parts.MODULE_SETS rather than "
            f"letting it be packaged as a piston block")
    g.motion_group = motion_group
    g.assembly = assembly
    made = []
    for m in mods:
        ident = f"{m.identity}"
        g.node(ident,
               [at[0] + m.position[0], at[1] + m.position[1],
                at[2] + m.position[2]],
               "load-bearing-structure", material=m.material,
               mass_kg=m.mass_kg, mass_in_total=False,
               part_role=m.role, label=m.label,
               of_engine=getattr(engine, "identity", None),
               half_extent_m=tuple(float(v) for v in m.half_extent),
               **m.attributes)
        made.append(ident)
    # the modules are bolted to each other along the axis, in order
    order = sorted(mods, key=lambda m: m.position[0])
    for a, b in zip(order, order[1:]):
        g.edge(f"{a.identity}__{b.identity}", a.identity, b.identity,
               "rigid-distance", radius=0.030, palette="chassis-grey",
               load_path="turbine-module-flange")
    # and the whole string sits on its bearers
    for node_id in carried_by:
        for m in (order[0], order[len(order) // 2], order[-1]):
            g.edge(f"{m.identity}.bearer.{node_id.split('.')[-1]}",
                   m.identity, node_id, "rigid-distance", radius=0.024,
                   palette="chassis-grey",
                   load_path="engine-on-its-bearers")
    return made
