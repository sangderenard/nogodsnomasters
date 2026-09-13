"""Standard military ammunition containers, and the racks they live in.

AMMUNITION IS NOT A NUMBER. A magazine that declares `rounds_stowed=320`
has said nothing about whether those rounds fit, what they weigh, how
they are handled, or what comes out of the machine when they are gone.
Real ammunition arrives in standard containers, those containers are
the unit of handling, and the rack is built around the container --
not around the round.

So the containers are declared here as what they actually are, with
their real outside dimensions, because those dimensions are what
decides how many go in a bay:

  120 mm tank ammunition ships one round to a container: a sealed
  fibre-reinforced tube a little over a metre long. You carry the
  container, you break the seal at the gun, and the empty tube is
  rubbish you now have to store or dump. Two to a man-carry, and the
  loaded tube is over thirty kilos.

  .50 BMG is linked -- a RIBBON of belted cartridges -- and it ships
  coiled flat in an M2A1 steel can. That can is the oldest standardised
  box in the inventory and everything in the world is built to accept
  it: 305 x 155 x 188 mm, one hundred linked rounds, fifteen kilos.

A RACK IS BUILT FOR ONE CONTAINER. A rack that takes "ammunition"
takes nothing: the pitch between tubes is the tube's own diameter plus
its retention, the number of rows is the bay's height divided by that,
and mixing two container types in one rack means sizing for the larger
and wasting the difference. So a rack declares which container it is
for and derives everything else from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Container:
    """A standard container, by its OUTSIDE dimensions -- which is what
    a rack is built around. The round inside is an attribute of it."""
    key: str
    label: str
    shape: str                      # "tube" | "box"
    length_m: float                 # the long axis
    width_m: float
    height_m: float
    empty_kg: float
    loaded_kg: float
    rounds: int
    calibre_mm: float
    note: str = ""

    @property
    def payload_kg(self) -> float:
        return self.loaded_kg - self.empty_kg

    @property
    def round_kg(self) -> float:
        return self.payload_kg / max(self.rounds, 1)


CONTAINERS = {
    "120mm-single-round-tube": Container(
        "120mm-single-round-tube", "120 mm single-round shipping tube",
        "tube", 1.060, 0.240, 0.240, 5.4, 34.0, 1, 120.0,
        note="one round, sealed. The empty tube is bulky rubbish and has "
             "to go somewhere -- a gun that fires a hundred of these has "
             "a hundred metre-long tubes to get rid of"),
    "m2a1-50cal-can": Container(
        "m2a1-50cal-can", "M2A1 can, .50 BMG linked",
        "box", 0.305, 0.155, 0.188, 3.2, 15.4, 100, 12.7,
        note="one hundred rounds of linked belt -- a RIBBON -- coiled "
             "flat. The universal steel can: everything that handles "
             "ammunition is built to take this box"),
    "pa108-40mm-can": Container(
        "pa108-40mm-can", "PA108 can, 40 mm grenades",
        "box", 0.470, 0.300, 0.250, 6.0, 34.0, 32, 40.0,
        note="carried when the mount runs a 40 mm inner barrel"),
    "20mm-linked-can": Container(
        "20mm-linked-can", "20 mm linked can",
        "box", 0.560, 0.230, 0.280, 7.5, 48.0, 100, 20.0,
        note="linked 20 mm for the light inner barrel"),
}


@dataclass
class Rack:
    """A rack built for ONE container type, sized to a real volume.

    Given the space available it works out how many fit and in what
    arrangement, rather than being told a capacity that may or may not
    be physically possible."""
    identity: str
    container: Container
    centre: tuple
    span_m: tuple                   # (x, y, z) of the volume available
    axis: str = "z"                 # which way the containers point
    clearance_m: float = 0.018      # retention, guides, and getting a hand in
    rows: int = field(init=False)
    columns: int = field(init=False)
    deep: int = field(init=False)

    def __post_init__(self) -> None:
        c = self.container
        pitch = self._pitch()
        sx, sy, sz = self.span_m
        self.columns = max(1, int(sx / pitch[0]))
        self.rows = max(1, int(sy / pitch[1]))
        self.deep = max(1, int(sz / pitch[2]))

    def _pitch(self):
        c, g = self.container, self.clearance_m
        # the long axis lies along `axis`; the other two are the grid
        if self.axis == "z":
            return (c.width_m + g, c.height_m + g, c.length_m + g)
        if self.axis == "x":
            return (c.length_m + g, c.height_m + g, c.width_m + g)
        return (c.width_m + g, c.length_m + g, c.height_m + g)

    @property
    def capacity(self) -> int:
        return self.rows * self.columns * self.deep

    @property
    def rounds(self) -> int:
        return self.capacity * self.container.rounds

    @property
    def loaded_kg(self) -> float:
        return self.capacity * self.container.loaded_kg

    def positions(self):
        """Where every container in the rack actually sits."""
        px, py, pz = self._pitch()
        cx, cy, cz = self.centre
        for k in range(self.deep):
            for j in range(self.rows):
                for i in range(self.columns):
                    yield (i, j, k), (
                        cx + (i - (self.columns - 1) / 2.0) * px,
                        cy + (j - (self.rows - 1) / 2.0) * py,
                        cz + (k - (self.deep - 1) / 2.0) * pz)

    def half_extent(self):
        px, py, pz = self._pitch()
        return (self.columns * px / 2.0, self.rows * py / 2.0,
                self.deep * pz / 2.0)

    def describe(self) -> list[str]:
        c = self.container
        he = self.half_extent()
        return [
            f"{self.identity}: {c.label}",
            f"  {self.columns} x {self.rows} x {self.deep} = {self.capacity} "
            f"containers, {self.rounds} rounds",
            f"  {self.loaded_kg:.0f} kg loaded, "
            f"{self.capacity * c.empty_kg:.0f} kg of that is the containers",
            f"  rack envelope {he[0] * 2:.2f} x {he[1] * 2:.2f} x "
            f"{he[2] * 2:.2f} m",
        ]


def emit_rack(g, rack: Rack, *, carried_by: list[str],
              motion_group: str = "frame", assembly: str = "base-stores",
              palette: str = "chassis-grey") -> list[str]:
    """Write the rack frame and every container in it into the graph.

    The containers are real bodies with real mass, so the machine's
    weight includes its ammunition and the outrigger jacks are sized
    against the loaded vehicle rather than the empty one."""
    g.motion_group = motion_group
    g.assembly = assembly
    c = rack.container
    he = rack.half_extent()
    g.node(rack.identity, list(rack.centre), "load-bearing-structure",
           material="steel-plate", mass_in_total=False,
           mass_kg=round(rack.capacity * 1.9, 1),
           part_role="ammunition-rack",
           container_key=c.key, container_label=c.label,
           holds_containers=rack.capacity, holds_rounds=rack.rounds,
           calibre_mm=c.calibre_mm,
           loaded_mass_kg=round(rack.loaded_kg, 1),
           grid=(rack.columns, rack.rows, rack.deep),
           half_extent_m=tuple(round(v, 4) for v in he))
    made = [rack.identity]
    for node_id in carried_by:
        g.edge(f"{rack.identity}.mount.{node_id.split('.')[-1]}",
               rack.identity, node_id, "rigid-distance", radius=0.020,
               palette=palette, load_path="rack-carried-on-the-bay-floor")
    # the containers themselves
    if rack.axis == "z":
        hx, hy, hz = c.width_m / 2, c.height_m / 2, c.length_m / 2
    elif rack.axis == "x":
        hx, hy, hz = c.length_m / 2, c.height_m / 2, c.width_m / 2
    else:
        hx, hy, hz = c.width_m / 2, c.length_m / 2, c.height_m / 2
    for (i, j, k), pos in rack.positions():
        ident = f"{rack.identity}.{c.key}.{i}{j}{k}"
        g.node(ident, list(pos), "load-bearing-structure",
               material="pressed-steel" if c.shape == "box" else "nylon-airline",
               mass_in_total=False, mass_kg=round(c.loaded_kg, 1),
               part_role="ammunition-container",
               container_key=c.key, rounds=c.rounds, calibre_mm=c.calibre_mm,
               stowed_in=rack.identity,
               half_extent_m=(round(hx, 4), round(hy, 4), round(hz, 4)))
        made.append(ident)
        g.edge(f"{ident}.retained", ident, rack.identity, "rigid-distance",
               radius=0.010, palette=palette,
               load_path="container-retained-in-its-rack-cell")
    # CONTAINERS BEAR ON EACH OTHER. Packed in a rack they are not
    # hanging individually off a frame -- they are stacked against
    # their neighbours, and that is both what holds them still and why
    # a half-empty rack rattles.
    cells = {c: f"{rack.identity}.{rack.container.key}.{c[0]}{c[1]}{c[2]}"
             for c, _ in rack.positions()}
    for (i, j, k), ident in cells.items():
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nb = (i + di, j + dj, k + dk)
            if nb in cells:
                g.edge(f"{ident}.bears.{di}{dj}{dk}", ident, cells[nb],
                       "rigid-distance", radius=0.008, palette=palette,
                       load_path="container-bearing-on-the-next-one-in-the-rack")
    return made


def containers_for_bore(bore_mm: float) -> str:
    """Which container this mount's main gun eats, by calibre.

    Declared by matching the container's OWN calibre, not by reading a
    name: a rack is built for a container and a container is built for
    a round, so the round's calibre is the only question."""
    best = min(CONTAINERS.values(),
               key=lambda c: abs(c.calibre_mm - float(bore_mm)))
    if abs(best.calibre_mm - float(bore_mm)) > 1.0:
        raise KeyError(
            f"no standard container is declared for {bore_mm:.0f} mm -- "
            f"the closest is {best.label} at {best.calibre_mm:.0f} mm. "
            f"Add it to CONTAINERS rather than packing it in the wrong box")
    return best.key
