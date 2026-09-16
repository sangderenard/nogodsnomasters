"""Putting a machine into a chassis, and what does not line up.

BOTH HALVES OF THIS ALREADY EXISTED AND HAD NEVER BEEN INTRODUCED.
stand.py emits real engine bays -- braced squares of corner blocks with
ties into the work area and a lowering frame beneath. engine_package.py
builds a real installable crate and offers `correlate_mounts(cage_nodes)`
to check its mounts against a bay's structural nodes. That method has no
callers anywhere, and its own docstring says why: "there is no real
source for this in the toy yet, so callers without a real cage should
pass an empty dict and expect every mount to come back unmatched,
honestly, rather than a fake match." Nothing ever handed it a cage.

What station_reference.py does instead is the measure of the gap. It
installs two prime movers into two bays as ONE condensed node each, with
a half-extent typed in by hand, and four mounts drawn from that node to
four pallet corners -- while engine_mounts, which resolves technique,
hardware grade, stability and mount loads from the block's own modal
solve, sits unused beside it.

THE ANSWER IS ALMOST NEVER THAT THE BOLTS LINE UP. A crate's feet are
where its own structure puts them and a bay's mount points are where the
frame's members meet; two independently designed bolt patterns coincide
by accident or not at all. Real installations know this and it is why a
mounting frame exists -- an adapter carrying the machine's pattern on
one face and the bay's on the other, which is a fabricated part with a
real mass and a real stack height that raises the machine and moves its
centre of gravity up.

So an installation that mates nothing is not a failure to report. It is
an adapter frame to fabricate, and the useful output is both bolt
patterns and what the frame costs -- which is the cheap description the
fiddly part actually needs.

WHAT SEATING MEANS. A machine is lowered until its feet touch and it is
centred in the space it is going into. Nothing here places a prism by
its middle: the feet land on the bay floor, because that is what the
feet are for, and a crate seated by its centroid would float or sink by
half whatever hangs below it -- on this autoclave, the condensate
collector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from assembly_ports import PartPort, mate_ports, MATE_TOLERANCE_M
from machine_package import MOUNT_PORT_ROLE, MachinePackage


@dataclass(frozen=True)
class Bay:
    """A place in a host structure that a machine can be installed in.

    `mount_ports` are the host's own declared structural mounts. They are
    selected by what the node SAYS -- `port_role` and `bay` -- and never
    by the shape of its identity, so a bay built by stand.py and a bay
    built by hand are the same thing to this module."""
    identity: str
    centre: tuple
    clear_half_extent_m: tuple
    floor_y: float
    mount_ports: list = field(default_factory=list)
    #: what the host can hand a machine across its boundary
    services: tuple = ()
    host_document: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_graph(cls, graph, bay: str | None = None, *,
                   identity: str | None = None,
                   clear_half_extent_m=None, services=()) -> "Bay":
        doc = graph if isinstance(graph, dict) else graph.as_document()
        nodes = doc["nodes"]
        chosen = [n for n in nodes
                  if n.get("port_role") == MOUNT_PORT_ROLE
                  and (bay is None or n.get("bay") == bay)]
        if not chosen:
            raise ValueError(
                f"{doc.get('identity', 'host')}: no node declares "
                f"port_role={MOUNT_PORT_ROLE!r}"
                + (f" for bay {bay!r}" if bay else "")
                + " -- a bay whose mounts are not declared is a space, not "
                  "a bay, and guessing at its corners would be a drawing of "
                  "a mounting")
        pts = np.asarray([n["reference_position"] for n in chosen], float)
        centre = tuple(float(v) for v in pts.mean(axis=0))
        floor_y = float(pts[:, 1].max())
        ports = [PartPort(identity=n["identity"],
                          part=identity or doc.get("identity", "host"),
                          kind=MOUNT_PORT_ROLE,
                          position=np.asarray(n["reference_position"], float),
                          direction=np.asarray(
                              n.get("outward", (0.0, 1.0, 0.0)), float),
                          radius_m=float(n.get("bolt_radius_m", 0.012)),
                          mating=True, fluid="", closure="",
                          joint=str(n.get("joint", "bolted-flange")))
                 for n in chosen]
        if clear_half_extent_m is None:
            span = pts.max(axis=0) - pts.min(axis=0)
            clear_half_extent_m = (float(span[0] / 2.0), 1.0,
                                   float(span[2] / 2.0))
        return cls(identity=identity or f"{doc.get('identity', 'host')}.bay"
                   + (f".{bay}" if bay else ""),
                   centre=centre, clear_half_extent_m=tuple(clear_half_extent_m),
                   floor_y=floor_y, mount_ports=ports, services=tuple(services),
                   host_document=doc)


def seat(package: MachinePackage, bay: Bay) -> np.ndarray:
    """Where the crate goes: centred over the bay, feet on its floor.

    Returns the translation, so nothing is mutated and the same package
    can be offered to several bays."""
    if package.mounts:
        foot_y = min(float(p.position[1]) for p in package.mounts)
    else:
        foot_y = package.prism.min_corner[1]
    centre = package.prism.centre
    return np.array([bay.centre[0] - centre[0],
                     bay.floor_y - foot_y,
                     bay.centre[2] - centre[2]], float)


@dataclass
class AdapterFrame:
    """What has to be fabricated because two bolt patterns do not agree.

    Not a fudge factor and not a fallback: this is the part a real
    installation buys. It carries the machine's pattern on its top face
    and the bay's on its bottom, and it has a real stack height that
    lifts the machine and takes its centre of gravity with it.

    IT IS BEARERS, NOT A FLOOR. The first version of this sized a plate
    over the union of both patterns and reported 584 kg of steel to
    carry an 841 kg machine, which is very nearly the correct weight for
    a bay floor and entirely the wrong part. A machine that is narrower
    than its bay goes on two beams spanning the bay's own mounts, with
    its feet bolted down wherever they land along them -- which is why
    a small machine in a big bay is an ordinary installation rather than
    an absurd one. Sized to its own bending stress at 77 kg, the same
    job."""
    identity: str
    machine_pattern: list          # (name, x, z) the machine bolts to
    bay_pattern: list              # (name, x, z) that bolt to the bay
    span_m: float
    section_b_m: float
    section_h_m: float
    stack_height_m: float
    mass_kg: float

    @property
    def needed(self) -> bool:
        return bool(self.machine_pattern)

    def describe(self) -> list:
        if not self.needed:
            return []
        out = [f"  BEARERS {self.identity}: two, {self.span_m:.2f} m span, "
               f"{self.section_b_m * 1000:.0f} x {self.section_h_m * 1000:.0f} mm "
               f"section, {self.stack_height_m * 1000:.0f} mm stack, "
               f"{self.mass_kg:.0f} kg",
               "    the machine bolts down at:"]
        for name, x, z in self.machine_pattern:
            tail = name.split(".")
            out.append(f"      {'/'.join(tail[-2:]):24s} x {x:+.3f}  z {z:+.3f}")
        out.append("    the bearers land on:")
        for name, x, z in self.bay_pattern:
            out.append(f"      {name.split('.')[-1]:24s} x {x:+.3f}  z {z:+.3f}")
        return out


#: Structural steel worked at a real allowable. 165 MPa is about two
#: thirds of mild steel's yield, which is the ordinary working figure --
#: not a number chosen to make a bearer come out light.
BEARER_ALLOWABLE_PA = 165e6
#: The width of the section. Everything else about it is derived: pick a
#: width a fabricator actually stocks and let the bending decide the
#: depth, rather than typing in a beam and hoping.
BEARER_WIDTH_M = 0.060
STEEL_DENSITY_KG_M3 = 7850.0
GRAVITY_M_S2 = 9.80665


def size_bearers(load_n: float, span_m: float,
                 width_m: float = BEARER_WIDTH_M) -> tuple:
    """How deep a bearer has to be to carry this, over this span.

    Simply supported with the load at midspan, which is the worst place
    it can be and the only assumption worth making when the machine's
    feet can land anywhere along the beam: M = P.L/4, and the section
    modulus has to be M over the allowable. A solid rectangle is the
    conservative section -- a real channel of the same depth is lighter
    and stiffer per kilogramme -- and saying so is cheaper than pretending
    to pick one out of a catalogue."""
    if load_n <= 0.0 or span_m <= 0.0:
        return 0.0, 0.0
    moment = load_n * span_m / 4.0
    z_required = moment / BEARER_ALLOWABLE_PA
    depth = (6.0 * z_required / width_m) ** 0.5
    return width_m, depth


@dataclass
class Installation:
    """One machine, in one bay, and everything that had to be true."""
    package: MachinePackage
    bay: Bay
    offset: np.ndarray
    fits: bool
    seals: list                    # (machine port, bay port) that mated
    unmated_machine: list
    unmated_bay: list
    adapter: AdapterFrame
    unmet_services: tuple
    loads: dict

    @property
    def installed(self) -> bool:
        """Fitted and carried. An installation that needs an adapter is
        still an installation -- the frame is a part, not an excuse --
        but one that does not fit the space is not."""
        return self.fits and bool(self.seals or self.adapter.needed)

    def describe(self) -> str:
        p, b = self.package, self.bay
        sx, sy, sz = p.prism.size_m
        hx, hy, hz = b.clear_half_extent_m
        lines = [
            f"{p.identity} into {b.identity}",
            f"  crate {sx:.2f} x {sy:.2f} x {sz:.2f} m into a clear "
            f"{2 * hx:.2f} x {2 * hy:.2f} x {2 * hz:.2f} m -- "
            + ("fits" if self.fits else "DOES NOT FIT"),
            f"  seated {self.offset[0]:+.3f} {self.offset[1]:+.3f} "
            f"{self.offset[2]:+.3f} m, feet on the bay floor",
            f"  {len(self.seals)} of {len(p.mounts)} feet land on a bay mount"]
        for a, c in self.seals:
            lines.append(f"    bolted  {a.identity.split('.')[-1]:10s} -> "
                         f"{c.identity}")
        lines.extend(self.adapter.describe())
        if self.loads:
            worst = max(self.loads.values(), key=lambda f: abs(f[1]))
            lines.append(f"  heaviest mount carries "
                         f"{abs(worst[1]) / 1000.0:.2f} kN")
        if self.unmet_services:
            lines.append(f"  THE BAY CANNOT SUPPLY: "
                         f"{', '.join(self.unmet_services)}")
        return "\n".join(lines)


def install(package: MachinePackage, bay: Bay, *,
            requires: tuple = (),
            tolerance_m: float = MATE_TOLERANCE_M) -> Installation:
    """Seat a machine in a bay and report what it took.

    The mating is assembly_ports.mate_ports, unchanged -- the same
    solver that decides whether a head lands on a block, on the same
    kind of port, because MATE_PAIRS already pairs structural-mount with
    itself. Nothing here is a second opinion about what a mount is."""
    offset = seat(package, bay)
    moved = [PartPort(identity=p.identity, part=package.identity, kind=p.kind,
                      position=np.asarray(p.position, float) + offset,
                      direction=np.asarray(p.direction, float),
                      radius_m=p.radius_m, mating=True, fluid="",
                      closure="", joint=p.joint)
             for p in package.mounts]
    bay_copy = [PartPort(identity=p.identity, part=bay.identity, kind=p.kind,
                         position=np.asarray(p.position, float),
                         direction=np.asarray(p.direction, float),
                         radius_m=p.radius_m, mating=True, fluid="",
                         closure="", joint=p.joint)
                for p in bay.mount_ports]
    result = mate_ports(moved + bay_copy, tolerance_m=tolerance_m)
    machine_ids = {p.identity for p in moved}
    seals = [(a, c) if a.identity in machine_ids else (c, a)
             for a, c in result.seals]
    sealed = {p.identity for pair in seals for p in pair}
    unmated_machine = [p for p in moved if p.identity not in sealed]
    unmated_bay = [p for p in bay_copy if p.identity not in sealed]

    # WHAT THE FRAME HAS TO BE. Two bearers spanning the bay's own
    # leftover mounts, carrying every foot that found nothing. The span
    # is the bay's, because that is what the beam has to reach; the
    # depth is whatever the bending needs, because that is not a choice.
    if unmated_machine and unmated_bay:
        bx = [float(p.position[0]) for p in unmated_bay]
        bz = [float(p.position[2]) for p in unmated_bay]
        span = max(max(bx) - min(bx), max(bz) - min(bz))
        carried = package.rigid_body.total_mass_kg * GRAVITY_M_S2
        width, depth = size_bearers(carried / 2.0, span)
        mass = round(2.0 * span * width * depth * STEEL_DENSITY_KG_M3, 1)
    else:
        span = width = depth = mass = 0.0
    adapter = AdapterFrame(
        identity=f"{bay.identity}.bearers",
        machine_pattern=[(p.identity, float(p.position[0]),
                          float(p.position[2])) for p in unmated_machine],
        bay_pattern=[(p.identity, float(p.position[0]),
                      float(p.position[2])) for p in unmated_bay],
        span_m=span, section_b_m=width, section_h_m=depth,
        stack_height_m=depth, mass_kg=mass)

    fits = package.prism.fits_within(bay.clear_half_extent_m)
    unmet = tuple(s for s in requires if s not in bay.services)
    return Installation(package=package, bay=bay, offset=offset, fits=fits,
                        seals=seals, unmated_machine=unmated_machine,
                        unmated_bay=unmated_bay, adapter=adapter,
                        unmet_services=unmet,
                        loads=package.static_mount_loads())
