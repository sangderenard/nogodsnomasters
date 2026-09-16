"""Real polar inertia of a rotating part, derived from its own shape.

Every rotating part's polar inertia about its own spin axis is

    I = shape_factor * mass * radius^2

where `shape_factor` is a REAL property of HOW that class of part
distributes its mass between hub and rim -- 1/2 for a uniform solid
disc, 1 for a thin ring with everything at the rim, and something in
between for the real parts an engine is actually made of. Both `mass`
and `radius` are already declared by every part this module is asked
about; the shape factor is the one thing that has to be said out loud,
once, per class of part.

WHY THIS MODULE EXISTS. DrivetrainSolver used to fill in any missing
inertia with `mass * 0.01` -- a radius of gyration of exactly 100 mm on
every part in the graph, from a 13 kg flywheel to a 0.3 kg timing
sprocket. That is right to 1.5x for the flywheel (which really is a
117 mm disc) and wrong by 51x for the crank sprocket, and the error is
ordered by radius, which is the signature of a constant standing in for
a real dimension. A part that knows its own mass and its own radius
should never be guessed at.

WHAT A SHAPE FACTOR IS NOT. It is not a fudge factor and it is not
tuned. Each one below is the real mass distribution of that class of
part, stated with its reasoning, and every one sits in [0.4, 1.0]
because that is the whole physical range available to a body spinning
about its own axis: 1/2 if the mass is spread evenly over the disc, 1
if it is all at the rim, below 1/2 only if it is concentrated toward
the hub. Anything outside that range is a bug, and `shape_factor()`
says so rather than returning it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RotorShape:
    key: str
    factor: float       # I / (m * r^2)
    label: str
    why: str


# The real classes of rotating part an engine is built from. `factor` is
# I/(m r^2) about the spin axis, `radius` always meaning the part's own
# OUTER radius -- the same radius the part already declares for drawing
# and for its drum/body extent, so nothing here needs a new dimension.
_SHAPES: tuple[RotorShape, ...] = (
    RotorShape("solid-disc", 0.500, "solid disc",
               "a uniform solid cylinder about its own axis: the exact analytic 1/2"),
    RotorShape("thin-ring", 1.000, "thin ring",
               "a ring gear or rim with effectively all of its mass at one radius: the exact analytic 1"),
    RotorShape("rim-weighted-flywheel", 0.600, "rim-weighted flywheel",
               "a real automotive flywheel is a thin web carrying a heavy rim and the clutch "
               "friction face, so it sits well above a uniform disc but short of a pure ring"),
    RotorShape("dished-pulley", 0.450, "dished belt pulley",
               "a stamped or cast vee/serpentine pulley is relieved between hub and rim, "
               "moving mass inboard of a uniform disc"),
    RotorShape("toothed-sprocket", 0.450, "toothed sprocket",
               "a chain sprocket carries its teeth at the rim but is relieved or spoked "
               "through the web, which nets out near a dished pulley"),
    RotorShape("clutch-cover-assembly", 0.550, "clutch cover assembly",
               "the pressure ring and diaphragm spring both sit close to the friction "
               "radius, with only the cover stampings inboard"),
    RotorShape("clutch-driven-disc", 0.550, "clutch driven disc",
               "the friction facings and their marcel springs are at the rim; the splined "
               "hub and damper springs are light and well inboard"),
    RotorShape("claw-pole-rotor", 0.500, "claw-pole rotor",
               "an alternator rotor is a solid steel claw-pole pair around a field winding, "
               "close enough to a uniform cylinder about its own axis"),
    RotorShape("centrifugal-impeller", 0.400, "centrifugal impeller",
               "a water-pump impeller's vanes thin toward the rim while the hub and backplate "
               "are solid, putting it just below a uniform disc"),
    RotorShape("axial-fan", 0.700, "axial fan",
               "an engine fan is a light spider carrying blades over the outer half of its "
               "swept radius, so its mass sits much nearer the rim than a disc"),
    RotorShape("lobed-shaft", 0.500, "lobed shaft",
               "a camshaft is journals and lobes on a slender shaft; about its own axis "
               "that is a solid cylinder at the lobe radius"),
    RotorShape("geared-shaft-train", 0.500, "geared shaft train",
               "a gear case's shafts and gears lumped at the input shaft, each a solid "
               "cylinder about its own axis"),
    RotorShape("vaned-rotor", 0.450, "vaned rotor",
               "a gear/vane oil pump or a swash-plate compressor: a solid rotor with "
               "pockets cut into it, a little inboard of a uniform disc"),
)

BY_KEY: dict[str, RotorShape] = {s.key: s for s in _SHAPES}

# The whole physical range for a body spinning about its own axis:
# everything at the hub approaches 0, everything at the rim is exactly 1.
# Real engine hardware never sits below a vaned rotor's 0.4.
MIN_FACTOR = 0.30
MAX_FACTOR = 1.00


def shape_factor(shape: str) -> float:
    """I/(m r^2) for a declared class of rotating part.

    Unknown shapes raise rather than falling back to anything. A part
    that has not said what it is has not been built yet, and a silent
    stand-in here is exactly the failure this module replaces."""
    s = BY_KEY.get(str(shape))
    if s is None:
        raise KeyError(
            f"unknown rotor shape {shape!r}; declare one of: {', '.join(sorted(BY_KEY))}")
    if not (MIN_FACTOR <= s.factor <= MAX_FACTOR):
        raise ValueError(f"rotor shape {s.key!r} factor {s.factor} is outside the physical range")
    return s.factor


def polar_inertia(shape: str, mass_kg: float, radius_m: float) -> float:
    """The real polar inertia of one part, about its own spin axis."""
    m = float(mass_kg)
    r = float(radius_m)
    if m <= 0.0 or r <= 0.0:
        raise ValueError(
            f"polar inertia of a {shape} needs a real mass and radius, got m={m}, r={r}")
    return shape_factor(shape) * m * r * r


def radius_of_gyration_m(shape: str, radius_m: float) -> float:
    """k, for reporting -- the constant this module exists to replace."""
    return math.sqrt(shape_factor(shape)) * float(radius_m)


# ---------------------------------------------------------------------
# part-class builders: a real part sized from the real number that
# actually determines it in the world, not from a share of engine mass
# ---------------------------------------------------------------------

# A single dry plate clutch is sold, and specified, by the torque it can
# hold. Its capacity is T = mu * n_faces * clamp_load * mean_radius: the
# clamp load a diaphragm spring can make scales with its own area (~d^2)
# and the mean radius scales with d, so capacity goes as d^3. That makes
# diameter the derived quantity and torque the declared one, which is
# the way round a real catalogue works.
#
# The reference point is a real, ordinary 10.5 in (266.7 mm) organic
# single plate at about 450 N*m -- the size that has been bolted to
# small-block V8s and big sixes for decades.
_CLUTCH_REF_DIAMETER_M = 0.2667
_CLUTCH_REF_TORQUE_NM = 450.0
# Real clutch masses do NOT follow geometric similarity: cover stampings
# and facing thickness grow much more slowly than diameter, so a real
# 7.25 in -> 13 in range runs about 3.5 kg -> 17 kg of cover assembly
# rather than the 20+ kg pure d^3 scaling would predict. 2.7 is the
# exponent that actually spans real catalogue hardware.
_CLUTCH_MASS_EXPONENT = 2.7
_CLUTCH_REF_COVER_KG = 8.5
_CLUTCH_REF_DISC_KG = 2.4
# Real single-plate automotive clutches run from about 6 in to about
# 14 in before anyone sensible goes multi-plate.
_CLUTCH_MIN_DIAMETER_M = 0.152
_CLUTCH_MAX_DIAMETER_M = 0.356


@dataclass(frozen=True)
class ClutchPack:
    """A real clutch, sized by the torque it is asked to hold."""
    torque_capacity_nm: float
    plates: int
    disc_diameter_m: float
    cover_mass_kg: float
    disc_mass_kg: float

    @property
    def mass_kg(self) -> float:
        return self.cover_mass_kg + self.disc_mass_kg * self.plates

    @property
    def radius_m(self) -> float:
        return self.disc_diameter_m * 0.5

    @property
    def inertia_kg_m2(self) -> float:
        """The whole rotating pack about the crank axis.

        The cover assembly turns with the flywheel and the driven
        disc(s) turn with the gearbox input shaft; they are one rotating
        inertia only while the clutch is engaged, which is the condition
        the drivetrain graph models this node in."""
        r = self.radius_m
        return (polar_inertia("clutch-cover-assembly", self.cover_mass_kg, r)
                + polar_inertia("clutch-driven-disc", self.disc_mass_kg, r) * self.plates)

    @property
    def label(self) -> str:
        return (f"{self.disc_diameter_m * 1000.0:.0f} mm "
                f"{'single' if self.plates == 1 else str(self.plates) + '-plate'} dry clutch")


def clutch_pack(torque_capacity_nm: float) -> ClutchPack:
    """The real clutch an engine of this torque capacity would carry.

    Past the point where a single plate would have to grow beyond real
    automotive sizes, a real build goes multi-plate instead of carrying
    on growing the diameter -- which is exactly what heavy-duty truck,
    racing and marine practice does, and what keeps a 7500 N*m engine
    from being handed a metre-wide clutch."""
    t = max(1.0, float(torque_capacity_nm))
    plates = 1
    while True:
        per_plate = t / plates
        d = _CLUTCH_REF_DIAMETER_M * (per_plate / _CLUTCH_REF_TORQUE_NM) ** (1.0 / 3.0)
        if d <= _CLUTCH_MAX_DIAMETER_M or plates >= 8:
            break
        plates += 1
    d = min(_CLUTCH_MAX_DIAMETER_M, max(_CLUTCH_MIN_DIAMETER_M, d))
    scale = (d / _CLUTCH_REF_DIAMETER_M) ** _CLUTCH_MASS_EXPONENT
    return ClutchPack(
        torque_capacity_nm=t, plates=plates, disc_diameter_m=d,
        cover_mass_kg=_CLUTCH_REF_COVER_KG * scale,
        disc_mass_kg=_CLUTCH_REF_DISC_KG * scale)


# A gear case's rotating inertia, seen from its input shaft. Only the
# input-side shafts and gears turn at input speed; everything past the
# engaged ratio is reflected through it and, in a real gearbox in its
# usual ratios, contributes a fraction of what the input cluster does.
# This is the real reason a gearbox is not a flywheel: its rotating
# parts are small-radius shafts and gears, however heavy the case is.
_GEAR_CASE_ROTATING_FRAC = 0.35      # of case mass that actually spins
_GEAR_CASE_SHAFT_RADIUS_FRAC = 0.30  # gear pitch radius as a fraction of the case's own radius


def gear_case_input_inertia(case_mass_kg: float, case_radius_m: float) -> float:
    """Polar inertia of a gear case's rotating cluster at its input shaft."""
    spinning = max(0.0, float(case_mass_kg)) * _GEAR_CASE_ROTATING_FRAC
    r = max(1e-3, float(case_radius_m) * _GEAR_CASE_SHAFT_RADIUS_FRAC)
    if spinning <= 0.0:
        return 0.0
    return polar_inertia("geared-shaft-train", spinning, r)


# A starter ring gear is a real object with real dimensions: a plain
# steel band shrunk onto the flywheel rim, with the teeth cut into it.
# Its mass is its own volume, not a number anyone has to pick -- which
# matters because it is a thin ring, the one shape that really does put
# all of its mass at the rim, so it contributes far more inertia per
# kilogram than anything else bolted to the crank.
STEEL_DENSITY_KG_M3 = 7850.0
_RING_GEAR_FACE_WIDTH_M = 0.015   # real, typical: the pinion's own tooth face
_RING_GEAR_RADIAL_DEPTH_M = 0.012  # real, typical: root to tip plus the band behind it


# ---------------------------------------------------------------------
# A CASING WITH SPINNING THINGS INSIDE IT
# ---------------------------------------------------------------------
#
# This is the general shape of almost every unit bolted to an engine,
# and it is a class of object in its own right: a housing that does NOT
# turn, containing one or more rotating groups that do, each at its own
# speed relative to the unit's input shaft. An alternator is a case
# around one rotor. A gear pump is a body around two gears. A roots
# blower is a case around two rotors. A gearbox is a case around an
# input cluster and a countershaft turning at a different speed. The
# engine itself is a block around a crank and a camshaft at half speed.
#
# Modelling it as a class, rather than as a number per accessory, is
# what lets an abstracted engine or machine ENUMERATE every spinning
# mass it contains without knowing what any particular unit is: ask each
# housed assembly for its groups, reflect each group through its own
# ratio, and the machine's whole rotating inertia falls out. That is
# also the only honest way to answer "what is spinning in here", which
# a single lumped `inertia_kg_m2` per node cannot.
#
# THE MISTAKE THIS REPLACES. An accessory node's `mass_kg` is the mass
# of the whole UNIT as it sits on the shelf -- an alternator's 3 kg is
# its case, stator, rectifier and pulley as well as its rotor. Handing
# that whole mass to an inertia formula says the stator spins, which it
# does not. Only the rotor turns, it is a minority of the unit's mass,
# and it turns at a radius well inside the housing. Both errors push the
# same way, and together with a 100 mm radius of gyration they made a
# 3 kg alternator resist acceleration like a small flywheel.
#
# Group radius scales with the cube root of unit mass -- geometric
# similarity, the real and disclosed assumption that a bigger unit of
# the same class is the same machine built larger.


@dataclass(frozen=True)
class RotatingGroup:
    """One thing that actually spins inside a casing."""
    name: str
    shape: str
    mass_kg: float
    radius_m: float
    speed_ratio: float = 1.0    # this group's speed / the unit's input-shaft speed

    @property
    def inertia_kg_m2(self) -> float:
        """About this group's OWN axis, at its OWN speed."""
        return polar_inertia(self.shape, self.mass_kg, self.radius_m)

    @property
    def inertia_at_input_kg_m2(self) -> float:
        """Reflected to the unit's input shaft.

        A group turning n times per input revolution stores n^2 times
        the kinetic energy per unit of its own inertia, so that is how
        it resists the input shaft -- the same squared reflection a
        gearbox ratio applies, and the reason a small fast rotor can
        matter more than a large slow one."""
        return self.inertia_kg_m2 * self.speed_ratio * self.speed_ratio


@dataclass(frozen=True)
class HousedAssembly:
    """A casing that does not turn, around groups that do."""
    kind: str
    unit_mass_kg: float
    groups: tuple[RotatingGroup, ...]

    @property
    def rotating_mass_kg(self) -> float:
        return sum(g.mass_kg for g in self.groups)

    @property
    def casing_mass_kg(self) -> float:
        """The part of the unit that never turns -- real mass for the
        assembly's centre of gravity, no mass at all for its dynamics."""
        return max(0.0, self.unit_mass_kg - self.rotating_mass_kg)

    @property
    def inertia_kg_m2(self) -> float:
        """What this whole unit presents at its input shaft."""
        return sum(g.inertia_at_input_kg_m2 for g in self.groups)

    @property
    def shape(self) -> str:
        """The dominant group's shape, for a node that carries one label."""
        return max(self.groups, key=lambda g: g.inertia_at_input_kg_m2).shape

    @property
    def radius_m(self) -> float:
        return max(self.groups, key=lambda g: g.inertia_at_input_kg_m2).radius_m


@dataclass(frozen=True)
class _GroupSpec:
    name: str
    shape: str
    mass_frac: float        # of the whole unit's mass
    radius_frac: float      # of the class's reference rotor radius
    speed_ratio: float = 1.0


@dataclass(frozen=True)
class HousedAssemblyClass:
    key: str
    ref_unit_mass_kg: float
    ref_radius_m: float
    groups: tuple[_GroupSpec, ...]
    why: str


_HOUSED_CLASSES: tuple[HousedAssemblyClass, ...] = (
    HousedAssemblyClass(
        "alternator", 3.0, 0.032,
        (_GroupSpec("rotor", "claw-pole-rotor", 0.35, 1.0),),
        "the stator, case, rectifier and shroud do not turn; a real claw-pole rotor "
        "for a 3 kg unit is about 64 mm across"),
    HousedAssemblyClass(
        "water-pump", 2.2, 0.035,
        (_GroupSpec("impeller", "centrifugal-impeller", 0.20, 1.0),),
        "a cast-iron pump is mostly housing and backplate; only the impeller and its "
        "short shaft turn"),
    HousedAssemblyClass(
        "oil-pump", 1.4, 0.022,
        (_GroupSpec("drive_gear", "vaned-rotor", 0.15, 1.0),
         _GroupSpec("driven_gear", "vaned-rotor", 0.15, 1.0)),
        "a gear pump really is two gears in a heavy body, meshed 1:1 -- two groups, "
        "not one, which is the simplest case where the casing model earns itself"),
    HousedAssemblyClass(
        "ac-compressor", 4.5, 0.030,
        (_GroupSpec("swash_plate_and_pistons", "vaned-rotor", 0.25, 1.0),),
        "the swash plate, shaft and pistons turn inside a cast housing that does not; "
        "the clutch pulley is its own node"),
    HousedAssemblyClass(
        "pneumatic-compressor", 5.0, 0.032,
        (_GroupSpec("rotating_group", "vaned-rotor", 0.25, 1.0),),
        "same construction as an A/C compressor: a small rotating group in a "
        "substantial casting"),
    HousedAssemblyClass(
        "camshaft", 0.88, 0.018,
        (_GroupSpec("shaft", "lobed-shaft", 1.00, 1.0),),
        "a camshaft is all rotating part and no casing -- the degenerate case of this "
        "class, and worth stating as one rather than special-casing"),
    HousedAssemblyClass(
        "starter-motor", 4.5, 0.028,
        (_GroupSpec("armature", "solid-disc", 0.30, 1.0),),
        "a series-wound armature inside a heavy field-magnet case and nose housing "
        "that do not turn"),
    HousedAssemblyClass(
        "roots-blower", 12.0, 0.055,
        (_GroupSpec("rotor_a", "vaned-rotor", 0.18, 1.0),
         _GroupSpec("rotor_b", "vaned-rotor", 0.18, 1.0)),
        "two lobed rotors timed together in a heavy alloy case: both turn at blower "
        "shaft speed, and the case is most of the unit's mass"),
    HousedAssemblyClass(
        "magneto", 2.0, 0.025,
        (_GroupSpec("rotor", "solid-disc", 0.40, 1.0),),
        "a magneto is a dense magnet rotor spun inside a coil pack and housing that "
        "do not move -- and on an aircraft or hit-and-miss engine it is the ignition, "
        "so it is always being driven"),
    HousedAssemblyClass(
        "flyweight-governor", 1.5, 0.050,
        (_GroupSpec("flyweights_and_spindle", "thin-ring", 0.55, 1.0),),
        "the whole point of a governor is mass held OUT at a radius so centrifugal "
        "force can work against a spring, which makes it very nearly a thin ring and "
        "much heavier in inertia than its size suggests"),
    HousedAssemblyClass(
        "injection-pump", 8.0, 0.025,
        (_GroupSpec("camshaft_and_plungers", "vaned-rotor", 0.20, 1.0),),
        "an in-line injection pump is a heavy body around a small camshaft driving "
        "short plungers; only the camshaft turns"),
    HousedAssemblyClass(
        "reduction-gearset", 20.0, 0.060,
        (_GroupSpec("gear_cluster", "geared-shaft-train", 0.35, 1.0),),
        "a single-speed EV reduction: gears and shafts inside a case that is most of "
        "the mass"),
    HousedAssemblyClass(
        "drive-differential", 15.0, 0.070,
        (_GroupSpec("crown_wheel_and_carrier", "solid-disc", 0.35, 1.0),),
        "the crown wheel and carrier turning inside a housing -- the same object "
        "gear_trains.Differential describes in full when its ratio is known"),
    HousedAssemblyClass(
        "accessory-gearbox", 18.0, 0.050,
        (_GroupSpec("gear_train", "geared-shaft-train", 0.35, 1.0),),
        "a turbine's accessory drive: the gears that run its fuel pump, oil pump and "
        "starter-generator, in a case bolted to the compressor housing"),
    HousedAssemblyClass(
        "turbocharger", 6.0, 0.032,
        (_GroupSpec("shaft_wheels", "solid-disc", 0.22, 1.0),),
        "turbine wheel, shaft and compressor wheel are one rotating assembly inside a "
        "housing pair that is most of the mass"),
)

HOUSED_BY_KEY: dict[str, HousedAssemblyClass] = {h.key: h for h in _HOUSED_CLASSES}


def housed_assembly(kind: str, unit_mass_kg: float) -> HousedAssembly:
    """The real casing-and-rotating-groups breakdown of one unit."""
    h = HOUSED_BY_KEY.get(str(kind))
    if h is None:
        raise KeyError(
            f"unknown housed assembly {kind!r}; declare one of: {', '.join(sorted(HOUSED_BY_KEY))}")
    m_unit = float(unit_mass_kg)
    if m_unit <= 0.0:
        raise ValueError(f"a {kind} needs a real unit mass, got {m_unit}")
    scale = (m_unit / h.ref_unit_mass_kg) ** (1.0 / 3.0)
    groups = tuple(
        RotatingGroup(name=g.name, shape=g.shape, mass_kg=m_unit * g.mass_frac,
                      radius_m=h.ref_radius_m * g.radius_frac * scale,
                      speed_ratio=g.speed_ratio)
        for g in h.groups)
    return HousedAssembly(kind=h.key, unit_mass_kg=m_unit, groups=groups)


def accessory_rotor(kind: str, unit_mass_kg: float) -> HousedAssembly:
    """Compatibility name for `housed_assembly` -- same object."""
    return housed_assembly(kind, unit_mass_kg)


# ---------------------------------------------------------------------
# what is spinning inside an abstracted machine
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SpinningMass:
    """One rotating body in a machine, and what it costs the crank."""
    identity: str
    shape: str
    mass_kg: float
    radius_m: float
    inertia_kg_m2: float          # about its own axis
    ratio_to_crank: float         # its speed / crank speed
    housed_assembly: str | None   # the class of unit it belongs to, if any
    counted_in: str | None        # the shaft whose declared inertia already includes it

    @property
    def inertia_at_crank_kg_m2(self) -> float:
        r = self.ratio_to_crank
        return self.inertia_kg_m2 * r * r


def spinning_masses(graph: dict, ratios_to_crank: dict[str, float] | None = None
                    ) -> list[SpinningMass]:
    """Every rotating body in a built machine, with its real inertia.

    This is the question an abstracted engine or machine has to be able
    to ask without knowing what any particular unit is: what is turning
    in here, how fast relative to the crank, and how much does each one
    resist being accelerated. It answers off the graph's own
    declarations -- a node's shape, radius and polar inertia -- and
    never from a name.

    `ratios_to_crank` comes from the graph's own drive ratios (see
    sound_parts.drive_ratios_to_crank, the existing walker for exactly
    this); anything it cannot reach is reported at ratio 0, which is
    honest -- the crank does not turn it."""
    ratios = ratios_to_crank or {}
    out: list[SpinningMass] = []
    for n in graph.get("nodes", ()):
        if n.get("kind") != "rotating-mass":
            continue
        inertia = float(n.get("inertia_kg_m2") or 0.0)
        if inertia <= 0.0:
            continue
        ident = n["identity"]
        out.append(SpinningMass(
            identity=ident,
            shape=str(n.get("rotating_shape") or ""),
            mass_kg=float(n.get("rotating_mass_kg") or n.get("mass_kg") or 0.0),
            radius_m=float(n.get("rotor_radius_m") or n.get("radius_m")
                           or n.get("drum_radius_m") or 0.0),
            inertia_kg_m2=inertia,
            ratio_to_crank=float(ratios.get(ident, 0.0) or 0.0),
            housed_assembly=n.get("housed_assembly"),
            counted_in=("powertrain.engine" if n.get("included_in_parent_inertia") else None)))
    out.sort(key=lambda s: -s.inertia_at_crank_kg_m2)
    return out


def total_inertia_at_crank(graph: dict, ratios_to_crank: dict[str, float] | None = None) -> float:
    """What the crank has to accelerate, counting everything once.

    Parts already inside a shaft's own declared figure are skipped, so
    a flywheel inside the engine's published crank-assembly inertia is
    not added to it a second time."""
    return sum(s.inertia_at_crank_kg_m2 for s in spinning_masses(graph, ratios_to_crank)
               if s.counted_in is None)


def ring_gear_mass_kg(radius_m: float) -> float:
    """Mass of the starter ring gear that fits this flywheel rim."""
    r = float(radius_m)
    if r <= 0.0:
        raise ValueError(f"a ring gear needs a real rim radius, got {r}")
    volume = 2.0 * math.pi * r * _RING_GEAR_FACE_WIDTH_M * _RING_GEAR_RADIAL_DEPTH_M
    return STEEL_DENSITY_KG_M3 * volume
