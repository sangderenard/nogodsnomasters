"""Manufacturing declarations carried by existing engine-toy objects.

This is inert authorship data, not a physics engine, material bank, graph
replacement, or player-interaction runtime.  ``electrical_hardware`` emits
Machine/MachinePart/MachineLine objects using these declarations.

SI units; unknown is None with evidence, never a fabricated zero.  A cable's
strand lay, pair lay, overall cabling lay and reel winding are separate facts.
Published qualification ratings are evidence, not constitutive laws.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import math
from typing import Any, Iterable

BASES = frozenset({"published", "derived", "authored", "unknown"})
Vec3 = tuple[float, float, float]


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and infinity are not construction data")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite(child)


def document(value: Any) -> Any:
    """JSON-safe independent copy; fail on non-finite or opaque values."""
    raw = asdict(value) if is_dataclass(value) else value
    return json.loads(json.dumps(raw, allow_nan=False))


def positive(value: float, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (not zero and value == 0.0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def unique(values: Iterable[str], what: str) -> None:
    values = tuple(values)
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{what} requires nonempty string identities")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {what}")


@dataclass(frozen=True)
class Fact:
    value: Any
    unit: str = "1"
    basis: str = "unknown"
    sources: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"unknown evidence basis {self.basis!r}")
        if (self.value is None) != (self.basis == "unknown"):
            raise ValueError("unknown facts have value=None; known facts need a value")
        if self.basis == "published" and not self.sources:
            raise ValueError("published facts need source IDs")
        if self.basis in {"authored", "derived", "unknown"} and not self.note:
            raise ValueError("authored, derived and unknown facts need an explanation")
        _finite(self.value)
        document(self.value)


def known(value: Any, unit: str, source: str, note: str = "") -> Fact:
    return Fact(value, unit, "published", (source,), note)


def authored(value: Any, unit: str = "1", note: str = "Game specimen; not a manufacturer specification") -> Fact:
    return Fact(value, unit, "authored", (), note)


def unknown(unit: str = "1", note: str = "Not established by the reviewed sources") -> Fact:
    return Fact(None, unit, "unknown", (), note)


@dataclass(frozen=True)
class Layer:
    identity: str
    material: Fact
    function: str
    thickness_m: Fact = field(default_factory=lambda: unknown("m"))
    topology: str = "annular"
    interface: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity or not self.function:
            raise ValueError("layers need identity and function")
        if self.topology not in {"annular", "conformal", "tape", "braid", "plating", "filler", "sheet"}:
            raise ValueError("unsupported layer topology")
        if self.thickness_m.value is not None:
            positive(self.thickness_m.value, "layer thickness")
        if self.thickness_m.unit != "m":
            raise ValueError("thickness is in metres")


@dataclass(frozen=True)
class Lay:
    """One hierarchy level, NOT an effective twist for an entire cable.

    Pitch means axial metres per revolution. Unknown manufacturing pitches
    stay unknown. Geometry methods apply only to a single ideal helix.
    """
    identity: str
    level: str                     # strand | pair | overall | tape | reel
    members: tuple[str, ...]
    pattern: str                   # straight | helix | braid | unknown
    pitch_m: Fact = field(default_factory=lambda: unknown("m/turn"))
    radius_m: Fact = field(default_factory=lambda: unknown("m"))
    handedness: str = "unknown"     # right | left | alternating | none | unknown
    phase_rad: Fact = field(default_factory=lambda: unknown("rad"))
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in {"strand", "pair", "overall", "tape", "reel"}:
            raise ValueError("invalid lay level")
        if self.pattern not in {"straight", "helix", "braid", "unknown"}:
            raise ValueError("invalid lay pattern")
        if self.handedness not in {"right", "left", "alternating", "none", "unknown"}:
            raise ValueError("invalid handedness")
        if self.pitch_m.unit != "m/turn" or self.radius_m.unit != "m":
            raise ValueError("lay uses metres and metres per turn")
        if self.pitch_m.value is not None:
            positive(self.pitch_m.value, "pitch")
        if self.radius_m.value is not None:
            positive(self.radius_m.value, "helix radius", zero=True)
        unique(self.members, "lay member")

    def turns_for(self, axial_length_m: float) -> float:
        length = positive(axial_length_m, "axial length", zero=True)
        if self.pattern == "straight":
            return 0.0
        if self.pattern != "helix" or self.pitch_m.value is None:
            raise ValueError("turn count needs a declared helical pitch")
        return length / float(self.pitch_m.value)

    def ideal_path_length_m(self, axial_length_m: float) -> float:
        length = positive(axial_length_m, "axial length", zero=True)
        if self.pattern == "straight":
            return length
        if self.pattern != "helix" or self.pitch_m.value is None or self.radius_m.value is None:
            raise ValueError("path length needs a declared single-helix radius and pitch")
        return length * math.hypot(1.0, 2.0 * math.pi * float(self.radius_m.value) / float(self.pitch_m.value))


@dataclass(frozen=True)
class Core:
    identity: str
    role: str
    material: Fact
    awg: Fact
    strand_count: Fact
    strand_diameter_m: Fact = field(default_factory=lambda: unknown("m"))
    metal_area_mm2: Fact = field(default_factory=lambda: unknown("mm2"))
    insulation: tuple[Layer, ...] = ()
    strand_lays: tuple[Lay, ...] = ()
    color: Fact = field(default_factory=unknown)
    temper: Fact = field(default_factory=unknown)
    plating: tuple[Layer, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity or not self.role:
            raise ValueError("cores need identity and role")
        n = self.strand_count.value
        if n is not None and (isinstance(n, bool) or int(n) != n or n < 1):
            raise ValueError("strand count must be a positive integer")
        for prop, name in ((self.strand_diameter_m, "strand diameter"), (self.metal_area_mm2, "metal area")):
            if prop.value is not None:
                positive(prop.value, name)
        unique((layer.identity for layer in self.insulation + self.plating), "core layer")


@dataclass(frozen=True)
class CableConstruction:
    identity: str
    label: str
    sector: str
    cores: tuple[Core, ...]
    layers: tuple[Layer, ...]
    lays: tuple[Lay, ...] = ()
    cross_section: Fact = field(default_factory=unknown)
    outer_dimensions_m: Fact = field(default_factory=lambda: unknown("m"))
    ratings: dict[str, Fact] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sector not in {"home", "military", "industrial"}:
            raise ValueError("invalid catalogue sector")
        if not self.cores:
            raise ValueError("a cable needs cores")
        unique((c.identity for c in self.cores), "core")
        unique((c.role for c in self.cores), "core role")
        unique((x.identity for x in self.layers), "cable layer")
        ids = {c.identity for c in self.cores}
        for lay in self.lays:
            if lay.level in {"pair", "overall"} and not set(lay.members) <= ids:
                raise ValueError(f"{lay.identity}: unknown core in lay")
        for dim in self.outer_dimensions_m.value or ():
            positive(dim, "outer dimension")

    def graph_attributes(self) -> dict[str, Any]:
        return {"construction_schema": "engine-toy-hardware-construction-v1",
                "physical_construction": document(self),
                "construction_is_runtime_solver": False}


@dataclass(frozen=True)
class Contact:
    identity: str
    role: str
    position_m: Fact
    material: Fact
    shape: Fact
    plating: tuple[Layer, ...] = ()
    dimensions_m: Fact = field(default_factory=lambda: unknown("m"))
    normal_force_n: Fact = field(default_factory=lambda: unknown("N"))
    termination: Fact = field(default_factory=unknown)
    terminal_torque_nm: Fact = field(default_factory=lambda: unknown("N m"))
    first_touch_m: Fact = field(default_factory=lambda: unknown("m"))
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.position_m.value is not None and len(self.position_m.value) != 3:
            raise ValueError("contact positions are local 3-vectors")
        if self.position_m.unit != "m":
            raise ValueError("contact positions are in metres")


@dataclass(frozen=True)
class Seal:
    identity: str
    regions: tuple[str, str]
    material: Fact
    mechanism: str
    active_configurations: tuple[str, ...]
    section_m: Fact = field(default_factory=lambda: unknown("m"))
    installed_compression: Fact = field(default_factory=unknown)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.regions) != 2 or self.regions[0] == self.regions[1]:
            raise ValueError("a seal separates two distinct named regions")
        unique(self.active_configurations, "seal configuration")


@dataclass(frozen=True)
class ConnectorConstruction:
    identity: str
    label: str
    sector: str
    family: str
    interface_key: str
    form: str                     # plug | receptacle | inlet | cable-coupler
    contacts: tuple[Contact, ...]
    shell_material: Fact
    insert_material: Fact
    coupling: Fact
    seals: tuple[Seal, ...] = ()
    ratings: dict[str, Fact] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unique((c.identity for c in self.contacts), "contact")
        unique((c.role for c in self.contacts), "contact role")
        unique((s.identity for s in self.seals), "seal")
        if self.form not in {"plug", "receptacle", "inlet", "cable-coupler"}:
            raise ValueError("invalid connector form")
        if not self.contacts or not self.interface_key:
            raise ValueError("connector needs contacts and an explicit interface key")

    def graph_attributes(self) -> dict[str, Any]:
        return {"construction_schema": "engine-toy-hardware-construction-v1",
                "physical_construction": document(self),
                "connector_standard": self.family,
                "connector_interface_key": self.interface_key,
                "terminal_roles": [c.role for c in self.contacts]}


def compatibility(a: ConnectorConstruction, b: ConnectorConstruction) -> tuple[bool, tuple[str, ...]]:
    """Catalogue mating gate only: NOT a voltage, safety or engagement solver.

    This intentionally does not infer a common pinout from equal contact counts,
    and does not claim all variants of the same connector family can mate.
    """
    reasons = []
    if a.interface_key != b.interface_key:
        reasons.append("different keyed interfaces")
    pair = frozenset((a.form, b.form))
    if pair not in {frozenset(("plug", "receptacle")), frozenset(("inlet", "cable-coupler")), frozenset(("plug", "cable-coupler"))}:
        reasons.append("forms are not a mating pair")
    if {c.identity: c.role for c in a.contacts} != {c.identity: c.role for c in b.contacts}:
        reasons.append("terminal role maps differ")
    return not reasons, tuple(reasons)


def route_length_m(points: Iterable[Iterable[float]]) -> float:
    pts = tuple(tuple(float(x) for x in row) for row in points)
    if len(pts) < 2 or any(len(p) != 3 for p in pts):
        raise ValueError("a route needs at least two 3D points")
    _finite(pts)
    lengths = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    if any(x <= 0.0 for x in lengths):
        raise ValueError("zero-length route segments are not permitted")
    return sum(lengths)


def unknown_facts(value: Any, path: str = "") -> list[str]:
    """Report incomplete research without silently filling in missing facts."""
    data = document(value)
    def walk(item, here):
        if isinstance(item, dict):
            if item.get("basis") == "unknown" and "value" in item:
                return [here]
            return [v for k, child in item.items() for v in walk(child, f"{here}.{k}".strip("."))]
        if isinstance(item, list):
            return [v for i, child in enumerate(item) for v in walk(child, f"{here}[{i}]" )]
        return []
    return walk(data, path)
