"""Spoil, piles and berms: what to do with the hole you took out.

Digging produces two things and only one of them is a hole. The other is
a heap, and it is bigger than the hole was -- broken ground does not go
back into the space it came from. That heap has to be somewhere, and the
obvious somewhere is around whatever you just dug next to.

WHY THE DEFAULT IS A BERM. Spoil is free protective material that has
already been paid for in labour. Earth is a genuinely good bullet stop:
half a metre of packed earth defeats small arms, and it does it by being
cheap and thick rather than by being clever. A position that has just
dug an ice pit has, without doing any further work, enough material to
revet itself -- and moving it twice is the mistake, so it should go
where it is wanted the first time.

THE COSTS THAT ARE REAL AND USUALLY FORGOTTEN:

  swell         soil bulks when you break it. A cubic metre in the
                ground is about a quarter more than that in a heap, and
                tamping it back only gets part of that back. The heap is
                always bigger than the hole.
  angle         a pile cannot be steeper than its material allows. Sand
                stands at about thirty degrees and nothing will make it
                stand steeper, so a tall sand berm is a WIDE berm, and
                the footprint is what runs out first.
  visibility    fresh spoil is a different colour and temperature from
                the ground it came from, and a berm is a straight line
                where there was not one. It is protection that announces
                the thing it protects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpoilClass:
    key: str
    label: str
    swell_factor: float          # loose volume / in-situ volume
    compacted_factor: float      # placed and tamped, relative to in-situ
    repose_deg: float            # the steepest a heap of it will stand
    density_kg_m3: float
    stopping_m: float            # thickness that defeats rifle-calibre small arms
    why: str


SPOIL: dict[str, SpoilClass] = {
    "sand": SpoilClass(
        "sand", "sand", 1.12, 0.95, 31.0, 1600.0, 0.60,
        why="flows rather than stands, so a sand berm is wide and low unless it is "
            "revetted. It is also the best of these at stopping things, because it "
            "moves out of the way and takes energy doing it"),
    "ordinary-soil": SpoilClass(
        "ordinary-soil", "ordinary soil", 1.25, 1.05, 35.0, 1800.0, 0.50,
        why="the reference. Bulks a quarter when broken and never quite goes back"),
    "hardpan": SpoilClass(
        "hardpan", "broken hardpan", 1.35, 1.10, 38.0, 1900.0, 0.45,
        why="comes out in lumps, which is why it bulks most and why a pile of it has "
            "voids that rain will find"),
    "caliche": SpoilClass(
        "caliche", "broken caliche", 1.40, 1.15, 40.0, 2000.0, 0.40,
        why="rubble rather than soil: it stands steeply and it does not knit together, "
            "so a berm of it is a pile of rocks with the strengths of one"),
    "rock": SpoilClass(
        "rock", "rock spoil", 1.50, 1.30, 42.0, 2400.0, 0.35,
        why="all voids. Excellent revetment fill inside something that holds it, and "
            "useless on its own because anything that hits it makes secondary "
            "fragments out of it"),
}


def spoil_class(key: str) -> SpoilClass:
    s = SPOIL.get(str(key))
    if s is None:
        raise KeyError(f"unknown spoil {key!r}; declared: {', '.join(sorted(SPOIL))}")
    return s


@dataclass
class DirtPile:
    """The default place dirt goes.

    Anything that digs puts its spoil here, and anything that wants
    earth takes it from here. It is deliberately one shared heap rather
    than a property of whatever made it, because the whole point is that
    the material from one job is the material for the next."""
    identity: str = "station.spoil"
    kind: str = "hardpan"
    in_situ_m3: float = 0.0          # measured as it was in the ground
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def spec(self) -> SpoilClass:
        return spoil_class(self.kind)

    @property
    def loose_m3(self) -> float:
        """What it actually occupies as a heap -- more than the hole."""
        return self.in_situ_m3 * self.spec.swell_factor

    @property
    def mass_kg(self) -> float:
        return self.in_situ_m3 * self.spec.density_kg_m3

    @property
    def placed_m3(self) -> float:
        """What it makes when placed and tamped into a berm."""
        return self.in_situ_m3 * self.spec.compacted_factor

    def add(self, in_situ_m3: float) -> float:
        self.in_situ_m3 += max(0.0, float(in_situ_m3))
        return self.in_situ_m3

    def take(self, in_situ_m3: float) -> float:
        got = min(self.in_situ_m3, max(0.0, float(in_situ_m3)))
        self.in_situ_m3 -= got
        return got

    def heap_height_m(self) -> float:
        """A free-standing cone of it, at the angle it will stand at.

        Not decorative: it is how much room the heap takes, and a heap
        that will not fit is a heap that has to be carried further."""
        v = self.loose_m3
        if v <= 0.0:
            return 0.0
        t = math.tan(math.radians(self.spec.repose_deg))
        # v = pi r^2 h / 3, with h = r tan(repose)
        r = (3.0 * v / (math.pi * t)) ** (1.0 / 3.0)
        return r * t

    def heap_footprint_m2(self) -> float:
        h = self.heap_height_m()
        if h <= 0.0:
            return 0.0
        r = h / math.tan(math.radians(self.spec.repose_deg))
        return math.pi * r * r


@dataclass
class Berm:
    """A wall of earth, laid straight or run around a position.

    Sized the way one really is: a top width you can stand a thing
    behind, a height, and slopes the material will actually hold. The
    cross-section follows from those, and the volume from the length --
    which is what tells you whether the spoil you have is enough."""
    identity: str = "station.berm"
    kind: str = "hardpan"
    length_m: float = 0.0
    height_m: float = 1.6
    top_width_m: float = 1.0
    built_m3: float = 0.0            # in-situ equivalent placed so far
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def spec(self) -> SpoilClass:
        return spoil_class(self.kind)

    @property
    def slope_run_m(self) -> float:
        """How far out each face has to go to stand up."""
        return self.height_m / math.tan(math.radians(self.spec.repose_deg))

    @property
    def base_width_m(self) -> float:
        return self.top_width_m + 2.0 * self.slope_run_m

    @property
    def cross_section_m2(self) -> float:
        return 0.5 * (self.top_width_m + self.base_width_m) * self.height_m

    @property
    def required_m3(self) -> float:
        """In-situ volume needed, allowing for it compacting when placed."""
        placed = self.cross_section_m2 * max(0.0, self.length_m)
        return placed / max(0.05, self.spec.compacted_factor)

    @property
    def complete_frac(self) -> float:
        need = self.required_m3
        return 1.0 if need <= 0.0 else min(1.0, self.built_m3 / need)

    @property
    def stops_small_arms(self) -> bool:
        """Whether it is actually thick enough to be cover.

        The top width is what a round crosses when somebody is sheltering
        directly behind it, so that is the number that has to beat the
        material's stopping thickness -- not the base width, which is
        what makes a berm look thicker than it is."""
        return self.top_width_m >= self.spec.stopping_m

    def build_from(self, pile: DirtPile, in_situ_m3: float) -> float:
        """Move spoil out of the heap and into the wall."""
        want = min(max(0.0, float(in_situ_m3)), max(0.0, self.required_m3 - self.built_m3))
        got = pile.take(want)
        self.built_m3 += got
        return got

    def need(self, pile: "DirtPile | None" = None):
        """An unfinished berm is a job, and it says whether the material
        for it is even on site."""
        import servicing as sv
        if self.complete_frac >= 1.0:
            return None
        short = self.required_m3 - self.built_m3
        have = pile.in_situ_m3 if pile is not None else 0.0
        return sv.Need(
            identity=self.identity, want="build-berm", position=tuple(self.position),
            quantity=short, unit="m3", matches="spoil",
            urgency=1.0, minutes=min(600.0, short * 20.0), skill="labourer",
            label=(f"{self.identity} {self.complete_frac * 100:.0f}% built, "
                   f"{short:.1f} m3 short, {have:.1f} m3 on the pile"),
            why="spoil is protective material already paid for in labour; moving it "
                "twice is the mistake, so it should go where it is wanted the first "
                "time")


def berm_around(perimeter_m: float, kind: str = "hardpan", height_m: float = 1.6,
                top_width_m: float = 1.0, identity: str = "station.berm.perimeter") -> Berm:
    """A berm run right around a position."""
    return Berm(identity=identity, kind=kind, length_m=max(0.0, float(perimeter_m)),
                height_m=height_m, top_width_m=top_width_m)


def spoil_from_excavation(exc, pile: DirtPile) -> float:
    """Tip an excavation's spoil onto the pile.

    The hole and the heap are the same job seen from two ends, so the
    dig should never have to be told where the dirt goes."""
    moved = getattr(exc, "excavated_m3", 0.0) - getattr(exc, "_tipped_m3", 0.0)
    if moved > 0.0:
        pile.add(moved)
        setattr(exc, "_tipped_m3", getattr(exc, "_tipped_m3", 0.0) + moved)
    return max(0.0, moved)
