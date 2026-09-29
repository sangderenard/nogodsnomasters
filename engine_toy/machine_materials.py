"""One registry from a machine part's material string to what it is MADE OF.

`MachinePart.material` is a free string today -- "steel-plate",
"hardened-steel", "cast-iron" -- and those strings resolve only to the
render table (engine_mesh), the EM table (em_materials) and the armour
table (ballistics).  None of those is a mechanical record.  Contact
response has to come from one: a modulus, a yield, a density for a
solid; a density and a viscosity for a fluid; a bearing pressure and a
modulus for ground.  And it has to come with a declared NATURE, because
a structural solid pushes back, a granular medium bears and heaps, and a
fluid flows, and no number tells the three apart.

NOTHING HERE IS INVENTED.  A string is either bound to one record that
already exists in this repository, or it is UNDECLARED, and asking an
undeclared material for its mechanics raises rather than defaulting.
The records this registry binds to, and nothing else:

    milspec.MATERIAL_BY_KEY       StructuralMaterial   alloys, to spec
    woodshop.KILN_DRY_PINE_STUD   OrthotropicWood      the one wood
    fluids.BY_KEY                 Fluid                liquids and gases
    earthworks.SPOIL              SpoilClass           broken, loose earth
    feet.GROUNDS                  Ground               what a pad bears on

Adding a material is adding a row to one of those registries -- it is
then declared here under its own key -- or adding one line to
`BINDINGS` that puts an existing record under the string a machine
already uses.  `UNDECLARED` lists the strings machines carry today that
have no record; each row names the existing records it MIGHT be bound
to, and choosing is the owner's, not this module's.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import earthworks
import feet
import fluids
import milspec


class Nature(str, Enum):
    """How a body of this material meets another body."""
    STRUCTURAL_SOLID = "structural-solid"   # holds shape; elastic, then yields
    GRANULAR = "granular"                   # bears and heaps; no shape of its own
    FLUID = "fluid"                         # flows; no shear stiffness at all


class UndeclaredMaterialError(LookupError):
    """A material string with no mechanical record was asked for one."""


@dataclass(frozen=True)
class DeclaredMaterial:
    material: str
    nature: Nature
    source: str                       # where the record lives, for the reader
    fetch: Callable[[], object]       # the record itself; lazy so a light
                                      # registry is not paid for with sympy

    @property
    def record(self):
        return self.fetch()


@dataclass(frozen=True)
class UndeclaredMaterial:
    """A string a machine carries that no mechanical record answers for."""
    material: str
    used_by: tuple[str, ...] = ()     # modules whose parts carry it
    candidates: tuple[str, ...] = ()  # existing records it MIGHT be bound
                                      # to -- flagged for the owner, not chosen
    note: str = ""

    def _refuse(self, asked: str):
        hint = (f"; existing records it might be bound to: "
                f"{', '.join(self.candidates)}" if self.candidates else "")
        raise UndeclaredMaterialError(
            f"material {self.material!r} has no declared mechanical record, so "
            f"its {asked} cannot be answered. Declare it in machine_materials."
            f"{self.note and ' ' + self.note}{hint}")

    @property
    def nature(self) -> Nature:
        self._refuse("nature")

    @property
    def record(self):
        self._refuse("mechanical record")


# ---------------------------------------------------------------------
#  THE TABLE
# ---------------------------------------------------------------------
# Every alloy in milspec is a structural solid, every row in fluids is a
# fluid, and spoil is by definition broken loose earth, so those three
# registries declare themselves by key.  Ground rows are the same
# record family with two different natures, so each is named.
_GROUND_NATURE: dict[str, Nature] = {
    "soft-clay": Nature.GRANULAR,
    "firm-soil": Nature.GRANULAR,
    "compacted-gravel": Nature.GRANULAR,
    # bound pavement and reinforced slab hold their shape; the pad bears
    # on them, it does not sink into them
    "tarmac": Nature.STRUCTURAL_SOLID,
    "concrete-slab": Nature.STRUCTURAL_SOLID,
}


def _pine():
    # woodshop imports sympy and the Turing runtime; the record is fetched
    # only when a part made of it asks
    import woodshop
    record = woodshop.KILN_DRY_PINE_STUD
    if record.identity != _PINE_KEY:
        raise UndeclaredMaterialError(
            f"woodshop.KILN_DRY_PINE_STUD is now {record.identity!r}; this "
            f"registry binds {_PINE_KEY!r}. Update the binding.")
    return record


_PINE_KEY = "kiln-dry-pine-stud-v0"      # woodshop.KILN_DRY_PINE_STUD.identity

#: The strings machines carry that ARE an existing record, by that
#: record's own key.  One line per record family; add a line here to put
#: an existing record under a string a machine already uses.
BINDINGS: tuple[DeclaredMaterial, ...] = (
    *(DeclaredMaterial(m.key, Nature.STRUCTURAL_SOLID,
                       f"milspec.MATERIAL_BY_KEY[{m.key!r}]", lambda m=m: m)
      for m in milspec.MATERIALS),
    DeclaredMaterial(_PINE_KEY, Nature.STRUCTURAL_SOLID,
                     "woodshop.KILN_DRY_PINE_STUD", _pine),
    *(DeclaredMaterial(f.key, Nature.FLUID,
                       f"fluids.BY_KEY[{f.key!r}]", lambda f=f: f)
      for f in fluids.FLUIDS),
    *(DeclaredMaterial(s.key, Nature.GRANULAR,
                       f"earthworks.SPOIL[{s.key!r}]", lambda s=s: s)
      for s in earthworks.SPOIL.values()),
    *(DeclaredMaterial(g.key, _GROUND_NATURE[g.key],
                       f"feet.GROUNDS[{g.key!r}]", lambda g=g: g)
      for g in feet.GROUNDS.values() if g.key in _GROUND_NATURE),
)

#: The strings machines carry TODAY that no record answers for.  Each row
#: says where it is used and which existing records it might be bound to.
#: A string that could be two alloys lists both; the choice is declared
#: by moving the string into BINDINGS, not by picking here.
UNDECLARED: tuple[UndeclaredMaterial, ...] = (
    UndeclaredMaterial("steel-plate", ("machines (MachinePart default)", "station_powerplant",
                                       "electrical_hardware", "starter_garage"),
                       candidates=("a36", "hy80"),
                       note="em_materials calls it mild steel plate; A36 is the mild "
                            "structural steel, HY-80 the hull plate."),
    UndeclaredMaterial("steel-pipe", ("machines (MachineLine default)",),
                       candidates=("a36",),
                       note="ASME B36.10 pipe steel (A53/A106) is not in milspec; A36 is "
                            "the nearest plain carbon steel there."),
    UndeclaredMaterial("steel-shaft", ("machines",), candidates=("4340qt", "4130n")),
    UndeclaredMaterial("steel-tube", ("station_powerplant",), candidates=("4130n", "a36"),
                       note="milspec.TubeSection defaults to 4130n; a shelter frame is "
                            "not a roll cage."),
    UndeclaredMaterial("hardened-steel", ("machines", "woodshop"), candidates=("4340qt", "300m")),
    UndeclaredMaterial("gun-steel", ("machines",), candidates=("300m", "4340qt"),
                       note="milspec notes 300M for gun tubes."),
    UndeclaredMaterial("armour-plate", ("machines",), candidates=("rha",)),
    UndeclaredMaterial("pressed-steel", ("machines",), candidates=("a36",)),
    UndeclaredMaterial("cast-iron", ("machines",),
                       note="No cast iron record exists. materials_handling.SHEAR_STRENGTH_PA "
                            "and burst.SOLID_DENSITY hold one number each, not a record."),
    UndeclaredMaterial("stainless-steel", ("station_cooling",),
                       note="em_materials has a 304 EM row; no mechanical record."),
    UndeclaredMaterial("aluminium-cast", ("station_cooling",),
                       note="milspec's aluminiums (6061-T6, 7075-T6) are wrought, not "
                            "casting alloys."),
    UndeclaredMaterial("aluminium", ("electrical_hardware (catalogue shell)",),
                       candidates=("6061t6", "7075t6"),
                       note="Unspecified aluminium shell."),
    UndeclaredMaterial("brass", ("electrical_hardware (catalogue contact)",)),
    UndeclaredMaterial("copper", ("electrical_hardware",)),
    UndeclaredMaterial("nylon", ("electrical_hardware (catalogue shell)",)),
    UndeclaredMaterial("impact-modified-nylon", ("electrical_hardware (catalogue shell)",)),
    UndeclaredMaterial("canvas", ("station_powerplant",),
                       note="ballistics.MATERIAL_PROFILES['canvas'] is the armour table, "
                            "not a mechanical record."),
    UndeclaredMaterial("glass-filled-structural-resin", ("woodworking_joints",)),
    UndeclaredMaterial("medium-carbon-steel", ("woodworking_joints",),
                       note="4130/4340 are low-alloy steels; no plain medium-carbon record."),
    UndeclaredMaterial("kiln-dry-hardwood-handle", ("woodshop",),
                       note="The only wood record is pine, a softwood."),
    UndeclaredMaterial("unresolved-material", ("starter_garage", "electrical_hardware"),
                       note="The deliberate marker for absent product data; undeclared "
                            "by design, not an omission."),
    # Ground rows added to feet.GROUNDS without a nature in _GROUND_NATURE
    # land here rather than silently vanishing.
    *(UndeclaredMaterial(g.key, ("feet.GROUNDS",),
                         note="A Ground row with no nature declared in _GROUND_NATURE.")
      for g in feet.GROUNDS.values() if g.key not in _GROUND_NATURE),
)

_DECLARED: dict[str, DeclaredMaterial] = {d.material: d for d in BINDINGS}
_UNDECLARED: dict[str, UndeclaredMaterial] = {u.material: u for u in UNDECLARED}
if _DECLARED.keys() & _UNDECLARED.keys():
    raise ValueError(f"material listed as both declared and undeclared: "
                     f"{sorted(_DECLARED.keys() & _UNDECLARED.keys())}")


# ---------------------------------------------------------------------
#  THE TWO QUESTIONS
# ---------------------------------------------------------------------
def resolve(material: str) -> DeclaredMaterial | UndeclaredMaterial:
    """The declaration behind a part's material string.  Never raises: a
    string nobody has listed is still an UndeclaredMaterial."""
    key = str(material)
    if key in _DECLARED:
        return _DECLARED[key]
    return _UNDECLARED.get(key) or UndeclaredMaterial(
        key, note="Not listed in machine_materials.UNDECLARED either.")


def mechanical_record(material: str):
    """The existing record a part of this material is made of:
    StructuralMaterial, OrthotropicWood, Fluid, SpoilClass or Ground.
    Raises UndeclaredMaterialError for anything unbound."""
    return resolve(material).record


def nature(material: str) -> Nature:
    """Structural solid, granular or fluid.  Raises UndeclaredMaterialError
    for anything unbound."""
    return resolve(material).nature


def declared_materials() -> tuple[DeclaredMaterial, ...]:
    """Every material string this registry can answer for."""
    return BINDINGS


def undeclared_materials() -> tuple[UndeclaredMaterial, ...]:
    """Every material string machines carry that the owner has yet to declare."""
    return UNDECLARED
