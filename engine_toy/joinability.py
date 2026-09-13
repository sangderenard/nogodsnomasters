"""What can actually be joined to what, and by which process.

A capability matrix, not a preference list. Several of these are hard
constraints that decide a design rather than trade against it:

  7075-T6 CANNOT BE FUSION WELDED. It hot-cracks along the weld and
  nothing about technique fixes it. Aircraft aluminium of this grade is
  bolted, riveted or bonded, and a design that assumed it could be
  welded is not a design that can be built.

  6061-T6 CAN be welded and LOSES ITS TEMPER doing it. The heat-affected
  zone drops from 276 MPa to something near 165, permanently, unless the
  whole part is solution treated and aged afterwards. So a welded 6061
  structure is designed around the weakest metal being at the joints.

  TITANIUM NEEDS AN INERT ATMOSPHERE, not just a gas cup. It takes up
  oxygen and nitrogen above about 500 C and turns brittle; a trailing
  shield or a purge chamber is the difference between a joint and a
  crack.

  HY-80 AND ARMOUR NEED PREHEAT AND LOW-HYDROGEN CONSUMABLES. Without
  both, the heat-affected zone cracks days later, cold, from hydrogen.

  SILVER BRAZE DOES NOT JOIN ALUMINIUM. It needs an aluminium-silicon
  filler and a flux that attacks the oxide, and even then the working
  window is tens of degrees wide.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    """Whether a process can join a material, and at what cost."""
    possible: bool
    #: what fraction of the parent's strength the JOINT REGION retains
    joint_efficiency: float
    requires: tuple = ()
    note: str = ""


#: material -> process -> capability. Blank entries mean "not done".
JOINABILITY = {
    "a36": {
        "fusion-weld": Capability(True, 1.00, (), "as strong as the parent; the default"),
        "braze": Capability(True, 0.45, ("clean, close gap",)),
        "adhesive": Capability(True, 0.25, ("abraded, degreased",)),
        "bolted": Capability(True, 0.85, ()),
        "hot-rivet": Capability(True, 0.80, (), "the traditional structural joint"),
    },
    "4130n": {
        "fusion-weld": Capability(True, 0.95, ("normalise after welding",),
                                  "full strength only if normalised; as-welded "
                                  "the HAZ is hard and brittle"),
        "braze": Capability(True, 0.40, ("close gap", "flux"),
                            "traditional for thin-wall aircraft tube: no HAZ, "
                            "no distortion, and a third of the strength"),
        "adhesive": Capability(True, 0.20, ()),
        "bolted": Capability(True, 0.85, ()),
        "hot-rivet": Capability(True, 0.75, ()),
    },
    "4340qt": {
        "fusion-weld": Capability(True, 0.70, ("preheat 300 C", "low hydrogen",
                                               "temper after"),
                                  "welding undoes the quench and temper; the "
                                  "joint is never the parent again"),
        "braze": Capability(True, 0.25, (), "the braze is far below this steel"),
        "bolted": Capability(True, 0.90, ()),
    },
    "hy80": {
        "fusion-weld": Capability(True, 0.95, ("preheat", "low hydrogen",
                                               "controlled heat input"),
                                  "designed to be welded, and unforgiving if "
                                  "the procedure is not followed"),
        "bolted": Capability(True, 0.85, ()),
    },
    "hy100": {
        "fusion-weld": Capability(True, 0.90, ("preheat", "low hydrogen",
                                               "tight heat input window")),
        "bolted": Capability(True, 0.85, ()),
    },
    "rha": {
        "fusion-weld": Capability(True, 0.75, ("preheat", "low hydrogen",
                                               "austenitic filler"),
                                  "armour is welded with stainless filler so the "
                                  "joint stays tough where the plate is hard"),
        "bolted": Capability(True, 0.70, ()),
    },
    "300m": {
        "fusion-weld": Capability(False, 0.0, (),
                                  "not welded in service: the strength comes from "
                                  "heat treatment that welding destroys"),
        "bolted": Capability(True, 0.90, ()),
        "adhesive": Capability(True, 0.10, ()),
    },
    "6061t6": {
        "fusion-weld": Capability(True, 0.60, ("ER4043 or ER5356",
                                               "re-solution and age for full strength"),
                                  "WELDABLE BUT IT LOSES ITS TEMPER: the heat-"
                                  "affected zone falls to about 165 MPa and stays "
                                  "there unless the whole part is heat treated"),
        "braze": Capability(True, 0.30, ("Al-Si filler", "aggressive flux"),
                            "narrow working window; silver braze does not wet it"),
        "adhesive": Capability(True, 0.35, ("anodised or etched",),
                               "bonding is genuinely competitive on aluminium"),
        "bolted": Capability(True, 0.80, ()),
        "hot-rivet": Capability(True, 0.85, (), "cold-driven; aluminium is riveted, "
                                               "not hot-driven"),
    },
    "7075t6": {
        "fusion-weld": Capability(False, 0.0, (),
                                  "HOT CRACKS. This grade is not fusion welded at "
                                  "all, by anyone, and no technique changes that"),
        "adhesive": Capability(True, 0.35, ("anodised", "controlled bondline")),
        "bolted": Capability(True, 0.80, ()),
        "hot-rivet": Capability(True, 0.85, (), "cold-driven rivets: how aircraft "
                                               "structure in this alloy is assembled"),
    },
    "ti64": {
        "fusion-weld": Capability(True, 0.90, ("inert chamber or trailing shield",
                                               "backing purge"),
                                  "picks up oxygen above 500 C and turns brittle; "
                                  "the shielding is the process"),
        "adhesive": Capability(True, 0.25, ()),
        "bolted": Capability(True, 0.90, ()),
    },
}


def can_join(material: str, process: str) -> Capability:
    """What happens if you try. An absent entry is a refusal, not a
    silent default -- which matters, because the interesting answers
    here are the ones that say no."""
    entry = JOINABILITY.get(material, {}).get(process)
    if entry is None:
        return Capability(False, 0.0, (),
                          f"{process} is not a way of joining {material}")
    return entry


def matrix_lines(processes=("fusion-weld", "braze", "adhesive", "bolted", "hot-rivet")) -> list:
    """The whole table, as it would be read off a wall."""
    head = f"  {'material':10s}" + "".join(f"{p:>14s}" for p in processes)
    out = [head, "  " + "-" * (len(head) - 2)]
    for material in JOINABILITY:
        row = f"  {material:10s}"
        for process in processes:
            c = can_join(material, process)
            row += f"{('--' if not c.possible else f'{c.joint_efficiency * 100:.0f}%'):>14s}"
        out.append(row)
    out.append("    '--' means the process does not join that material at all;")
    out.append("    percentages are what the JOINT REGION retains of the parent.")
    return out
