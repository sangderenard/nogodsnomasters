"""The gear case, as one primitive.

A manual gearbox, a transaxle, a transfer case, a differential, a final
drive and a propeller reduction box are the same machine wearing
different jobs: a sealed housing with meshing gears in it, a measured
charge of oil, a breather so it does not pressurise itself when it gets
hot, and -- if it works hard enough -- a cooler. They fail the same
ways too: lose the oil and the gears weld themselves together, block
the breather and the seals push out, overheat it and the oil stops
carrying the load between the teeth.

So they are one thing here, `GearCase`, with a kind. Each one declares
its own real oil charge and gets it from the same fluid registry every
other fluid comes from (fluids.py), which is what makes a differential
leak, spray, pour, colour its own streaks and appear on the tank gauges
without any of those systems knowing what a differential is.

An AUTOMATIC transmission is NOT one of these, and putting it here
would have been the wrong abstraction. What defines this family is gear
oil: an EP lubricant whose only job is to survive the tooth face. An
automatic's fluid is a WORKING fluid -- it carries torque through the
converter and applies the clutch packs through a valve body -- so the
machine is hydraulic first and a case of gears second. It lives in
automatic_transmission.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GearCaseKind:
    key: str
    label: str
    fluid: str                  # a fluids.py key
    capacity_l: float           # a typical real charge for this kind
    cooler_default: bool


GEAR_CASE_KINDS: tuple[GearCaseKind, ...] = (
    GearCaseKind("manual-transmission", "manual gearbox", "gear-oil", 2.2, False),
    GearCaseKind("transaxle", "transaxle", "gear-oil", 3.4, False),
    GearCaseKind("transfer-case", "transfer case", "gear-oil", 1.6, False),
    GearCaseKind("differential", "differential", "gear-oil", 1.9, False),
    GearCaseKind("reduction-gearbox", "reduction gearbox", "gear-oil", 6.0, False),
)
BY_KIND = {k.key: k for k in GEAR_CASE_KINDS}


@dataclass
class GearCase:
    """One real sealed case of gears on a built machine."""
    identity: str               # the graph node it is
    kind: str                   # a GEAR_CASE_KINDS key
    fluid: str
    capacity_l: float
    cooler: bool

    @property
    def label(self) -> str:
        return BY_KIND[self.kind].label if self.kind in BY_KIND else self.kind

    @property
    def short_label(self) -> str:
        """A gauge label: "trans", "t-case", "diff"."""
        return {"manual-transmission": "gearbox",
                "transaxle": "transax", "transfer-case": "t-case",
                "differential": "diff", "reduction-gearbox": "red-box"}.get(self.kind, "gears")


def _kind_for(node: dict) -> GearCaseKind | None:
    """Which kind of case this node is -- by DECLARATION.

    A node says `gear_case="transfer-case"` when it is built, and that
    is the only thing consulted here. The earlier version of this
    matched substrings of the node's identity, which is not
    identification at all: it made a breather called
    `powertrain.transfer_case_breather` match the very rule that created
    it, so the machine grew a second transfer case out of its own vent
    fitting. A part knows what it is; nothing should have to guess from
    its name."""
    return BY_KIND.get(str(node.get("gear_case") or ""))


def discover(graph: dict, engine) -> list[GearCase]:
    """Every gear case this build actually carries, read off the graph.

    Every one of them is here because it SAID so when it was built --
    a transverse build's transaxle, a four-wheel-drive build's transfer
    case, a turboprop's reduction box. Nothing is inferred from a
    name."""
    if not graph:
        return []
    tr = getattr(engine, "transmission", None)
    declared_l = float(getattr(tr, "fluid_capacity_l", 0.0) or 0.0)
    cooler_override = getattr(tr, "cooler_fitted", None)
    out: list[GearCase] = []
    for n in graph.get("nodes", []):
        kind = _kind_for(n)
        if kind is None:
            continue
        is_main = kind.key == "manual-transmission"
        out.append(GearCase(
            identity=n.get("identity", ""), kind=kind.key, fluid=kind.fluid,
            capacity_l=float(n.get("fluid_volume_l") or 0.0)
            or (declared_l if (is_main and declared_l > 0.0) else kind.capacity_l),
            cooler=(bool(cooler_override) if (is_main and cooler_override is not None)
                    else kind.cooler_default)))
    return out
