"""EVERY ARTICULATION, AND WHETHER ANYTHING ACTUALLY SOLVES IT.

    python articulation_audit.py

A joint in this graph can be in one of four states, and only the last
one is finished:

    DECLARED     the edge exists and names a constraint
    RELEASED     the frame solve knows which freedoms it does not carry
    LAWED        it carries the parameters its own force law needs, and
                 a law exists that reads them
    WRENCHED     the load it negotiates is written back onto the graph
                 as a force and a moment at a stated point, so the beam
                 solve and everything downstream sees it

The gap between DECLARED and WRENCHED is where a model quietly stops
being physics and becomes a drawing. This prints the gap rather than
asserting it is closed, because it has twice been asserted closed here
and twice was not: thirty-nine sliders were solved as welded bars, and
a ring negotiating 218 kN of hold-down had none of it anywhere in the
document.

WHAT COUNTS AS A LAW is a declaration, not a guess -- an element has
one when it carries the parameters that law needs. A spring with no
rate is a name.
"""
from __future__ import annotations

import collections

from joints import MEMBER_CONSTRAINTS, member_constraint


#: Which module owns each articulating constraint's force law, and what
#: that law needs to be given. An empty requirement means the law takes
#: only geometry and state.
#: WHAT PISTON IS THIS, asked of the edge rather than of its constraint
#: spelling. `linear-hydraulic-actuator` is how a thing MOVES; it says
#: nothing about what the thing IS, and this graph has four different
#: devices under that one spelling -- an outrigger jack, an elevation
#: ram, a fine-aim twitch leg and a magnetorheological damper. The first
#: version of this table demanded MR parameters from all of them and
#: reported twenty-three perfectly well specified jacks as unlawed,
#: which is the same string-matching mistake the joint registry exists
#: to end, made one level up.
#:
#: So the key is the edge's own declared `kind`, falling back to the
#: constraint only when nothing finer is declared. The value is the
#: class in the keeper system that models it, and what that class needs.
PISTON_LAWS = {
    "outrigger-lift-jack": ("actuators.LinearActuator",
                            ("bore_m", "rod_m")),
    "commanded-rest-length-elevation-ram": ("actuators.LinearActuator",
                                            ("bore_m", "rod_m")),
    "fine-aim-twitch-actuator": ("actuators.LinearActuator",
                                 ("bore_m", "rod_m")),
    "magnetorheological": ("actuators.MagnetorheologicalDamper",
                           ("yield_min_pa", "yield_max_pa", "gap_m")),
}

#: Falls back to these when an edge declares no finer `kind`.
FORCE_LAWS = {
    "spring-damper": ("actuators", ("spring_rate_n_per_m",)),
    "oleo-recoil-slide": ("actuators.OleoStrut", ("orifice_area_m2",)),
    "linear-hydraulic-actuator": ("actuators.LinearActuator",
                                  ("bore_m", "rod_m")),
    "belleville-preload-stack": ("actuators", ()),
    "preloaded-captive-body-retainer-spring": ("actuators", ()),
    "single-axis-slider": (None, ()),          # a guide: carries no axial law
    "prismatic-guide-strut": (None, ()),
    "greased-thrust-track": ("ring_joints.MatedPlanesRing",
                             ("track_radius_m", "track_width_m")),
    "slew-thrust-roller": ("ring_joints.MatedPlanesRing",
                           ("roller_diameter_m", "roller_length_m")),
    "slew-holddown-roller": ("ring_joints.MatedPlanesRing",
                             ("roller_diameter_m",)),
    "captive-pinion-mesh": ("ring_joints.PlanetaryRingMesh",
                            ("ring_teeth", "module_m")),
    "pinion-carrier-bearing": (None, ()),
    "gimbal-yaw-bearing": (None, ()),
    "gimbal-pitch-bearing": (None, ()),
    "pinned-trunnion-mount": (None, ()),
    "spherical-thrust-seat": (None, ()),
    "actuated-damped-clutch-gimbal-base": ("couplings", ()),
    "tension-limit-strap": (None, ("release_force_n",)),
}


def law_for(edge: dict):
    """The law that owns this member, by what it DECLARES itself to be."""
    kind = edge.get("kind")
    if kind in PISTON_LAWS:
        return kind, PISTON_LAWS[kind]
    # PART_ROLE IS A DECLARATION; AN IDENTITY IS A NAME. Matching the
    # identity as well looked like a harmless fallback and immediately
    # claimed `turret.absorber.land.magnetorheological` -- a plain
    # bearing land in a bore -- as a magnetorheological damper, then
    # reported it as missing the yield stresses a bearing land has no
    # business carrying. The registry exists to stop exactly this.
    role = str(edge.get("part_role", ""))
    for tag, entry in PISTON_LAWS.items():
        if role and tag in role:
            return tag, entry
    return edge["constraint"], FORCE_LAWS.get(edge["constraint"], (None, ()))


def _wrenched(edge: dict) -> bool:
    """Has a negotiated load been written back onto this edge?

    Not "could it be" -- has it. A wrench is a force and a moment at a
    point, and until one is on the edge the solve downstream is working
    from geometry alone."""
    w = edge.get("wrench")
    if isinstance(w, dict):
        return any(any(abs(float(c)) > 0.0 for c in w.get(k, ()))
                   for k in ("force", "moment"))
    return bool(edge.get("negotiated_force_n") is not None
                or edge.get("negotiated_moment_nm") is not None)


def audit(document: dict) -> dict:
    """One row per constraint spelling that articulates."""
    rows = {}
    for e in document["edges"]:
        key = e["constraint"]
        mc = MEMBER_CONSTRAINTS.get(key)
        if mc is None or mc.routed:
            continue
        if mc.welded and key not in FORCE_LAWS:
            continue          # a plain welded member is not an articulation
        label, (owner, needs) = law_for(e)
        key = label
        row = rows.setdefault(key, {
            "count": 0, "released": bool(mc.freedoms()),
            "freedoms": ",".join(mc.freedoms()) or "-",
            "law_owner": owner, "needs": needs,
            "lawed": 0, "wrenched": 0, "in_frame": 0, "examples": [],
        })
        row["count"] += 1
        if all(n in e for n in needs):
            row["lawed"] += 1
        elif len(row["examples"]) < 2:
            row["examples"].append(
                (e["identity"], [n for n in needs if n not in e]))
        if _wrenched(e):
            row["wrenched"] += 1
        if (e.get("damage") or {}).get("model"):
            row["in_frame"] += 1
    return rows


def report(document: dict) -> list:
    rows = audit(document)
    out = [f"{sum(r['count'] for r in rows.values())} articulating members "
           f"in {len(rows)} kinds", ""]
    out.append(f"  {'constraint':38s} {'n':>4s} {'free':>6s} "
               f"{'frame':>6s} {'lawed':>6s} {'wrench':>7s}  law")
    unfinished = []
    for key in sorted(rows):
        r = rows[key]
        law = r["law_owner"] or "-- guide only, no axial law --"
        out.append(f"  {key:38s} {r['count']:4d} {r['freedoms']:>6s} "
                   f"{r['in_frame']:6d} {r['lawed']:6d} {r['wrenched']:7d}  {law}")
        if r["law_owner"] and r["lawed"] < r["count"]:
            unfinished.append((key, "missing law parameters", r["examples"]))
        if r["law_owner"] and r["wrenched"] == 0:
            unfinished.append((key, "law exists but NOTHING is written back "
                                    "to the graph", []))
    if unfinished:
        out += ["", "NOT FINISHED:"]
        seen = set()
        for key, why, ex in unfinished:
            if (key, why) in seen:
                continue
            seen.add((key, why))
            out.append(f"  {key}: {why}")
            for ident, missing in ex:
                out.append(f"      e.g. {ident} lacks {', '.join(missing)}")
    return out


if __name__ == "__main__":
    import turret_production as tp
    doc = tp.balanced_station(bore_mm=20.0).as_document()
    print("\n".join(report(doc)))
