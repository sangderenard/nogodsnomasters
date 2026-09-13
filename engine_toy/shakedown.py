"""Start at the bottom, load it, and upgrade only what broke.

THE MINIMUM THAT MEETS THE REQUIREMENT, found by asking the structure
instead of by guessing. Every member starts as the cheapest thing that
could conceivably work -- mild steel, thin wall, the simplest joint --
and the whole assembled object is loaded to what it is actually required
to survive. Whatever fails is replaced with the next thing up the
ladder, one step, and the load is applied again. Repeat until nothing
fails.

WHY THIS AND NOT A SIZING FORMULA. A sizing formula answers for one
member in isolation. A structure redistributes: stiffen one brace and
the load moves to its neighbour, which then fails instead. The only
honest way to find the minimum configuration is to load the REAL
ASSEMBLED OBJECT and watch where it actually goes, which is what this
does -- and it is why the answer frequently upgrades a member nobody
would have suspected while leaving the obvious one at mild steel.

MILSPEC AS AN EXPECTATION, NOT A MATERIAL. The requirement is stated as
performance -- survive this load case, with this margin -- and the
materials that satisfy it are the output. Specifying 4340 everywhere is
not a specification, it is an assumption with a number on it.

It is also meant to be quick. Each pass is one solve of an already-built
graph with a few edge records rewritten, so a shakedown is seconds, not
a study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from milspec import MATERIAL_BY_KEY, TubeSection, damage_record


#: Cheapest and weakest first. A shakedown walks UP this, one step at a
#: time, and the step it stops on is the answer.
STEEL_LADDER = ("a36", "4130n", "hy80", "hy100", "4340qt", "rha", "300m")

#: For anything where mass is the binding constraint instead of cost.
LIGHT_LADDER = ("6061t6", "7075t6", "ti64")

#: Wall thickness multipliers, walked when the alloy ladder runs out.
#: Going thicker is always available and always costs mass, so it is
#: the last resort rather than the first.
WALL_LADDER = (1.0, 1.25, 1.6, 2.0, 2.6)


@dataclass
class Expectation:
    """What the object is required to survive, in performance terms."""
    name: str = "40 mm firing, full recoil"
    #: the load case, as a prescribed displacement of named bodies
    displace: dict = field(default_factory=dict)
    #: how much margin on fracture demand counts as passing
    margin: float = 1.5
    #: members allowed to be at their limit (things designed to travel)
    exempt: tuple = ()

    def passes(self, demand: float) -> bool:
        return demand * self.margin <= 1.0


@dataclass
class Shakedown:
    """One structure, walked up from the bottom until it holds."""
    document: dict
    expectation: Expectation
    ladder: tuple = STEEL_LADDER
    max_passes: int = 40
    #: per-member current position on the ladders
    alloy_step: dict = field(default_factory=dict)
    wall_step: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    # ------------------------------------------------------------------
    def _structural_edges(self) -> list:
        return [e for e in self.document["edges"]
                if (e.get("damage") or {}).get("model")
                == "elastic-plastic-member-with-shear-fracture"]

    def start_at_the_bottom(self) -> None:
        """Every member to the weakest alloy and the thinnest wall."""
        for e in self._structural_edges():
            self.alloy_step.setdefault(e["identity"], 0)
            self.wall_step.setdefault(e["identity"], 0)
        self._respec_all()

    def _respec_all(self) -> None:
        for e in self._structural_edges():
            self._respec(e)

    def _respec(self, edge: dict) -> None:
        """Rewrite one member's section and material at its current step."""
        ident = edge["identity"]
        alloy = self.ladder[min(self.alloy_step[ident], len(self.ladder) - 1)]
        scale = WALL_LADDER[min(self.wall_step[ident], len(WALL_LADDER) - 1)]
        radius = float(edge.get("radius", 0.012))
        base_wall = max(0.0025, min(radius * 0.42, 0.0045 + radius * 0.16))
        section = TubeSection(outer_diameter_m=radius * 2.0,
                              wall_m=min(base_wall * scale, radius * 0.95),
                              material=alloy,
                              designation=f"{radius * 2000:.0f} OD x "
                                          f"{base_wall * scale * 1000:.1f} {alloy}")
        rest = float(edge.get("rest_length", 1.0))
        spring = edge["constraint"] in ("spring-damper", "linear-hydraulic-actuator",
                                        "oleo-recoil-slide")
        travels = spring or edge["constraint"] in ("single-axis-slider",
                                                   "prismatic-guide-strut")
        edge["damage"] = damage_record(section, rest, spring_like=spring, travels=travels)
        edge["material"] = alloy

    # ------------------------------------------------------------------
    def _solve(self) -> dict:
        from graph_physics import GraphPhysics
        gp = GraphPhysics(document=self.document)
        for ident, delta in self.expectation.displace.items():
            gp.displace(ident, delta)
        return gp.solve()

    def _upgrade(self, ident: str) -> str:
        """One step up, alloy first and wall only when the alloy runs
        out -- because a better alloy is free weight and a thicker wall
        never is."""
        if self.alloy_step[ident] < len(self.ladder) - 1:
            self.alloy_step[ident] += 1
            return f"{self.ladder[self.alloy_step[ident]]}"
        if self.wall_step[ident] < len(WALL_LADDER) - 1:
            self.wall_step[ident] += 1
            return f"wall x{WALL_LADDER[self.wall_step[ident]]}"
        return "EXHAUSTED"

    def run(self) -> dict:
        self.start_at_the_bottom()
        exhausted = set()
        for attempt in range(self.max_passes):
            solved = self._solve()
            demand = np.abs(solved["fracture_demand"])
            failing = []
            for i, ident in enumerate(solved["identities"]):
                if ident in self.expectation.exempt or ident in exhausted:
                    continue
                if not self.expectation.passes(float(demand[i])):
                    failing.append((ident, float(demand[i])))
            if not failing:
                return self._result(attempt, solved, True)
            # UPGRADE ONLY THE WORST. Upgrading everything that failed
            # in one pass overshoots, because relieving the worst member
            # moves load around and several of the others stop failing
            # on their own. One at a time is what finds the minimum.
            failing.sort(key=lambda kv: -kv[1])
            ident, worst = failing[0]
            moved = self._upgrade(ident)
            if moved == "EXHAUSTED":
                exhausted.add(ident)
                self.log.append(f"  {ident}: nothing left on the ladder "
                                f"(demand {worst:.2f}) -- the SHAPE is wrong, "
                                f"not the material")
                continue
            self.log.append(f"  pass {attempt + 1:2d}: {ident} -> {moved} "
                            f"(was at demand {worst:.2f})")
            self._respec(next(e for e in self._structural_edges()
                              if e["identity"] == ident))
        return self._result(self.max_passes, self._solve(), False)

    def _result(self, passes: int, solved: dict, ok: bool) -> dict:
        # HOW MUCH OF THE STRUCTURE WAS ACTUALLY TESTED. Without this the
        # tool will happily report that a mild-steel turret meets a
        # firing requirement, because 126 of its 129 members were never
        # loaded and unloaded members never fail. A pass is only worth
        # anything alongside its coverage.
        strained = int((np.abs(solved["axial_strain"]) > 1e-9).sum())
        exempt = set(self.expectation.exempt)
        tested = [abs(float(d)) for i, d in zip(solved["identities"],
                                                solved["fracture_demand"])
                  if i not in exempt]
        bill = {}
        mass = 0.0
        for e in self._structural_edges():
            d = e["damage"]
            bill[d["material"]] = bill.get(d["material"], 0) + 1
            mat = MATERIAL_BY_KEY[d["material"]]
            mass += d["section_area_m2"] * float(e.get("rest_length", 0.0)) \
                * mat.density_kg_m3
        return {"converged": ok, "passes": passes, "bill_of_materials": bill,
                "structural_mass_kg": mass, "log": self.log,
                "worst_demand": max(tested) if tested else 0.0,
                "worst_including_exempt": float(np.abs(solved["fracture_demand"]).max()),
                "members_loaded": strained,
                "members_total": len(solved["identities"]),
                "upgraded": {k: self.ladder[v] for k, v in self.alloy_step.items()
                             if v > 0}}

    def report(self, result: dict) -> list:
        loaded, total = result["members_loaded"], result["members_total"]
        coverage = loaded / max(total, 1)
        out = [f"  {self.expectation.name}: "
               f"{'MET' if result['converged'] else 'NOT MET'} "
               f"after {result['passes']} pass(es)",
               f"    structural mass {result['structural_mass_kg']:7.1f} kg, "
               f"worst demand {result['worst_demand']:.3f}",
               f"    COVERAGE {loaded} of {total} members carried any load "
               f"({coverage * 100:.0f}%)"]
        if coverage < 0.5:
            out.append("    -- THIS RESULT IS NOT A PASS. Most of the structure was")
            out.append("       never loaded, because prescribing displacement only")
            out.append("       strains the members touching it. Load propagates")
            out.append("       through an EQUILIBRIUM SOLVE, and there is not one yet.")
        out.append("    bill of materials:")
        for alloy, count in sorted(result["bill_of_materials"].items(),
                                   key=lambda kv: -kv[1]):
            out.append(f"      {count:4d} x {MATERIAL_BY_KEY[alloy].label}")
        if result["upgraded"]:
            out.append(f"    {len(result['upgraded'])} member(s) needed more than "
                       f"the base material:")
            for ident, alloy in sorted(result["upgraded"].items()):
                out.append(f"      {ident:36s} -> {alloy}")
        return out
