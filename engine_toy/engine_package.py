"""The crate engine: the real installable unit a toy-designed engine
becomes for the game.

Not a black box -- a real crate engine. A prism (bounding volume)
carrying a declared, finite set of real fluid/mechanical connectors on
its boundary, with an internal system that still resolves its own
constituent part identities (so damage/service/interaction can reach a
real sub-part, not just an opaque blob), even though it presents to
the game as one baked object. Some parts ship inside the crate
(BUILT_IN). Some parts the vehicle/game has to supply
(SUPPLIED_ELSEWHERE) -- the fuel tank is the settled real example (see
drivetrain_graph.py's own fuel-circuit comment: a real fuel tank is
chassis-mounted, never bolted to the engine itself).

Built entirely from the SAME real graph build_drivetrain_graph already
produces (the same one the toy shares with the piston/turbine/
atmospheric simulation, and a real structural PREFIX of a full
vehicle's own powertrain graph -- see ENGINE_TOY_ARCHITECTURE_NOTES.md
at the repo root for the full design rationale and why this is a
correlation, not a translation). No new node/edge vocabulary is
invented here -- this module only classifies and measures what the
graph already declares.

The part-sourcing classification is NOT settled beyond the fuel tank/
pump today -- every other node defaults to BUILT_IN until a real case
is worked out (coolant, starter battery, exhaust, pneumatic lines,
...). Extend _SOURCING_OVERRIDES as those get decided; don't guess
ahead of it here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from engines import Engine
import engine_mounts
from drivetrain_graph import build_drivetrain_graph


class PartSourcing(Enum):
    BUILT_IN = "built-in"                      # ships inside the crate/prism
    SUPPLIED_ELSEWHERE = "supplied-elsewhere"   # the vehicle/game provides it


# Real, settled classification, by node-identity prefix (dotted-path
# match: "fuel.tank" matches "fuel.tank" and "fuel.tank.anything").
# Only the fuel side is settled -- everything else defaults to
# BUILT_IN, not because it's assumed correct, but because guessing the
# rest ahead of real guidance would bake a wrong answer into the crate.
# fuel.gas_main_connection covers the Otto-Langen's real coal-gas
# utility-main fitting -- no tank/pump to classify (there's no
# depletable reservoir, see drivetrain_graph.py's own comment), but the
# real physical connection point is still supplied-elsewhere: a
# municipal gas main is never part of the engine itself, for the same
# real reason a car's fuel tank isn't, just a different real cause.
_SOURCING_OVERRIDES: tuple[tuple[str, PartSourcing], ...] = (
    ("fuel.tank", PartSourcing.SUPPLIED_ELSEWHERE),
    ("fuel.pump", PartSourcing.SUPPLIED_ELSEWHERE),
    ("fuel.gas_main_connection", PartSourcing.SUPPLIED_ELSEWHERE),
    # A real crate engine ships without its eventual transmission --
    # same real reason it ships without a fuel tank: the buyer pairs it
    # with whatever gearbox/transfer case their own build calls for,
    # not one baked in by the engine seller. EnginePackage.build
    # (include_transmission=True) overrides these back to BUILT_IN for
    # players who deliberately want the transmission baked into the
    # same crate instead.
    ("powertrain.transmission", PartSourcing.SUPPLIED_ELSEWHERE),
    ("powertrain.transfer_case", PartSourcing.SUPPLIED_ELSEWHERE),
    ("powertrain.direct_drive_bypass", PartSourcing.SUPPLIED_ELSEWHERE),
    ("mount.transmission_left", PartSourcing.SUPPLIED_ELSEWHERE),
    ("mount.transmission_right", PartSourcing.SUPPLIED_ELSEWHERE),
    ("mount.transfer_case_left", PartSourcing.SUPPLIED_ELSEWHERE),
    ("mount.transfer_case_right", PartSourcing.SUPPLIED_ELSEWHERE),
)

_TRANSMISSION_IDENTITY_PREFIXES = (
    "powertrain.transmission", "powertrain.transfer_case", "powertrain.direct_drive_bypass",
    "mount.transmission_", "mount.transfer_case_",
)


def _is_transmission_identity(identity: str) -> bool:
    return any(identity == p or identity.startswith(p) for p in _TRANSMISSION_IDENTITY_PREFIXES)


def part_sourcing(node_identity: str) -> PartSourcing:
    for prefix, sourcing in _SOURCING_OVERRIDES:
        if node_identity == prefix or node_identity.startswith(prefix + "."):
            return sourcing
    return PartSourcing.BUILT_IN


@dataclass(frozen=True)
class Connector:
    """One real point on the crate's own boundary where a supplied-
    elsewhere part attaches. This is the real graph edge the builder
    already declares between a BUILT_IN node and a SUPPLIED_ELSEWHERE
    one (e.g. fuel.tank_to_pump) -- not a newly invented port."""
    identity: str                          # the real graph edge identity
    kind: str                              # the edge's own real "constraint"
    built_in_node: str
    supplied_node: str
    position: tuple[float, float, float]   # the supplied-elsewhere node's own real position


@dataclass(frozen=True)
class Prism:
    """The crate's own real bounding volume -- the min/max corners of
    every BUILT_IN node's own real reference_position, in the graph's
    own local frame. A supplied-elsewhere part (the chassis-mounted
    fuel tank) doesn't count toward the crate's own physical footprint;
    it lives outside the prism by definition."""
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]

    @property
    def size_m(self) -> tuple[float, float, float]:
        return tuple(hi - lo for lo, hi in zip(self.min_corner, self.max_corner))

    def contains_point(self, p: tuple[float, float, float], tol_m: float = 0.0) -> bool:
        return all(lo - tol_m <= p[i] <= hi + tol_m
                   for i, (lo, hi) in enumerate(zip(self.min_corner, self.max_corner)))


@dataclass(frozen=True)
class MountCorrelation:
    """One real engine mount point (engine_mounts.assign_mounting's own
    resolved technique/position, not a bounding-volume containment
    test -- see ENGINE_TOY_ARCHITECTURE_NOTES.md's own reasoning on why
    "does it fit the silhouette" is the wrong question) checked against
    a supplied bar cage's real structural node positions."""
    mount_name: str
    mount_position: tuple[float, float, float]
    matched_cage_node: str | None
    distance_m: float | None


@dataclass
class EnginePackage:
    """The real crate: an engine's own drivetrain graph, classified
    into built-in vs supplied-elsewhere parts, with its own real prism
    and connector list. This is the installable unit -- not the whole
    live simulation, which stays exactly what it already is
    (EngineCycleSim, stepped live or baked separately; see the
    architecture note's own "best-case target" section)."""
    engine: Engine
    prism: Prism
    connectors: list[Connector]
    built_in_node_ids: frozenset[str]
    supplied_node_ids: frozenset[str]
    mounting: list[engine_mounts.MountAssignment]

    @classmethod
    def build(cls, engine: Engine, install_context: str = "automotive",
             include_transmission: bool = False, transmission_mass_kg: float = 0.0) -> "EnginePackage":
        graph = build_drivetrain_graph(engine)
        nodes = graph["nodes"]
        edges = graph["edges"]
        node_by_id = {n["identity"]: n for n in nodes}

        def _sourcing(identity: str) -> PartSourcing:
            base = part_sourcing(identity)
            if include_transmission and base is PartSourcing.SUPPLIED_ELSEWHERE and _is_transmission_identity(identity):
                # the player deliberately baked the transmission into
                # this same crate -- it's real cargo of the crate now,
                # not a part the vehicle still has to supply
                return PartSourcing.BUILT_IN
            return base

        sourcing = {n["identity"]: _sourcing(n["identity"]) for n in nodes}
        built_in_ids = frozenset(i for i, s in sourcing.items() if s is PartSourcing.BUILT_IN)
        supplied_ids = frozenset(i for i, s in sourcing.items() if s is PartSourcing.SUPPLIED_ELSEWHERE)

        positions = [tuple(node_by_id[i]["reference_position"]) for i in built_in_ids
                    if "reference_position" in node_by_id[i]]
        if not positions:
            prism = Prism((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        else:
            mins = tuple(min(p[axis] for p in positions) for axis in range(3))
            maxs = tuple(max(p[axis] for p in positions) for axis in range(3))
            prism = Prism(mins, maxs)

        connectors: list[Connector] = []
        for e in edges:
            a, b = e.get("a"), e.get("b")
            if a in built_in_ids and b in supplied_ids:
                built_in_node, supplied_node = a, b
            elif b in built_in_ids and a in supplied_ids:
                built_in_node, supplied_node = b, a
            else:
                continue
            pos = tuple(node_by_id.get(supplied_node, {}).get("reference_position", [0.0, 0.0, 0.0]))
            connectors.append(Connector(
                identity=e["identity"], kind=e.get("constraint", "?"),
                built_in_node=built_in_node, supplied_node=supplied_node, position=pos))

        mounting = engine_mounts.assign_mounting(
            engine, install_context=install_context, include_transmission=include_transmission,
            transmission_mass_kg=transmission_mass_kg)

        return cls(engine=engine, prism=prism, connectors=connectors,
                   built_in_node_ids=built_in_ids, supplied_node_ids=supplied_ids, mounting=mounting)

    def correlate_mounts(self, cage_nodes: dict[str, tuple[float, float, float]],
                         tolerance_m: float = 0.05) -> list[MountCorrelation]:
        """Checks this engine's own real mount points against a
        supplied bar cage's real structural node positions. `cage_nodes`
        is whatever the game declares for a given vehicle's engine bay
        -- there is no real source for this in the toy yet, so callers
        without a real cage should pass an empty dict and expect every
        mount to come back unmatched, honestly, rather than a fake
        match."""
        results: list[MountCorrelation] = []
        for m in self.mounting:
            best_node, best_dist = None, None
            for cage_name, cage_pos in cage_nodes.items():
                dist = math.dist(m.position, cage_pos)
                if best_dist is None or dist < best_dist:
                    best_node, best_dist = cage_name, dist
            matched = best_node if (best_dist is not None and best_dist <= tolerance_m) else None
            results.append(MountCorrelation(
                mount_name=m.identity, mount_position=m.position,
                matched_cage_node=matched, distance_m=best_dist))
        return results
