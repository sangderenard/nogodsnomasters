"""SDF material removal, separation, and conservative object fission.

The material body remains one object while subtractive tool SDFs remove mass
from it. When that one mesh becomes multiple separate material islands, the
predecessor is retired and every island is minted as a new Machine. The world
then contains separate object nodes with no edge between them. Ancestry lives
in a lineage ledger and is never represented as a physical edge.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import ndimage

from machines import Machine, MachinePart
from sdf_geometry import CapsuleSdf, MeshSdfKernel


class MaterialForm(str, Enum):
    SOLID = "solid"
    CHIPS = "chips"


@dataclass(frozen=True)
class RetiredIdentity:
    identity: str
    reason: str
    successors: tuple[str, ...]


@dataclass(frozen=True)
class ObjectEdge:
    """One explicit physical relationship between two world object nodes."""
    identity: str
    a: str
    b: str
    kind: str
    attributes: dict = field(default_factory=dict)


class WorldObjectGraph:
    """Globally identified world objects plus only their physical edges.

    Node payloads remain the game's real objects (Machines in this scene).
    Lineage is deliberately stored outside ``edges``. Replacing a node after
    mesh separation creates no edge among its successors.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, object] = {}
        self.edges: dict[str, ObjectEdge] = {}
        self.lineage: dict[str, RetiredIdentity] = {}

    def add_node(self, identity: str, payload: object) -> None:
        if identity in self.nodes or identity in self.lineage:
            raise ValueError(f"world object identity already used: {identity}")
        self.nodes[identity] = payload

    def add_edge(self, edge: ObjectEdge) -> None:
        if edge.identity in self.edges:
            raise ValueError(f"world object edge already exists: {edge.identity}")
        if edge.a not in self.nodes or edge.b not in self.nodes:
            raise KeyError("both edge endpoints must be live world object nodes")
        self.edges[edge.identity] = edge

    def remove_edge(self, identity: str) -> ObjectEdge:
        return self.edges.pop(identity)

    def edges_between(self, a: str, b: str) -> tuple[ObjectEdge, ...]:
        pair = {a, b}
        return tuple(edge for edge in self.edges.values()
                     if {edge.a, edge.b} == pair)

    def retire_and_add(self, predecessor: RetiredIdentity,
                       successors: dict[str, object]) -> tuple[ObjectEdge, ...]:
        """Atomically replace one node without inventing successor edges."""
        if predecessor.identity not in self.nodes:
            raise KeyError(predecessor.identity)
        collisions = set(successors) & (set(self.nodes) | set(self.lineage))
        if collisions:
            raise ValueError(f"successor identities already used: {sorted(collisions)}")
        existing_after = (set(self.nodes) - {predecessor.identity}) | set(successors)
        missing = set(predecessor.successors) - existing_after
        if missing:
            raise ValueError(f"lineage names missing successors: {sorted(missing)}")

        incident = tuple(edge for edge in self.edges.values()
                         if predecessor.identity in (edge.a, edge.b))
        self.nodes.pop(predecessor.identity)
        for edge in incident:
            self.edges.pop(edge.identity)
        self.nodes.update(successors)
        self.lineage[predecessor.identity] = predecessor
        return incident


@dataclass(frozen=True)
class Disincorporation:
    predecessor: RetiredIdentity
    retired_parts: tuple[RetiredIdentity, ...]
    solid_descendants: tuple[Machine, ...]
    material_products: tuple[Machine, ...]
    mass_before_kg: float
    mass_after_kg: float

    @property
    def mass_error_kg(self) -> float:
        return self.mass_after_kg - self.mass_before_kg


def _box_triangles(part: MachinePart) -> np.ndarray:
    c = np.asarray(part.position, dtype=float)
    h = np.asarray(part.half_extent_m, dtype=float)
    vertices = np.asarray([
        c + (-h[0], -h[1], -h[2]), c + (-h[0], -h[1], h[2]),
        c + (-h[0], h[1], -h[2]), c + (-h[0], h[1], h[2]),
        c + (h[0], -h[1], -h[2]), c + (h[0], -h[1], h[2]),
        c + (h[0], h[1], -h[2]), c + (h[0], h[1], h[2]),
    ])
    indices = np.asarray([
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
    ], dtype=np.int32)
    return vertices[indices]


@dataclass
class SdfMaterialBody:
    part_identity: str
    material: str
    density_kg_m3: float
    kernel: MeshSdfKernel
    initial_volume_m3: float
    removed_volume_m3: float = 0.0

    @classmethod
    def from_part(cls, part: MachinePart, density_kg_m3: float,
                  tool_width_m: float) -> "SdfMaterialBody":
        extent = 2.0 * np.asarray(part.half_extent_m, dtype=float)
        # Two cells across the tool width makes the tool, rather than a coarse
        # object grid, decide when the last material bridge is gone.
        cell = max(float(tool_width_m) / 2.0, 2.5e-4)
        dims = tuple(int(min(3072, max(8, math.ceil(value / cell))))
                     for value in extent)
        kernel = MeshSdfKernel.compile(_box_triangles(part), dims)
        return cls(part.identity, part.material, float(density_kg_m3), kernel,
                   kernel.enclosed_volume_m3)

    @property
    def volume_m3(self) -> float:
        return self.kernel.enclosed_volume_m3

    @property
    def mass_kg(self) -> float:
        return self.volume_m3 * self.density_kg_m3

    @property
    def removed_mass_kg(self) -> float:
        return self.removed_volume_m3 * self.density_kg_m3

    def subtract(self, sweeps: list[CapsuleSdf]) -> float:
        removed = self.kernel.subtract_capsules(sweeps)
        self.removed_volume_m3 += removed
        return removed

    def material_islands(self) -> tuple[np.ndarray, int]:
        """Label separate islands; no relationship exists between labels."""
        return ndimage.label(
            self.kernel.occupied,
            structure=ndimage.generate_binary_structure(3, 1),
        )


def chip_batch(source: Machine, source_part: MachinePart, *, identity: str,
               position: tuple[float, float, float], mass_kg: float,
               density_kg_m3: float, transition_identity: str,
               packing_fraction: float = 0.22) -> Machine:
    solid_volume = float(mass_kg) / float(density_kg_m3)
    bulk_volume = solid_volume / packing_fraction
    footprint = max(0.025, bulk_volume ** (1.0 / 3.0) * 2.0)
    height = max(0.001, bulk_volume / (footprint * footprint))
    part = MachinePart(
        f"{identity}.material", "loose-material-batch",
        (float(position[0]), float(position[1]), height / 2.0),
        (footprint / 2.0, footprint / 2.0, height / 2.0),
        mass_kg=float(mass_kg), material=source_part.material,
        part_role="material-product",
        attributes={
            "material_form": MaterialForm.CHIPS.value,
            "predecessor_machine": source.identity,
            "predecessor_part": source_part.identity,
            "topology_transition": transition_identity,
            "solid_volume_m3": solid_volume,
            "bulk_volume_m3": bulk_volume,
            "packing_fraction": packing_fraction,
            "particle_size_range_m": [0.0001, 0.004],
            "moisture_content": source_part.attributes.get("moisture_content"),
            "grain_axis_local": None,
        },
    )
    return Machine(identity, f"chips from {source.label}", "passive",
                   "loose-solid", parts=[part],
                   note="conserved removed material in chip form")


def update_chip_batch(machine: Machine, mass_kg: float,
                      density_kg_m3: float) -> None:
    part = machine.parts[0]
    part.mass_kg = float(mass_kg)
    solid = float(mass_kg) / float(density_kg_m3)
    packing = float(part.attributes["packing_fraction"])
    bulk = solid / packing
    part.attributes["solid_volume_m3"] = solid
    part.attributes["bulk_volume_m3"] = bulk
    footprint = 2.0 * float(part.half_extent_m[0])
    height = max(0.001, bulk / max(footprint * footprint, 1.0e-12))
    part.half_extent_m = (part.half_extent_m[0], part.half_extent_m[1],
                          height / 2.0)
    part.position = (part.position[0], part.position[1], height / 2.0)


def disincorporate_sdf_body(source: Machine, body: SdfMaterialBody, *,
                            transition_identity: str,
                            chips: Machine) -> Disincorporation:
    labels, count = body.material_islands()
    if count < 2:
        raise ValueError("a material bridge still makes this one object")
    source_part = next(p for p in source.parts if p.identity == body.part_identity)
    spacing = body.kernel.voxel_size_m
    children, successor_parts = [], []
    token = transition_identity.replace(".", "-")
    for number in range(1, count + 1):
        mask = labels == number
        indices = np.argwhere(mask)
        lower_i, upper_i = indices.min(axis=0), indices.max(axis=0) + 1
        bounds_min = body.kernel.bounds_min + lower_i * spacing
        bounds_max = body.kernel.bounds_min + upper_i * spacing
        centre = (bounds_min + bounds_max) * 0.5
        identity = f"{source.identity}.piece.{token}.{number:03d}"
        component_kernel = MeshSdfKernel(
            body.kernel.triangles, body.kernel.bounds_min.copy(),
            body.kernel.bounds_max.copy(), mask, True,
            copy.deepcopy(body.kernel.cutouts),
        )
        attributes = dict(source_part.attributes)
        attributes.update({
            "material_form": MaterialForm.SOLID.value,
            "predecessor_machine": source.identity,
            "predecessor_part": source_part.identity,
            "topology_transition": transition_identity,
            "sdf_kernel": component_kernel,
            "sdf_occupied_cells": int(np.count_nonzero(mask)),
            "sdf_voxel_size_m": spacing.tolist(),
        })
        child_part = MachinePart(
            f"{identity}.wood", source_part.kind,
            tuple(float(v) for v in centre),
            tuple(float(v) for v in (bounds_max - bounds_min) * 0.5),
            mass_kg=float(np.count_nonzero(mask) * np.prod(spacing)
                          * body.density_kg_m3),
            material=source_part.material, part_role=source_part.part_role,
            attributes=attributes,
        )
        successor_parts.append(child_part.identity)
        children.append(Machine(
            identity, f"{source.label} separated piece", source.power,
            source.medium, parts=[child_part],
            note=f"separate SDF material island from {source.identity}",
        ))
    successor_ids = tuple(c.identity for c in children) + (chips.identity,)
    transition = Disincorporation(
        RetiredIdentity(source.identity,
                        "last material bridge was removed",
                        successor_ids),
        (RetiredIdentity(source_part.identity,
                         "part became separate material islands",
                         tuple(successor_parts) + (chips.parts[0].identity,)),),
        tuple(children), (chips,),
        body.initial_volume_m3 * body.density_kg_m3,
        sum(c.parts[0].mass_kg for c in children) + chips.parts[0].mass_kg,
    )
    tolerance = body.density_kg_m3 * float(np.prod(spacing)) + 1.0e-12
    if abs(transition.mass_error_kg) > tolerance:
        raise AssertionError(f"non-conservative SDF fission: {transition.mass_error_kg}")
    return transition


__all__ = [
    "Disincorporation", "MaterialForm", "ObjectEdge", "RetiredIdentity",
    "SdfMaterialBody", "WorldObjectGraph", "chip_batch",
    "disincorporate_sdf_body", "update_chip_batch",
]
