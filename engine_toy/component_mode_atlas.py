"""Reusable component-mode reductions for structural assemblies.

This is the matrix boundary between the full reference beam solve and a
later, faster station atlas.  It implements Craig--Bampton component mode
synthesis: interface freedoms remain physical coordinates, while only a
component's internal elastic freedoms may be represented by modes.

The input matrices must be assembled *per component from its own members*.
Slicing the station's already-assembled global matrix is not equivalent at a
shared interface, because it loses which component owns each stiffness term.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# These tolerances diagnose matrix arithmetic.  They never decide whether a
# graph edge is structural and they never cull a member because its strain is
# small.  Scaling by the largest eigenvalue avoids treating units or component
# size as a physical exclusion rule.
SYMMETRY_RELATIVE_TOLERANCE = 1.0e-10
NEGATIVE_EIGENVALUE_RELATIVE_TOLERANCE = 1.0e-10
# Only a roundoff-sized positive eigenvalue is numerically zero.  A merely
# soft, low-frequency internal mode is physical and must remain eligible for
# retention; using an engineering-scale cutoff here would hide it.
INTERNAL_MECHANISM_RELATIVE_TOLERANCE = 64.0 * np.finfo(float).eps


@dataclass(frozen=True)
class ComponentModeAtlas:
    """One component's retained interface and internal modal coordinates."""

    identity: str
    physical_dofs: np.ndarray
    interface_dofs: np.ndarray
    transformation: np.ndarray
    reduced_mass: np.ndarray
    reduced_stiffness: np.ndarray
    fixed_interface_eigenvalues: np.ndarray
    retained_internal_modes: int
    discarded_flexibility_bound: float

    @property
    def reduced_size(self) -> int:
        return int(self.transformation.shape[1])

    def recover(self, reduced_coordinates: np.ndarray) -> np.ndarray:
        """Recover this component's complete physical displacement vector."""
        q = np.asarray(reduced_coordinates, dtype=float)
        if q.shape != (self.reduced_size,):
            raise ValueError(
                f"{self.identity}: expected {self.reduced_size} coordinates, "
                f"got {q.shape}")
        return self.transformation @ q


@dataclass(frozen=True)
class AssembledModeAtlas:
    """Several component atlases joined on their shared physical interfaces."""

    components: tuple[ComponentModeAtlas, ...]
    interface_dofs: np.ndarray
    physical_dofs: np.ndarray
    transformation: np.ndarray
    reduced_mass: np.ndarray
    reduced_stiffness: np.ndarray
    component_reduced_indices: tuple[np.ndarray, ...]

    @property
    def reduced_size(self) -> int:
        return int(self.reduced_stiffness.shape[0])

    def recover(self, reduced_coordinates: np.ndarray) -> np.ndarray:
        """Recover physical coordinates in ``physical_dofs`` row order."""
        q = np.asarray(reduced_coordinates, dtype=float)
        if q.shape != (self.reduced_size,):
            raise ValueError(f"expected {self.reduced_size} coordinates, got {q.shape}")
        return self.transformation @ q


@dataclass(frozen=True)
class ComponentPartition:
    """Topology-derived ownership boundary for one graph assembly."""

    identity: str
    member_identities: tuple[str, ...]
    node_indices: np.ndarray
    interface_node_indices: np.ndarray
    node_mass_fractions: dict[str, float]
    floating_reference_added: bool = False

    @property
    def interface_local_dofs(self) -> np.ndarray:
        local = {int(node): i for i, node in enumerate(self.node_indices)}
        return np.asarray([
            local[int(node)] * 6 + freedom
            for node in self.interface_node_indices
            for freedom in range(6)
        ], dtype=np.int64)


@dataclass(frozen=True)
class ComponentOrchestrationGraph:
    """CSR graph of weld continuity and constitutive component couplings."""

    partitions: tuple[ComponentPartition, ...]
    identity_index: dict[str, int]
    weld_indptr: np.ndarray
    weld_indices: np.ndarray
    coupling_indptr: np.ndarray
    coupling_indices: np.ndarray
    coupling_edges: dict[tuple[str, str], tuple[str, ...]]

    def _neighbors(self, identity: str, indptr: np.ndarray,
                   indices: np.ndarray) -> tuple[str, ...]:
        i = self.identity_index[identity]
        return tuple(self.partitions[int(j)].identity
                     for j in indices[indptr[i]:indptr[i + 1]])

    def weld_neighbors(self, identity: str) -> tuple[str, ...]:
        return self._neighbors(identity, self.weld_indptr, self.weld_indices)

    def coupling_neighbors(self, identity: str) -> tuple[str, ...]:
        return self._neighbors(
            identity, self.coupling_indptr, self.coupling_indices)

    def welded_clusters(self, selected) -> tuple[tuple[str, ...], ...]:
        """Connected components of selected vertices under weld adjacency."""
        remaining = {self.identity_index[str(identity)] for identity in selected}
        clusters = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            stack = [seed]
            cluster = []
            while stack:
                i = stack.pop()
                cluster.append(self.partitions[i].identity)
                neighbors = self.weld_indices[
                    self.weld_indptr[i]:self.weld_indptr[i + 1]]
                for neighbor in neighbors:
                    j = int(neighbor)
                    if j in remaining:
                        remaining.remove(j)
                        stack.append(j)
            clusters.append(tuple(sorted(cluster)))
        return tuple(sorted(clusters))


def _csr_from_pairs(size: int, pairs: set[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    neighbors = [set() for _ in range(size)]
    for a, b in pairs:
        if a == b:
            continue
        neighbors[a].add(b)
        neighbors[b].add(a)
    indptr = np.zeros(size + 1, dtype=np.int64)
    for i, values in enumerate(neighbors):
        indptr[i + 1] = indptr[i] + len(values)
    indices = np.asarray([j for values in neighbors for j in sorted(values)],
                         dtype=np.int64)
    return indptr, indices


def build_component_orchestration_graph(
        solver, *, ownership_field: str = "assembly"
        ) -> ComponentOrchestrationGraph:
    """Make component residency a graph operation over existing topology."""
    partitions = discover_component_partitions(
        solver, ownership_field=ownership_field)
    identity_index = {part.identity: i for i, part in enumerate(partitions)}
    node_partitions: dict[int, set[int]] = {}
    for i, partition in enumerate(partitions):
        for node in partition.node_indices:
            node_partitions.setdefault(int(node), set()).add(i)
    weld_pairs: set[tuple[int, int]] = set()
    for owners in node_partitions.values():
        ordered = sorted(owners)
        for at, a in enumerate(ordered):
            for b in ordered[at + 1:]:
                weld_pairs.add((a, b))

    member_ids = {identity for part in partitions
                  for identity in part.member_identities}
    from graph_columns import columns
    graph_columns = columns(solver.document)
    coupling_pairs: set[tuple[int, int]] = set()
    coupling_edges: dict[tuple[str, str], list[str]] = {}
    for edge_i, edge in enumerate(solver.document["edges"]):
        if (edge["identity"] in member_ids
                or graph_columns["is_routed"][edge_i]
                or not graph_columns["in_frame"][edge_i]):
            continue
        ma = int(solver.solver_master_index[solver.index[edge["a"]]])
        mb = int(solver.solver_master_index[solver.index[edge["b"]]])
        for a in node_partitions.get(ma, ()):
            for b in node_partitions.get(mb, ()):
                if a == b:
                    continue
                pair = (min(a, b), max(a, b))
                coupling_pairs.add(pair)
                names = tuple(sorted((partitions[a].identity,
                                      partitions[b].identity)))
                coupling_edges.setdefault(names, []).append(edge["identity"])
    weld_indptr, weld_indices = _csr_from_pairs(len(partitions), weld_pairs)
    coupling_indptr, coupling_indices = _csr_from_pairs(
        len(partitions), coupling_pairs)
    return ComponentOrchestrationGraph(
        partitions=partitions, identity_index=identity_index,
        weld_indptr=weld_indptr, weld_indices=weld_indices,
        coupling_indptr=coupling_indptr, coupling_indices=coupling_indices,
        coupling_edges={key: tuple(sorted(values))
                        for key, values in coupling_edges.items()})


@dataclass(frozen=True)
class GlobalGestaltSubspace:
    """Global free-coordinate map with selected components made rigid."""

    free_dofs: np.ndarray
    transformation: np.ndarray
    reduced_mass: np.ndarray
    reduced_stiffness: np.ndarray
    component_columns: dict[str, slice]

    @property
    def reduced_size(self) -> int:
        return int(self.transformation.shape[1])

    def recover(self, reduced_coordinates: np.ndarray) -> np.ndarray:
        value = np.asarray(reduced_coordinates, dtype=float)
        if value.shape != (self.reduced_size,):
            raise ValueError("reduced state does not match global gestalt subspace")
        return self.transformation @ value

    def project(self, physical_free_state: np.ndarray) -> np.ndarray:
        """Mass-project a physical free state into the dormant subspace."""
        value = np.asarray(physical_free_state, dtype=float)
        if value.shape != (len(self.free_dofs),):
            raise ValueError("physical state does not match global free coordinates")
        rhs = self.transformation.T @ self._mass_matrix @ value
        return np.linalg.solve(self.reduced_mass, rhs)

    def mass_orthonormal_basis(self, mechanism_eigenvalue: float) -> dict:
        """Diagonalize the projected system for the live Newmark lane."""
        mass_values, mass_vectors = np.linalg.eigh(self.reduced_mass)
        scale = max(float(np.max(np.abs(mass_values))), 1.0)
        if np.min(mass_values) <= np.finfo(float).eps * scale:
            raise ValueError("global gestalt reduced mass is not positive definite")
        inverse_sqrt = ((mass_vectors / np.sqrt(mass_values)[None, :])
                        @ mass_vectors.T)
        standard = inverse_sqrt @ self.reduced_stiffness @ inverse_sqrt
        values, vectors = np.linalg.eigh((standard + standard.T) * .5)
        keep = values > float(mechanism_eigenvalue)
        order = np.r_[np.flatnonzero(keep), np.flatnonzero(~keep)]
        basis = self.transformation @ inverse_sqrt @ vectors[:, order]
        omega = np.sqrt(np.maximum(values[order], 0.0))
        omega[len(np.flatnonzero(keep)):] = 0.0
        return {"basis_free": np.ascontiguousarray(basis),
                "omega": omega,
                "elastic_count": int(np.count_nonzero(keep)),
                "mechanism_eigenvalues": values[~keep]}

    # Kept out of repr/comparison while remaining immutable to callers.
    _mass_matrix: np.ndarray


def build_global_gestalt_subspace(
        solver, stiffness: np.ndarray, mass: np.ndarray, free_dofs: np.ndarray,
        dormant_partitions: tuple[ComponentPartition, ...] | list[ComponentPartition]
        ) -> GlobalGestaltSubspace:
    """Replace disjoint component rows by rigid maps in the global system.

    The global K is projected only after all neighboring member contributions
    have been assembled, so an outside beam remains attached to the component's
    moving interface point. Overlapping dormant partitions are rejected; they
    must first be merged into one welded gestalt rather than given conflicting
    rigid motions at a shared node.
    """
    free = np.asarray(free_dofs, dtype=np.int64)
    k = _symmetric(stiffness, "global-gestalt", "stiffness")
    mass_value = np.asarray(mass, dtype=float)
    m = (np.diag(mass_value) if mass_value.ndim == 1
         else _symmetric(mass_value, "global-gestalt", "mass"))
    if k.shape != m.shape or np.any(free < 0) or np.any(free >= len(k)):
        raise ValueError("global matrices/free coordinates do not match")
    free_row = {int(dof): i for i, dof in enumerate(free)}
    claimed: set[int] = set()
    component_maps = []
    for partition in dormant_partitions:
        assembled = assemble_component_partition(solver, partition)
        physical = np.asarray(assembled["physical_dofs"], dtype=np.int64)
        overlap = claimed.intersection(int(dof) for dof in physical)
        if overlap:
            raise ValueError(
                f"{partition.identity}: dormant components overlap; merge them")
        claimed.update(int(dof) for dof in physical)
        local_mass = np.asarray(assembled["mass_matrix"], dtype=float)
        point_mass = np.asarray([
            np.mean(np.diag(local_mass)[i * 6:i * 6 + 3])
            for i in range(len(assembled["node_positions"]))])
        total = float(np.sum(point_mass))
        positions = np.asarray(assembled["node_positions"], dtype=float)
        reference = ((point_mass @ positions) / total
                     if total > 0.0 else np.mean(positions, axis=0))
        rigid = build_rigid_gestalt_atlas(
            partition.identity, assembled["stiffness_matrix"], local_mass,
            physical, positions, reference).transformation
        local_free = np.asarray([int(dof) in free_row for dof in physical])
        fixed_map = rigid[~local_free]
        if fixed_map.size:
            _u, singular, vh = np.linalg.svd(fixed_map, full_matrices=True)
            tolerance = (max(fixed_map.shape) * np.finfo(float).eps
                         * (float(singular[0]) if singular.size else 1.0))
            rank = int(np.count_nonzero(singular > tolerance))
            admissible = vh[rank:].T
        else:
            admissible = np.eye(6)
        component_maps.append((partition.identity, physical[local_free],
                               rigid[local_free] @ admissible))

    claimed_free = {dof for dof in claimed if dof in free_row}
    outside = [int(dof) for dof in free if int(dof) not in claimed_free]
    column_count = len(outside) + sum(mapping.shape[1]
                                      for _name, _dofs, mapping in component_maps)
    transform = np.zeros((len(free), column_count), dtype=float)
    for column, dof in enumerate(outside):
        transform[free_row[dof], column] = 1.0
    component_columns = {}
    cursor = len(outside)
    for identity, physical, mapping in component_maps:
        columns = slice(cursor, cursor + mapping.shape[1])
        rows = np.asarray([free_row[int(dof)] for dof in physical], dtype=np.int64)
        transform[np.ix_(rows, np.arange(columns.start, columns.stop))] = mapping
        component_columns[identity] = columns
        cursor = columns.stop
    if np.linalg.matrix_rank(transform) != transform.shape[1]:
        raise ValueError("global gestalt transformation is rank deficient")
    kff = k[np.ix_(free, free)]
    mff = m[np.ix_(free, free)]
    return GlobalGestaltSubspace(
        free_dofs=free, transformation=transform,
        reduced_mass=transform.T @ mff @ transform,
        reduced_stiffness=transform.T @ kff @ transform,
        component_columns=component_columns, _mass_matrix=mff)


def discover_component_partitions(
        solver, *, ownership_field: str = "assembly") -> tuple[ComponentPartition, ...]:
    """Discover generic component boundaries from graph ownership/topology.

    No part names or machine types participate. An endpoint is an interface
    when another owned component touches the same solver master or when it is
    fixed externally. A completely free component retains one six-DOF
    reference node so its rigid-body motion remains explicit rather than being
    mistaken for an internal mechanism.
    """
    groups: dict[str, list[dict]] = {}
    for edge in solver.members:
        owner = str(edge.get(ownership_field, "unassigned"))
        groups.setdefault(owner, []).append(edge)
    node_owners: dict[int, set[str]] = {}
    group_nodes: dict[str, set[int]] = {}
    for owner, edges in groups.items():
        touched: set[int] = set()
        for edge in edges:
            for endpoint in (edge["a"], edge["b"]):
                master = int(solver.solver_master_index[
                    solver.index[endpoint]])
                touched.add(master)
        group_nodes[owner] = touched
    # Interface discovery must include constitutive connections that are not
    # beams: released actuators, dampers, contacts and currently open locks.
    # Their stiffness does not belong in this component matrix, but their
    # endpoint wrench can wake it and therefore crosses its boundary.
    from graph_columns import columns
    graph_columns = columns(solver.document)
    for edge_i, edge in enumerate(solver.document["edges"]):
        if graph_columns["is_routed"][edge_i] or not graph_columns["in_frame"][edge_i]:
            continue
        owner = str(edge.get(ownership_field, "unassigned"))
        for endpoint in (edge["a"], edge["b"]):
            master = int(solver.solver_master_index[solver.index[endpoint]])
            node_owners.setdefault(master, set()).add(owner)

    partitions = []
    nodes = solver.document["nodes"]
    for owner in sorted(groups):
        touched = group_nodes[owner]
        interface = sorted(
            node for node in touched
            if len(node_owners[node]) > 1 or bool(nodes[node].get("fixed_to")))
        floating_reference = False
        if not interface and touched:
            centre = np.mean(solver.position[list(touched)], axis=0)
            reference = min(
                touched,
                key=lambda node: float(np.linalg.norm(
                    solver.position[node] - centre)))
            interface = [reference]
            floating_reference = True
        mass_fractions = {
            nodes[node]["identity"]: 1.0
            for node in touched
            if str(nodes[node].get(ownership_field, "unassigned")) == owner
        }
        partitions.append(ComponentPartition(
            identity=owner,
            member_identities=tuple(sorted(
                edge["identity"] for edge in groups[owner])),
            node_indices=np.asarray(sorted(touched), dtype=np.int64),
            interface_node_indices=np.asarray(interface, dtype=np.int64),
            node_mass_fractions=mass_fractions,
            floating_reference_added=floating_reference))
    return tuple(partitions)


def assemble_component_partition(solver, partition: ComponentPartition) -> dict:
    """Assemble a discovered partition through the solver's owned path."""
    matrices = solver.assemble_component_matrices(
        partition.member_identities,
        node_mass_fractions=partition.node_mass_fractions)
    if not np.array_equal(matrices["node_indices"], partition.node_indices):
        raise ValueError(
            f"{partition.identity}: component topology changed after discovery")
    return {**matrices,
            "interface_local_dofs": partition.interface_local_dofs}


def merge_welded_partition_cluster(
        solver, graph: ComponentOrchestrationGraph, identities
        ) -> ComponentPartition:
    """Union one weld-connected selection into a single rigid candidate."""
    selected = {str(identity) for identity in identities}
    if not selected:
        raise ValueError("cannot merge an empty component cluster")
    clusters = graph.welded_clusters(selected)
    if len(clusters) != 1:
        raise ValueError("selected components are not one weld-connected cluster")
    pieces = [graph.partitions[graph.identity_index[identity]]
              for identity in selected]
    member_ids = tuple(sorted({identity for part in pieces
                               for identity in part.member_identities}))
    nodes = np.asarray(sorted({int(node) for part in pieces
                               for node in part.node_indices}), dtype=np.int64)
    node_set = set(int(node) for node in nodes)
    interface: set[int] = {
        int(node) for node in nodes
        if bool(solver.document["nodes"][int(node)].get("fixed_to"))}
    selected_indices = {graph.identity_index[identity] for identity in selected}
    outside_nodes = {int(node) for i, part in enumerate(graph.partitions)
                     if i not in selected_indices for node in part.node_indices}
    interface.update(node_set.intersection(outside_nodes))
    edge_by_identity = {edge["identity"]: edge
                        for edge in solver.document["edges"]}
    for pair, edge_ids in graph.coupling_edges.items():
        inside = [identity in selected for identity in pair]
        if inside[0] == inside[1]:
            continue
        for edge_id in edge_ids:
            edge = edge_by_identity[edge_id]
            for endpoint in (edge["a"], edge["b"]):
                master = int(solver.solver_master_index[
                    solver.index[endpoint]])
                if master in node_set:
                    interface.add(master)
    floating = False
    if not interface:
        centre = np.mean(solver.position[nodes], axis=0)
        interface.add(min(
            (int(node) for node in nodes),
            key=lambda node: float(np.linalg.norm(
                solver.position[node] - centre))))
        floating = True
    mass_fractions: dict[str, float] = {}
    for part in pieces:
        for identity, fraction in part.node_mass_fractions.items():
            mass_fractions[identity] = mass_fractions.get(identity, 0.0) + fraction
    if any(value > 1.0 + 1.0e-12 for value in mass_fractions.values()):
        raise ValueError("merged components claim the same node mass twice")
    return ComponentPartition(
        identity="gestalt[" + "+".join(sorted(selected)) + "]",
        member_identities=member_ids, node_indices=nodes,
        interface_node_indices=np.asarray(sorted(interface), dtype=np.int64),
        node_mass_fractions=mass_fractions,
        floating_reference_added=floating)


@dataclass(frozen=True)
class RigidGestaltAtlas:
    """Six-coordinate rigid representation of an elastic component.

    The component still exists structurally: every point is recovered from
    the gestalt translation/rotation, and physical loads project to the
    resultant force and moment at ``reference_point``.  This is the dormant
    representation paired with an elastic ``ComponentModeAtlas``; it is not
    an edge exclusion flag.
    """

    identity: str
    physical_dofs: np.ndarray
    reference_point: np.ndarray
    transformation: np.ndarray
    reduced_mass: np.ndarray
    reduced_stiffness: np.ndarray

    def recover(self, rigid_coordinates: np.ndarray) -> np.ndarray:
        q = np.asarray(rigid_coordinates, dtype=float)
        if q.shape != (6,):
            raise ValueError(f"{self.identity}: expected 6 rigid coordinates")
        return self.transformation @ q

    def project_load(self, physical_load: np.ndarray) -> np.ndarray:
        """Return the interface resultant wrench without losing moments."""
        load = np.asarray(physical_load, dtype=float)
        if load.shape != (len(self.physical_dofs),):
            raise ValueError(
                f"{self.identity}: physical load has shape {load.shape}")
        return self.transformation.T @ load


@dataclass(frozen=True)
class GestaltStressEnvelope:
    """Certified dormant-model bound from interface wrench to peak stress.

    Rows are stress probes and columns are absolute component interface-wrench
    coordinates. Each coefficient already bounds normal plus von-Mises shear
    contribution. ``residual_bound_pa`` covers validation error and omitted
    effects. The wake decision is therefore based on a stated engineering
    bound, never on render visibility or an arbitrary frequency cutoff.
    """

    influence_pa_per_wrench: np.ndarray
    residual_bound_pa: float
    probe_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        influence = np.asarray(self.influence_pa_per_wrench, dtype=float)
        if (influence.ndim != 2 or not np.all(np.isfinite(influence))
                or np.any(influence < 0.0)):
            raise ValueError("stress influence must be a finite nonnegative matrix")
        if not np.isfinite(self.residual_bound_pa) or self.residual_bound_pa < 0.0:
            raise ValueError("residual stress bound must be finite and nonnegative")
        if self.probe_identities and len(self.probe_identities) != influence.shape[0]:
            raise ValueError("stress probe identities must name every matrix row")
        object.__setattr__(self, "influence_pa_per_wrench", influence)

    def bound(self, interface_wrench: np.ndarray) -> float:
        wrench = np.asarray(interface_wrench, dtype=float)
        if wrench.shape != (self.influence_pa_per_wrench.shape[1],):
            raise ValueError("interface wrench does not match stress envelope")
        # Each coefficient is already an absolute upper bound. Applying the
        # triangle inequality here makes arbitrary combinations conservative:
        # cancellations may overpredict stress but can never hide a wake.
        predicted = self.influence_pa_per_wrench @ np.abs(wrench)
        peak = float(np.max(predicted)) if predicted.size else 0.0
        return peak + float(self.residual_bound_pa)


def build_component_stress_envelope(
        solver, partition: ComponentPartition, *, residual_bound_pa: float = 0.0
        ) -> GestaltStressEnvelope:
    """Bake a conservative interface-wrench -> member-stress envelope.

    Unit interface loads are solved with inertia relief. Six rigid-body
    coordinates therefore absorb resultant acceleration while the remaining
    deformation produces beam stress. A KKT constraint fixes the arbitrary
    rigid pose without pinning a physical node. Components with an additional
    unmodelled mechanism are rejected instead of regularized into plausibility.
    """
    from frame_solver import _element_frame

    assembled = assemble_component_partition(solver, partition)
    stiffness = np.asarray(assembled["stiffness_matrix"], dtype=float)
    mass = np.asarray(assembled["mass_matrix"], dtype=float)
    positions = np.asarray(assembled["node_positions"], dtype=float)
    physical_dofs = np.asarray(assembled["physical_dofs"], dtype=np.int64)
    boundary = np.asarray(assembled["interface_local_dofs"], dtype=np.int64)
    n = len(physical_dofs)
    if not len(boundary):
        raise ValueError(f"{partition.identity}: no interface wrench coordinates")
    point_mass = np.asarray([
        np.mean(np.diag(mass)[i * 6:i * 6 + 3])
        for i in range(len(positions))], dtype=float)
    total_mass = float(np.sum(point_mass))
    reference = ((point_mass @ positions) / total_mass
                 if total_mass > 0.0 else np.mean(positions, axis=0))
    rigid = build_rigid_gestalt_atlas(
        partition.identity, stiffness, mass, physical_dofs, positions,
        reference)
    transform = rigid.transformation
    coupling = mass @ transform
    kkt = np.block([[stiffness, coupling],
                    [coupling.T, np.zeros((6, 6), dtype=float)]])
    loads = np.zeros((n, len(boundary)), dtype=float)
    loads[boundary, np.arange(len(boundary))] = 1.0
    rhs = np.vstack((loads, np.zeros((6, len(boundary)), dtype=float)))
    try:
        solved = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{partition.identity}: component has an internal mechanism; "
            "promote it to the interface before baking") from exc
    displacement = solved[:n]
    relative_residual = float(np.linalg.norm(kkt @ solved - rhs)) / max(
        float(np.linalg.norm(rhs)), 1.0)
    if relative_residual > 1.0e-8:
        raise ValueError(
            f"{partition.identity}: inertia-relief solve residual "
            f"{relative_residual:g}")

    node_local = {int(node): i for i, node in enumerate(partition.node_indices)}
    edge_by_identity = {edge["identity"]: edge for edge in solver.members}
    coefficients = []
    identities = []
    for identity in partition.member_identities:
        edge = edge_by_identity.get(identity)
        if edge is None:
            continue
        ia, ib = solver.index[edge["a"]], solver.index[edge["b"]]
        ma = int(solver.solver_master_index[ia])
        mb = int(solver.solver_master_index[ib])
        ua = (solver.solver_endpoint_transform[ia]
              @ displacement[node_local[ma] * 6:node_local[ma] * 6 + 6])
        ub = (solver.solver_endpoint_transform[ib]
              @ displacement[node_local[mb] * 6:node_local[mb] * 6 + 6])
        a, b = solver.position[ia], solver.position[ib]
        length = float(np.linalg.norm(b - a))
        if length < 1.0e-9:
            continue
        rotation = _element_frame(edge, a, b)
        da, ra = rotation @ ua[:3], rotation @ ua[3:]
        db, rb = rotation @ ub[:3], rotation @ ub[3:]
        E, G, _A, _Iy, _Iz, _J, _kappa, _material = solver._section(edge)
        damage = edge.get("damage") or {}
        outer_y = float(damage.get(
            "section_outer_y_m", edge.get("radius", .012)))
        outer_z = float(damage.get(
            "section_outer_z_m", edge.get("radius", .012)))
        released = set(edge.get("freedoms_released") or ())
        axial = ((db[0] - da[0]) / length
                 if "x" not in released else np.zeros(len(boundary)))
        curvature_y = ((rb[1] - ra[1]) / length
                       if not released.intersection({"ry", "rz"})
                       else np.zeros(len(boundary)))
        curvature_z = ((rb[2] - ra[2]) / length
                       if not released.intersection({"ry", "rz"})
                       else np.zeros(len(boundary)))
        shear_y = ((db[1] - da[1]) / length - (ra[2] + rb[2]) / 2.0)
        shear_z = ((db[2] - da[2]) / length + (ra[1] + rb[1]) / 2.0)
        twist = (rb[0] - ra[0]) / length
        if released.intersection({"y", "z", "rx"}):
            shear_y = np.zeros(len(boundary))
            shear_z = np.zeros(len(boundary))
            twist = np.zeros(len(boundary))
        normal_bound = E * (np.abs(axial)
                            + np.abs(curvature_y) * outer_z
                            + np.abs(curvature_z) * outer_y)
        shear_bound = G * (np.abs(shear_y) + np.abs(shear_z)
                           + np.abs(twist) * max(outer_y, outer_z))
        coefficients.append(normal_bound + np.sqrt(3.0) * shear_bound)
        identities.append(identity)
    matrix = (np.asarray(coefficients, dtype=float)
              if coefficients else np.zeros((0, len(boundary)), dtype=float))
    return GestaltStressEnvelope(
        matrix, residual_bound_pa, tuple(identities))


@dataclass(frozen=True)
class AdaptiveGestaltPolicy:
    """Explicit physical policy for elastic <-> rigid residency.

    Thresholds are deliberately required from the owning scene/material
    policy.  The reference solver has no implicit "game" tolerance.
    """

    sleep_stress_bound_pa: float
    wake_stress_bound_pa: float
    sleep_internal_kinetic_bound_j: float
    sleep_dwell_s: float
    sleep_wrench_rate_limit_per_s: float | None = None

    def __post_init__(self) -> None:
        values = (self.sleep_stress_bound_pa, self.wake_stress_bound_pa,
                  self.sleep_internal_kinetic_bound_j, self.sleep_dwell_s)
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("gestalt policy bounds must be finite and nonnegative")
        if self.wake_stress_bound_pa <= self.sleep_stress_bound_pa:
            raise ValueError("wake stress must exceed sleep stress for hysteresis")
        limit = self.sleep_wrench_rate_limit_per_s
        if limit is not None and (not np.isfinite(limit) or limit < 0.0):
            raise ValueError("wrench-rate limit must be finite and nonnegative")


@dataclass
class AdaptiveGestaltState:
    """Residency state; ``elastic`` is the conservative initial state."""

    identity: str
    policy: AdaptiveGestaltPolicy
    stress_envelope: GestaltStressEnvelope
    elastic: bool = True
    quiet_time_s: float = 0.0
    transition_count: int = 0

    def observe(self, dt: float, interface_wrench: np.ndarray, *,
                solved_peak_stress_pa: float | None = None,
                solved_stress_error_bound_pa: float = 0.0,
                solved_internal_kinetic_j: float | None = None,
                interface_wrench_rate_per_s: float | None = None) -> str:
        """Return ``active``, ``dormant``, ``slept`` or ``woke``.

        An elastic component may sleep only from an actual local solve plus
        its stated error bound. A rigid component wakes immediately from its
        certified interface-wrench envelope. State transfer is intentionally
        owned by the scene integrator so it can conserve generalized momentum.
        """
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("observation dt must be finite and nonnegative")
        estimated = self.stress_envelope.bound(interface_wrench)
        if not self.elastic:
            if estimated >= self.policy.wake_stress_bound_pa:
                self.elastic = True
                self.quiet_time_s = 0.0
                self.transition_count += 1
                return "woke"
            return "dormant"

        if solved_peak_stress_pa is None or solved_internal_kinetic_j is None:
            self.quiet_time_s = 0.0
            return "active"
        actual_bound = float(solved_peak_stress_pa) + float(
            solved_stress_error_bound_pa)
        if (not np.isfinite(actual_bound) or actual_bound < 0.0):
            raise ValueError("solved stress bound must be finite and nonnegative")
        kinetic = float(solved_internal_kinetic_j)
        if not np.isfinite(kinetic) or kinetic < 0.0:
            raise ValueError("internal kinetic energy must be finite and nonnegative")
        rate_limit = self.policy.sleep_wrench_rate_limit_per_s
        rate_ok = (rate_limit is None or
                   (interface_wrench_rate_per_s is not None and
                    np.isfinite(interface_wrench_rate_per_s) and
                    interface_wrench_rate_per_s <= rate_limit))
        if (actual_bound <= self.policy.sleep_stress_bound_pa
                and kinetic <= self.policy.sleep_internal_kinetic_bound_j
                and rate_ok):
            self.quiet_time_s += dt
        else:
            self.quiet_time_s = 0.0
        if self.quiet_time_s >= self.policy.sleep_dwell_s:
            self.elastic = False
            self.quiet_time_s = 0.0
            self.transition_count += 1
            return "slept"
        return "active"


def build_rigid_gestalt_atlas(
        identity: str, stiffness: np.ndarray, mass: np.ndarray,
        physical_dofs: np.ndarray, node_positions: np.ndarray,
        reference_point: np.ndarray) -> RigidGestaltAtlas:
    """Bake six rigid coordinates while retaining every component point."""
    k = _symmetric(stiffness, identity, "stiffness")
    m = _symmetric(mass, identity, "mass")
    if k.shape != m.shape:
        raise ValueError(f"{identity}: stiffness and mass shapes differ")
    positions = np.asarray(node_positions, dtype=float)
    reference = np.asarray(reference_point, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{identity}: node_positions must have shape (nodes, 3)")
    n = len(positions)
    if k.shape != (6 * n, 6 * n) or reference.shape != (3,):
        raise ValueError(f"{identity}: matrices/point do not match node positions")
    physical = np.asarray(physical_dofs, dtype=np.int64)
    if physical.shape != (6 * n,) or len(np.unique(physical)) != 6 * n:
        raise ValueError(f"{identity}: physical_dofs must uniquely name each row")
    transform = np.zeros((6 * n, 6), dtype=float)
    transform.reshape(n, 6, 6)[:, :, :] = np.eye(6)
    for block, offset in zip(transform.reshape(n, 6, 6),
                             positions - reference):
        skew = np.asarray(((0.0, -offset[2], offset[1]),
                           (offset[2], 0.0, -offset[0]),
                           (-offset[1], offset[0], 0.0)))
        block[:3, 3:] = -skew
    return RigidGestaltAtlas(
        identity=identity, physical_dofs=physical,
        reference_point=reference, transformation=transform,
        reduced_mass=transform.T @ m @ transform,
        reduced_stiffness=transform.T @ k @ transform)


def _symmetric(matrix: np.ndarray, identity: str, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{identity}: {name} must be square")
    scale = max(float(np.max(np.abs(value))), 1.0)
    error = float(np.max(np.abs(value - value.T)))
    if error > SYMMETRY_RELATIVE_TOLERANCE * scale:
        raise ValueError(f"{identity}: {name} is not symmetric ({error:g})")
    return (value + value.T) * 0.5


@dataclass
class AdaptiveGestaltRuntime:
    """Runnable linear component with elastic and six-DOF representations.

    Both matrices are baked from the same physical component. Transitions use
    a mass-orthogonal projection, which preserves every linear/angular
    momentum resultant representable by the rigid basis. The elastic atlas
    must be the full reference atlas: a truncated atlas cannot be promoted as
    though discarded physical information had been solved.
    """

    elastic_atlas: ComponentModeAtlas
    rigid_atlas: RigidGestaltAtlas
    residency: AdaptiveGestaltState
    physical_mass: np.ndarray
    physical_damping: np.ndarray | None = None

    def __post_init__(self) -> None:
        physical = np.asarray(self.elastic_atlas.physical_dofs, dtype=np.int64)
        if not np.array_equal(physical, self.rigid_atlas.physical_dofs):
            raise ValueError("elastic and rigid atlases must name identical DOFs")
        n = len(physical)
        if self.elastic_atlas.transformation.shape != (n, n):
            raise ValueError("adaptive elastic atlas must retain every physical DOF")
        self.physical_mass = _symmetric(
            self.physical_mass, self.residency.identity, "physical mass")
        if self.physical_mass.shape != (n, n):
            raise ValueError("physical mass does not match component atlases")
        if self.physical_damping is None:
            self.physical_damping = np.zeros((n, n), dtype=float)
        else:
            self.physical_damping = _symmetric(
                self.physical_damping, self.residency.identity,
                "physical damping")
            if self.physical_damping.shape != (n, n):
                raise ValueError("physical damping does not match component atlases")
        self.coordinates = np.zeros(self.elastic_atlas.reduced_size)
        self.velocity = np.zeros_like(self.coordinates)
        self.acceleration = np.zeros_like(self.coordinates)

    @property
    def atlas(self) -> ComponentModeAtlas | RigidGestaltAtlas:
        return self.elastic_atlas if self.residency.elastic else self.rigid_atlas

    def physical_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transform = self.atlas.transformation
        return (transform @ self.coordinates, transform @ self.velocity,
                transform @ self.acceleration)

    def _project(self, transform: np.ndarray, physical: np.ndarray) -> np.ndarray:
        reduced_mass = transform.T @ self.physical_mass @ transform
        rhs = transform.T @ self.physical_mass @ physical
        return np.linalg.solve(reduced_mass, rhs)

    def set_physical_state(self, displacement: np.ndarray,
                           velocity: np.ndarray | None = None,
                           acceleration: np.ndarray | None = None) -> None:
        transform = self.atlas.transformation
        values = (displacement,
                  np.zeros(len(self.physical_mass)) if velocity is None else velocity,
                  np.zeros(len(self.physical_mass)) if acceleration is None
                  else acceleration)
        projected = [self._project(transform, np.asarray(value, dtype=float))
                     for value in values]
        self.coordinates, self.velocity, self.acceleration = projected

    def _adopt_current_representation(self) -> None:
        # ``residency`` has already switched, while the coordinate arrays are
        # still expressed in the previous atlas. Recover through the opposite
        # transform, then mass-project into the newly selected one.
        old = (self.rigid_atlas if self.residency.elastic
               else self.elastic_atlas)
        physical = (old.transformation @ self.coordinates,
                    old.transformation @ self.velocity,
                    old.transformation @ self.acceleration)
        new = self.atlas.transformation
        self.coordinates = self._project(new, physical[0])
        self.velocity = self._project(new, physical[1])
        self.acceleration = self._project(new, physical[2])

    def observe(self, dt: float, interface_wrench: np.ndarray, **metrics) -> str:
        if self.residency.elastic:
            physical_velocity = self.elastic_atlas.transformation @ self.velocity
            rigid_velocity = self._project(
                self.rigid_atlas.transformation, physical_velocity)
            residual = (physical_velocity
                        - self.rigid_atlas.transformation @ rigid_velocity)
            internal_kinetic = .5 * float(
                residual @ self.physical_mass @ residual)
            supplied = metrics.get("solved_internal_kinetic_j")
            metrics["solved_internal_kinetic_j"] = max(
                internal_kinetic, 0.0 if supplied is None else float(supplied))
        transition = self.residency.observe(dt, interface_wrench, **metrics)
        if transition in {"slept", "woke"}:
            self._adopt_current_representation()
        return transition

    def advance(self, dt: float, physical_load: np.ndarray) -> None:
        """Average-acceleration Newmark step in the current baked subspace."""
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("component dt must be finite and nonnegative")
        load = np.asarray(physical_load, dtype=float)
        if load.shape != (len(self.physical_mass),):
            raise ValueError("physical load does not match component")
        if dt == 0.0:
            return
        transform = self.atlas.transformation
        mass = self.atlas.reduced_mass
        stiffness = self.atlas.reduced_stiffness
        damping = transform.T @ self.physical_damping @ transform
        force = transform.T @ load
        beta, gamma = .25, .5
        q_predict = (self.coordinates + dt * self.velocity
                     + dt * dt * (.5 - beta) * self.acceleration)
        v_predict = self.velocity + dt * (1.0 - gamma) * self.acceleration
        effective = mass + gamma * dt * damping + beta * dt * dt * stiffness
        rhs = force - damping @ v_predict - stiffness @ q_predict
        acceleration = np.linalg.solve(effective, rhs)
        self.coordinates = q_predict + beta * dt * dt * acceleration
        self.velocity = v_predict + gamma * dt * acceleration
        self.acceleration = acceleration


def build_adaptive_gestalt_runtime(
        identity: str, stiffness: np.ndarray, mass: np.ndarray,
        physical_dofs: np.ndarray, node_positions: np.ndarray,
        interface_local_dofs: np.ndarray, reference_point: np.ndarray,
        policy: AdaptiveGestaltPolicy, stress_envelope: GestaltStressEnvelope,
        *, physical_damping: np.ndarray | None = None
        ) -> AdaptiveGestaltRuntime:
    """Bake the paired exact-elastic/rigid representations of one component."""
    elastic = build_component_mode_atlas(
        identity, stiffness, mass, physical_dofs, interface_local_dofs,
        internal_mode_count=None)
    rigid = build_rigid_gestalt_atlas(
        identity, stiffness, mass, physical_dofs, node_positions,
        reference_point)
    return AdaptiveGestaltRuntime(
        elastic, rigid,
        AdaptiveGestaltState(identity, policy, stress_envelope),
        np.asarray(mass, dtype=float), physical_damping)


def build_component_mode_atlas(
        identity: str,
        stiffness: np.ndarray,
        mass: np.ndarray,
        physical_dofs: np.ndarray,
        interface_local_dofs: np.ndarray,
        *,
        internal_mode_count: int | None = None) -> ComponentModeAtlas:
    """Build a Craig--Bampton basis without reducing interface freedoms.

    ``interface_local_dofs`` indexes the component matrices.  Every one is
    retained exactly.  An internal zero-stiffness freedom is rejected: it is
    a real articulation and must be promoted to the interface/state ABI, not
    hidden by a pseudo-inverse or an eigenvalue cutoff.

    With ``internal_mode_count=None`` all internal modes are retained and the
    transform is merely a change of basis.  This exact form is the required
    validation reference before choosing a smaller baked atlas.
    """
    k = _symmetric(stiffness, identity, "stiffness")
    m = _symmetric(mass, identity, "mass")
    if k.shape != m.shape:
        raise ValueError(f"{identity}: stiffness and mass shapes differ")
    n = k.shape[0]
    physical = np.asarray(physical_dofs, dtype=np.int64)
    boundary = np.asarray(interface_local_dofs, dtype=np.int64)
    if physical.shape != (n,) or len(np.unique(physical)) != n:
        raise ValueError(f"{identity}: physical_dofs must uniquely name each row")
    if boundary.ndim != 1 or len(np.unique(boundary)) != len(boundary):
        raise ValueError(f"{identity}: interface_local_dofs must be unique")
    if np.any(boundary < 0) or np.any(boundary >= n):
        raise ValueError(f"{identity}: interface_local_dofs are out of range")

    internal = np.setdiff1d(np.arange(n, dtype=np.int64), boundary,
                            assume_unique=True)
    if not len(boundary):
        raise ValueError(f"{identity}: at least one interface freedom is required")
    if not len(internal):
        transform = np.eye(n)[:, boundary]
        return ComponentModeAtlas(
            identity, physical, physical[boundary], transform,
            transform.T @ m @ transform, transform.T @ k @ transform,
            np.zeros(0), 0, 0.0)

    kii = k[np.ix_(internal, internal)]
    kib = k[np.ix_(internal, boundary)]
    mii = m[np.ix_(internal, internal)]
    mass_values, mass_vectors = np.linalg.eigh(mii)
    mass_scale = max(float(np.max(np.abs(mass_values))), 1.0)
    if np.min(mass_values) <= np.finfo(float).eps * mass_scale:
        raise ValueError(f"{identity}: internal mass matrix is not positive definite")
    mass_inverse_sqrt = ((mass_vectors / np.sqrt(mass_values)[None, :])
                         @ mass_vectors.T)
    standard = mass_inverse_sqrt @ kii @ mass_inverse_sqrt
    eigenvalues, vectors = np.linalg.eigh((standard + standard.T) * 0.5)
    eigen_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative_limit = NEGATIVE_EIGENVALUE_RELATIVE_TOLERANCE * eigen_scale
    if np.min(eigenvalues) < -negative_limit:
        raise ValueError(
            f"{identity}: internal stiffness has a negative mode "
            f"({float(np.min(eigenvalues)):g})")
    mechanism_limit = INTERNAL_MECHANISM_RELATIVE_TOLERANCE * eigen_scale
    if np.min(eigenvalues) <= mechanism_limit:
        raise ValueError(
            f"{identity}: internal mechanism must be an explicit interface freedom")

    available = len(internal)
    retained = available if internal_mode_count is None else int(internal_mode_count)
    if retained < 0 or retained > available:
        raise ValueError(
            f"{identity}: internal_mode_count must be between 0 and {available}")
    fixed_interface_modes = mass_inverse_sqrt @ vectors
    constraint_modes = -np.linalg.solve(kii, kib)
    transform = np.zeros((n, len(boundary) + retained), dtype=float)
    transform[np.ix_(boundary, np.arange(len(boundary)))] = np.eye(len(boundary))
    transform[np.ix_(internal, np.arange(len(boundary)))] = constraint_modes
    if retained:
        transform[np.ix_(internal,
                         np.arange(len(boundary), len(boundary) + retained))] = \
            fixed_interface_modes[:, :retained]

    discarded = eigenvalues[retained:]
    # ||K_discarded^-1|| in mass-normal coordinates.  Zero means exact/full.
    flexibility_bound = (0.0 if not len(discarded)
                         else float(1.0 / np.min(discarded)))
    return ComponentModeAtlas(
        identity=identity,
        physical_dofs=physical,
        interface_dofs=physical[boundary],
        transformation=transform,
        reduced_mass=transform.T @ m @ transform,
        reduced_stiffness=transform.T @ k @ transform,
        fixed_interface_eigenvalues=eigenvalues,
        retained_internal_modes=retained,
        discarded_flexibility_bound=flexibility_bound)


def assemble_component_mode_atlases(
        components: list[ComponentModeAtlas] | tuple[ComponentModeAtlas, ...]
        ) -> AssembledModeAtlas:
    """Assemble reduced pieces by identifying equal interface DOF names.

    Interface coordinates are allocated once globally.  Each component's
    internal modal coordinates remain private.  The reduced matrices are a
    direct finite-element-style scatter/sum, so shared mount reactions are
    equal and opposite through one coordinate rather than reconciled later.
    """
    pieces = tuple(components)
    if not pieces:
        raise ValueError("at least one component atlas is required")
    interface = np.unique(np.concatenate([p.interface_dofs for p in pieces]))
    physical = np.unique(np.concatenate([p.physical_dofs for p in pieces]))
    interface_index = {int(dof): i for i, dof in enumerate(interface)}
    physical_index = {int(dof): i for i, dof in enumerate(physical)}
    modal_offsets = []
    cursor = len(interface)
    for piece in pieces:
        modal_offsets.append(cursor)
        cursor += piece.retained_internal_modes
    size = cursor
    reduced_mass = np.zeros((size, size), dtype=float)
    reduced_stiffness = np.zeros((size, size), dtype=float)
    transformation = np.zeros((len(physical), size), dtype=float)
    assigned_physical_rows = np.zeros(len(physical), dtype=bool)
    mappings = []

    for piece, modal_offset in zip(pieces, modal_offsets):
        boundary_count = len(piece.interface_dofs)
        mapping = np.empty(piece.reduced_size, dtype=np.int64)
        mapping[:boundary_count] = [interface_index[int(dof)]
                                    for dof in piece.interface_dofs]
        mapping[boundary_count:] = np.arange(
            modal_offset, modal_offset + piece.retained_internal_modes)
        mappings.append(mapping)
        reduced_mass[np.ix_(mapping, mapping)] += piece.reduced_mass
        reduced_stiffness[np.ix_(mapping, mapping)] += piece.reduced_stiffness

        for local_row, dof in enumerate(piece.physical_dofs):
            row = physical_index[int(dof)]
            value = np.zeros(size, dtype=float)
            value[mapping] = piece.transformation[local_row]
            if assigned_physical_rows[row]:
                if not np.allclose(transformation[row], value,
                                   rtol=1.0e-10, atol=1.0e-12):
                    raise ValueError(
                        f"physical DOF {int(dof)} has incompatible component "
                        "recovery maps; it must be a shared interface")
            else:
                transformation[row] = value
                assigned_physical_rows[row] = True

    return AssembledModeAtlas(
        components=pieces,
        interface_dofs=interface,
        physical_dofs=physical,
        transformation=transformation,
        reduced_mass=reduced_mass,
        reduced_stiffness=reduced_stiffness,
        component_reduced_indices=tuple(mappings))
