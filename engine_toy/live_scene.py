"""THE SCENE, LIVE: the game engine ticking, the frame solved every tick.

    python live_scene.py

    drag            orbit          wheel / Z X    zoom
    SPACE           fire           hold G         centerline proof load
    R               reset          1 / 2          colour: yield | assembly
    ESC             quit

WHAT THIS IS AND WHAT THE OTHER TWO WERE NOT.

`firing_frame_trial` solves a static instant. `platform_ringdown`
precomputes a decay and plays it back. Both are films. Neither is the
machine running, and neither can be steered.

This advances the complete six-DOF-per-node Timoshenko beam assembly
every tick. The average-acceleration Newmark step is implicit and stable
for ideal rigid-link frequencies far above the display rate. Dropping
those modes would make an attractive animation rather than beam evolution;
the implicit solve retains the whole assembled system.

AND IT IS VECTORISED, because it has to be. A hundred and sixty-four
thousand member-strain evaluations a second is not a Python loop: the
member DOF indices, the local frames and the section properties are
gathered into arrays once, and a frame is four einsums and a scatter.
That is what the constraint tokens were for.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import firing_frame_trial as fft
from frame_solver import (DOF_PER_NODE, MECHANISM_EIGENVALUE, FrameSolver,
                          _element_frame)
from milspec import MATERIAL_BY_KEY


STRUCTURAL_DAMPING = 0.015
MODES = 40


class LiveStructure:
    """The authoritative full-scale beam solve, evolved every tick.

    ``reference_basis_free`` is a factorisation of the complete assembled
    effective inverse: it contains every elastic and mechanism freedom.  It
    is the truth solve from which cheaper matrices may later be projected and
    baked for other scales; it is not itself a reduced gameplay model.
    """

    def __init__(self, document: dict, *, modes: int = MODES,
                 solver: FrameSolver | None = None,
                 reference: dict | None = None,
                 settle_from_authored: bool = False):
        self.solver = solver or FrameSolver(document=document, loads={})
        # A caller that has already paid for the full reference assembly may
        # hand it through. This is the same K/M/eigenbasis, not a reduced or
        # separately fitted runtime model.
        reference_solve = reference if reference is not None else self.solver.modes(
            count=modes)
        # `modes` remains accepted for old callers, but no longer selects
        # which physics exists. Every elastic and mechanism DOF is retained.
        self.shapes = np.ascontiguousarray(reference_solve["shapes"])
        self.mechanism_shapes = np.ascontiguousarray(
            reference_solve["mechanism_shapes"])
        self.omega = 2.0 * math.pi * np.asarray(
            reference_solve["all_frequencies_hz"], dtype=float)
        self.mass = np.asarray(reference_solve["lumped_mass"], dtype=float)
        self.stiffness = np.asarray(
            reference_solve["stiffness_matrix"], dtype=float)
        self.free = np.asarray(reference_solve["free"], dtype=np.int64)
        n_dof = len(self.mass)
        applied = self.solver.applied_force_vector()
        if settle_from_authored:
            # The caller explicitly wants the assembled geometry.  Rebuilding
            # and recovering a complete static frame only to throw its result
            # away doubled startup work on the station.
            self.u_static = np.zeros(n_dof)
        else:
            kff = self.stiffness[np.ix_(self.free, self.free)]
            self.u_static = np.zeros(n_dof)
            try:
                self.u_static[self.free] = np.linalg.solve(
                    kff, applied[self.free])
            except np.linalg.LinAlgError:
                self.u_static[self.free] = np.linalg.lstsq(
                    kff, applied[self.free], rcond=None)[0]
        # Compatibility record: LiveStructure consumes only displacement.
        # The expensive per-member static recovery remains FrameSolver.solve's
        # explicit job when a caller actually requests it.
        self.rest = {"displacement": self.u_static.reshape(-1, DOF_PER_NODE)}
        self.dynamic_displacement = np.zeros(n_dof)
        self.velocity = np.zeros(n_dof)
        self.acceleration = np.zeros(n_dof)

        # Exact constant modal damping mapped back to physical coordinates.
        # All elastic eigenvectors participate; mechanisms are resisted by
        # their declared graph joints rather than invented modal damping.
        phi = self.shapes[self.free]
        self.mass_free = self.mass[self.free]
        weighted = self.mass_free[:, None] * phi
        self.damping_free = ((weighted * (2.0 * STRUCTURAL_DAMPING
                                           * self.omega)[None, :])
                             @ weighted.T)
        self.stiffness_free = self.stiffness[np.ix_(self.free, self.free)]
        # COMPLETE reference basis: elastic columns plus every zero-stiffness
        # mechanism column. It spans `free` exactly. The implicit effective
        # inverse is diagonal in this basis, so the expensive reference
        # eigensolve also bakes the matrix that reproduces the reference step
        # at any outer dt without refactorising a dense matrix every frame.
        self.reference_basis_free = np.ascontiguousarray(np.column_stack(
            (self.shapes[self.free], self.mechanism_shapes[self.free])))
        self.reference_omega = np.concatenate(
            (self.omega, np.zeros(self.mechanism_shapes.shape[1])))
        self.reference_damping = np.concatenate((
            2.0 * STRUCTURAL_DAMPING * self.omega,
            np.zeros(self.mechanism_shapes.shape[1])))
        if self.reference_basis_free.shape != (len(self.free), len(self.free)):
            raise ValueError("the baked reference basis does not span every free DOF")
        # The complete (not truncated) reference coordinates are the runtime
        # state.  In this basis M, C and K are diagonal, so a constitutive
        # microstep is O(n) plus the handful of joint endpoint rows. Keeping
        # physical coordinates as the inner-loop state made every oil step
        # perform dense O(n^2) products and looked exactly like a frozen UI.
        self.reference_q = np.zeros(len(self.free))
        self.reference_qdot = np.zeros(len(self.free))
        self.reference_qddot = np.zeros(len(self.free))
        self._full_reference_basis_free = self.reference_basis_free
        self._full_reference_omega = self.reference_omega
        self._full_reference_damping = self.reference_damping
        self._gestalt_subspace = None
        self._gestalt_diagonal = None
        self.gestalt_dormant = False
        # Four exchanges per 60-Hz outer step resolve recoil components.
        # The implicit beam step does not subdivide on artificial rigid-link
        # eigenfrequencies, which extend into MHz on this graph.
        self.max_dt = 1.0 / 240.0

        # ---- everything a strain needs, gathered once ----------------
        members, dof, endpoint_transform, rot, length = [], [], [], [], []
        outer_y, outer_z, rel_ax, rel_bd, rel_sh = [], [], [], [], []
        material_capacity_member = []
        for e in self.solver.members:
            ia, ib = self.solver.index[e["a"]], self.solver.index[e["b"]]
            a, b = self.solver.position[ia], self.solver.position[ib]
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            members.append(e)
            dof.append((self.solver._master_dofs(ia),
                        self.solver._master_dofs(ib)))
            endpoint_transform.append((
                self.solver.solver_endpoint_transform[ia],
                self.solver.solver_endpoint_transform[ib]))
            rot.append(_element_frame(e, a, b))
            length.append(L)
            damage = e.get("damage") or {}
            outer_y.append(float(damage.get(
                "section_outer_y_m", e.get("radius", 0.012))))
            outer_z.append(float(damage.get(
                "section_outer_z_m", e.get("radius", 0.012))))
            r = set(e.get("freedoms_released") or ())
            rel_ax.append("x" in r)
            rel_bd.append(bool(r & {"ry", "rz"}))
            rel_sh.append(bool(r & {"y", "z", "rx"}))
            material_capacity_member.append(not bool(e.get("rigid")))
        self.members = members
        self.dof = np.asarray(dof, dtype=np.int64)
        self.member_endpoint_transform = np.asarray(
            endpoint_transform, dtype=float)
        self.rot = np.asarray(rot, dtype=float)
        self.length = np.asarray(length, dtype=float)
        self.outer_y = np.asarray(outer_y, dtype=float)
        self.outer_z = np.asarray(outer_z, dtype=float)
        self.free_axial = np.asarray(rel_ax, dtype=bool)
        self.free_bend = np.asarray(rel_bd, dtype=bool)
        self.free_shear = np.asarray(rel_sh, dtype=bool)
        self.material_capacity_member = np.asarray(
            material_capacity_member, dtype=bool)
        yld, emod, gmod = [], [], []
        for e in members:
            d = e.get("damage") or {}
            mat = MATERIAL_BY_KEY.get(d.get("material", "4130n"),
                                      MATERIAL_BY_KEY["4130n"])
            yld.append(float(getattr(mat, "yield_pa", 460e6)))
            emod.append(float(d.get("youngs_modulus_pa", mat.youngs_pa)))
            gmod.append(float(d.get("shear_modulus_pa", mat.shear_pa)))
        self.yield_pa = np.asarray(yld)
        self.e_pa = np.asarray(emod)
        self.g_pa = np.asarray(gmod)
        self.identities = [e["identity"] for e in members]
        self.force = np.zeros(n_dof)
        self.actuator_force = np.zeros(n_dof)
        self._last_joint_force = np.zeros(n_dof)
        self._adaptive_gestalt = None
        self._adaptive_gestalts = {}
        self._adaptive_orchestration = None
        self._gestalt_projection_cache = {}
        self._dormant_partition_ids = frozenset()
        # Advance about the chosen reference displacement without discarding
        # any unresolved load.  For a nonsingular static equilibrium this is
        # numerically zero.  For an intentionally free mechanism, the least-
        # squares static solve cannot balance force along the nullspace; that
        # residual is precisely what must accelerate the mechanism.  Setting
        # this to zero made gravity disappear as soon as a real slide/pin was
        # released.
        self.base_force = applied - self.stiffness @ self.u_static
        self.base_modal_force = (
            self.reference_basis_free.T @ self.base_force[self.free])
        self._static_stiffness_force = self.stiffness @ self.u_static
        self._static_elastic_energy_j = 0.5 * float(
            self.u_static @ self._static_stiffness_force)
        self._static_stiffness_modal = (
            self.reference_basis_free.T
            @ self._static_stiffness_force[self.free])
        self.last_static_residual_free = np.zeros(len(self.free))
        self.last_coupled_steps = 0
        self.minimum_coupled_dt_s = math.inf
        self.minimum_coupled_limiter = None
        self.elapsed_s = 0.0
        self.solve_sequence = 0

        # The graph's travelling joints already have one constitutive
        # engine.  It owns only compression and force; this structure owns
        # the masses, velocities and distribution of those forces.
        from graph_physics import GraphJointForces
        self.joint_forces = GraphJointForces(document, linear_embedded=True)
        fixed_nodes = {
            node["identity"] for node in document["nodes"]
            if node.get("fixed_to")
            or node.get("kind") == "structural-body-pin-frame-foot"
        }
        self.joint_forces.configure_coupled_mass(
            self.mass.reshape(-1, DOF_PER_NODE), fixed_nodes)
        self.joint_edges = self.joint_forces.edges
        self._edge_by_identity = {
            edge["identity"]: edge for edge in self.solver.document["edges"]}
        # Compatibility for existing telemetry and reset controls.  The bank
        # is still the constitutive owner of its compression state.
        self.joints = self.joint_forces.bank
        self._joint_rest_force = self.joint_forces.rest_force
        joint_nodes = sorted({self.solver.index[name]
                              for edge in self.joint_edges
                              for name in (edge["a"], edge["b"])})
        self._joint_nodes = np.asarray(joint_nodes, dtype=np.int64)
        self._rebuild_joint_translation_basis()
        self._settled_positions = (
            self.solver.position
            + self.solver.endpoint_motion(self.u_static)[:, :3])

    def _rebuild_joint_translation_basis(self) -> None:
        basis = []
        free_row = {int(dof): i for i, dof in enumerate(self.free)}
        for node_i in self._joint_nodes:
            master_dofs = self.solver._master_dofs(int(node_i))
            rows = np.asarray([free_row.get(int(dof), -1)
                               for dof in master_dofs], dtype=np.int64)
            master_basis = np.zeros(
                (DOF_PER_NODE, self.reference_basis_free.shape[1]))
            valid = rows >= 0
            if np.any(valid):
                master_basis[valid] = self.reference_basis_free[rows[valid]]
            basis.append(self.solver.solver_endpoint_transform[node_i, :3]
                         @ master_basis)
        self._joint_translation_basis = np.asarray(basis, dtype=float)

    def configure_gestalt_subspace(self, subspace) -> None:
        """Install a pre-baked optional dormant subspace without enabling it."""
        if not np.array_equal(np.asarray(subspace.free_dofs), self.free):
            raise ValueError("gestalt subspace does not match live free coordinates")
        self._gestalt_subspace = subspace
        self._gestalt_diagonal = subspace.mass_orthonormal_basis(
            MECHANISM_EIGENVALUE)

    def _switch_runtime_basis(self, basis: np.ndarray,
                              omega: np.ndarray) -> None:
        physical = self.reference_basis_free @ np.column_stack((
            self.reference_q, self.reference_qdot, self.reference_qddot))
        weighted = self.mass_free[:, None] * physical
        coordinates = basis.T @ weighted
        self.reference_basis_free = np.ascontiguousarray(basis)
        self.reference_omega = np.asarray(omega, dtype=float)
        self.reference_damping = 2.0 * STRUCTURAL_DAMPING * self.reference_omega
        self.reference_q = coordinates[:, 0]
        self.reference_qdot = coordinates[:, 1]
        self.reference_qddot = coordinates[:, 2]
        self.base_modal_force = (
            self.reference_basis_free.T @ self.base_force[self.free])
        self._static_stiffness_modal = (
            self.reference_basis_free.T
            @ self._static_stiffness_force[self.free])
        self._rebuild_joint_translation_basis()
        recovered = self.reference_basis_free @ coordinates
        self.dynamic_displacement[self.free] = recovered[:, 0]
        self.velocity[self.free] = recovered[:, 1]
        self.acceleration[self.free] = recovered[:, 2]

    def set_gestalt_dormant(self, dormant: bool) -> None:
        """Reversibly select the configured rigid-component global basis."""
        requested = bool(dormant)
        if requested == self.gestalt_dormant:
            return
        if requested:
            if self._gestalt_diagonal is None:
                raise RuntimeError("no gestalt subspace has been configured")
            self._switch_runtime_basis(
                self._gestalt_diagonal["basis_free"],
                self._gestalt_diagonal["omega"])
        else:
            self._switch_runtime_basis(
                self._full_reference_basis_free, self._full_reference_omega)
            self.reference_damping = self._full_reference_damping
        self.gestalt_dormant = requested

    def configure_adaptive_partition(self, partition, policy, envelope) -> None:
        """Attach one generic certified partition to automatic residency."""
        if self._adaptive_gestalts:
            raise RuntimeError("an adaptive partition is already configured")
        from component_mode_atlas import (
            AdaptiveGestaltState, assemble_component_partition,
            build_global_gestalt_subspace, build_rigid_gestalt_atlas)
        assembled = assemble_component_partition(self.solver, partition)
        subspace = build_global_gestalt_subspace(
            self.solver, self.stiffness, self.mass, self.free, [partition])
        self.configure_gestalt_subspace(subspace)
        local_mass = assembled["mass_matrix"]
        position = assembled["node_positions"]
        diagonal = np.diag(local_mass)
        point_mass = np.asarray([
            np.mean(diagonal[i * 6:i * 6 + 3]) for i in range(len(position))])
        total = float(np.sum(point_mass))
        reference = ((point_mass @ position) / total
                     if total > 0.0 else np.mean(position, axis=0))
        rigid = build_rigid_gestalt_atlas(
            partition.identity, assembled["stiffness_matrix"], local_mass,
            assembled["physical_dofs"], position, reference)
        self._adaptive_gestalt = {
            "partition": partition,
            "assembled": assembled,
            "rigid": rigid,
            "state": AdaptiveGestaltState(partition.identity, policy, envelope),
            "last_wrench": np.zeros(len(assembled["interface_local_dofs"])),
        }
        self._adaptive_gestalts = {partition.identity: self._adaptive_gestalt}

    def configure_adaptive_orchestration(self, orchestration, configurations) -> None:
        """Install generic per-vertex policies/envelopes on a component graph."""
        if self._adaptive_gestalts:
            raise RuntimeError("adaptive residency is already configured")
        from component_mode_atlas import (
            AdaptiveGestaltState, assemble_component_partition,
            build_rigid_gestalt_atlas)
        by_identity = {part.identity: part for part in orchestration.partitions}
        entries = {}
        for identity, (policy, envelope) in configurations.items():
            partition = by_identity[identity]
            assembled = assemble_component_partition(self.solver, partition)
            local_mass = assembled["mass_matrix"]
            position = assembled["node_positions"]
            diagonal = np.diag(local_mass)
            point_mass = np.asarray([
                np.mean(diagonal[i * 6:i * 6 + 3])
                for i in range(len(position))])
            total = float(np.sum(point_mass))
            reference = ((point_mass @ position) / total
                         if total > 0.0 else np.mean(position, axis=0))
            rigid = build_rigid_gestalt_atlas(
                identity, assembled["stiffness_matrix"], local_mass,
                assembled["physical_dofs"], position, reference)
            entries[identity] = {
                "partition": partition, "assembled": assembled,
                "rigid": rigid,
                "state": AdaptiveGestaltState(identity, policy, envelope),
                "last_wrench": np.zeros(
                    len(assembled["interface_local_dofs"])),
            }
        self._adaptive_gestalts = entries
        self._adaptive_orchestration = orchestration

    def _set_dormant_partitions(self, identities) -> None:
        selected = frozenset(identities)
        if selected == self._dormant_partition_ids:
            return
        if not selected:
            self._switch_runtime_basis(
                self._full_reference_basis_free, self._full_reference_omega)
            self.reference_damping = self._full_reference_damping
            self.gestalt_dormant = False
            self._dormant_partition_ids = selected
            return
        cached = self.prime_gestalt_projection(selected)
        self._gestalt_subspace, self._gestalt_diagonal = cached
        self._switch_runtime_basis(
            self._gestalt_diagonal["basis_free"],
            self._gestalt_diagonal["omega"])
        self.gestalt_dormant = True
        self._dormant_partition_ids = selected

    def prime_gestalt_projection(self, identities):
        """Bake/cache one graph-selected dormant set without switching state."""
        selected = frozenset(identities)
        cached = self._gestalt_projection_cache.get(selected)
        if cached is not None:
            return cached
        from component_mode_atlas import build_global_gestalt_subspace
        partitions = [self._adaptive_gestalts[identity]["partition"]
                      for identity in sorted(selected)]
        subspace = build_global_gestalt_subspace(
            self.solver, self.stiffness, self.mass, self.free, partitions)
        diagonal = subspace.mass_orthonormal_basis(MECHANISM_EIGENVALUE)
        cached = (subspace, diagonal)
        self._gestalt_projection_cache[selected] = cached
        return cached

    def update_gestalt_residency(self, dt: float):
        """Evaluate graph vertices, then project the independent dormant set."""
        if not self._adaptive_gestalts:
            return "unconfigured"
        response = (self.member_response() if any(
            entry["state"].elastic for entry in self._adaptive_gestalts.values())
                    else None)
        member_index = {identity: i for i, identity in enumerate(self.identities)}
        transitions = {}
        for identity, adaptive in self._adaptive_gestalts.items():
            assembled = adaptive["assembled"]
            physical = assembled["physical_dofs"]
            boundary = assembled["interface_local_dofs"]
            local_u = self.displacement()[physical]
            local_a = self.acceleration[physical]
            effort = (assembled["mass_matrix"] @ local_a
                      + assembled["stiffness_matrix"] @ local_u)
            applied = (self.base_force + self.force + self.actuator_force
                       + self._last_joint_force)[physical]
            wrench = np.abs(effort[boundary]) + np.abs(applied[boundary])
            previous = adaptive["last_wrench"]
            rate = (float(np.linalg.norm(wrench - previous)) / dt
                    if dt > 0.0 else math.inf)
            adaptive["last_wrench"] = wrench
            state = adaptive["state"]
            metrics = {"interface_wrench_rate_per_s": rate}
            if state.elastic:
                selected = [member_index[name]
                            for name in adaptive["partition"].member_identities
                            if name in member_index]
                peak = (float(np.max(response["stress_pa"][selected]))
                        if selected else 0.0)
                local_v = self.velocity[physical]
                transform = adaptive["rigid"].transformation
                mass = assembled["mass_matrix"]
                rigid_v = np.linalg.solve(
                    transform.T @ mass @ transform,
                    transform.T @ mass @ local_v)
                residual_v = local_v - transform @ rigid_v
                kinetic = .5 * float(residual_v @ mass @ residual_v)
                metrics.update(solved_peak_stress_pa=peak,
                               solved_stress_error_bound_pa=0.0,
                               solved_internal_kinetic_j=kinetic)
            transitions[identity] = state.observe(dt, wrench, **metrics)

        dormant = {identity for identity, entry in self._adaptive_gestalts.items()
                   if not entry["state"].elastic}
        if self._adaptive_orchestration is not None:
            accepted = set()
            order = sorted(dormant, key=lambda identity: (
                identity not in self._dormant_partition_ids, identity))
            for identity in order:
                welded = set(self._adaptive_orchestration.weld_neighbors(identity))
                if accepted.isdisjoint(welded):
                    accepted.add(identity)
                else:
                    state = self._adaptive_gestalts[identity]["state"]
                    state.elastic = True
                    state.quiet_time_s = 0.0
                    transitions[identity] = "weld-conflict"
            dormant = accepted
        self._set_dormant_partitions(dormant)
        if len(transitions) == 1:
            return next(iter(transitions.values()))
        return transitions

    def apply_baked_reference_inverse(self, rhs: np.ndarray,
                                      dt: float) -> np.ndarray:
        """Apply the full reference Newmark matrix inverse to ``rhs``.

        The dense effective matrix is represented by its complete spectral
        factorisation.  No column is truncated: this is algebraically the
        full physical-coordinate solve, merely baked so a frame does not
        refactor the same matrix.  Reduced matrices for other simulation
        scales must be derived from and validated against this operator.
        """
        beta, gamma = 0.25, 0.5
        denominator = (1.0 + gamma * dt * self._full_reference_damping
                       + beta * dt * dt * self._full_reference_omega ** 2)
        baked_rhs = self._full_reference_basis_free.T @ rhs
        return self._full_reference_basis_free @ (baked_rhs / denominator)

    # ------------------------------------------------------------------
    def apply(self, node: str, vector) -> None:
        self.apply_wrench(node, force=vector)

    def apply_wrench(self, node: str, force=(0.0, 0.0, 0.0),
                     moment=(0.0, 0.0, 0.0)) -> None:
        """Replace the external load with one nodal force and moment."""
        i = self.solver.index[node]
        self.force[:] = 0.0
        base = i * DOF_PER_NODE
        self.force[base:base + 3] = force
        self.force[base + 3:base + 6] = moment

    def clear(self) -> None:
        self.force[:] = 0.0

    def set_edge_axial_forces(self, force_by_identity: dict[str, float]) -> None:
        """Replace forces supplied by external actuator engines.

        Positive force extends an edge. The actuator owns pressure, flow and
        force; this beam engine owns distribution into endpoint wrenches.
        """
        self.actuator_force[:] = 0.0
        if not force_by_identity:
            return
        position = self.positions()
        identities = tuple(force_by_identity)
        edges = [self._edge_by_identity[identity]
                 for identity in identities]
        ia = np.fromiter((self.solver.index[edge["a"]] for edge in edges),
                         dtype=np.int64, count=len(edges))
        ib = np.fromiter((self.solver.index[edge["b"]] for edge in edges),
                         dtype=np.int64, count=len(edges))
        delta = position[ib] - position[ia]
        length = np.linalg.norm(delta, axis=1)
        bad = np.flatnonzero(length <= 1.0e-12)
        if bad.size:
            raise ValueError(
                f"{identities[int(bad[0])]}: actuator endpoints coincide")
        axis = delta / length[:, None]
        value = np.fromiter((float(force_by_identity[identity])
                             for identity in identities), dtype=float,
                            count=len(identities))
        endpoint_wrench = np.zeros((len(edges), DOF_PER_NODE), dtype=float)
        endpoint_wrench[:, :3] = axis * value[:, None]
        # P.T maps a surface-point wrench to its gestalt master. np.add.at
        # retains exact accumulation when several actuators share a master.
        mapped_a = np.einsum(
            "nji,nj->ni", self.solver.solver_endpoint_transform[ia],
            -endpoint_wrench)
        mapped_b = np.einsum(
            "nji,nj->ni", self.solver.solver_endpoint_transform[ib],
            endpoint_wrench)
        nodal = self.actuator_force.reshape(-1, DOF_PER_NODE)
        np.add.at(nodal, self.solver.solver_master_index[ia], mapped_a)
        np.add.at(nodal, self.solver.solver_master_index[ib], mapped_b)

    def reset(self) -> None:
        self.dynamic_displacement[:] = 0.0
        self.velocity[:] = 0.0
        self.acceleration[:] = 0.0
        self.reference_q[:] = 0.0
        self.reference_qdot[:] = 0.0
        self.reference_qddot[:] = 0.0
        self.joints.compression[:] = 0.0
        self.joints.arrays["velocity_m_s"][:] = 0.0
        self.joints.step(0.0)
        self.elapsed_s = 0.0
        self.solve_sequence = 0
        self.clear()
        self.actuator_force[:] = 0.0
        self._last_joint_force[:] = 0.0

    def _coupled_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct only the nodes touched by a constitutive element."""
        position = self._settled_positions.copy()
        velocity = np.zeros_like(position)
        if len(self._joint_nodes):
            position[self._joint_nodes] += np.einsum(
                "nij,j->ni", self._joint_translation_basis,
                self.reference_q)
            velocity[self._joint_nodes] = np.einsum(
                "nij,j->ni", self._joint_translation_basis,
                self.reference_qdot)
        return position, velocity

    def _joint_forces(self, dt: float, *, coupled: bool = False,
                      position=None, velocity=None) -> np.ndarray:
        """Evaluate joint laws and scatter their axial forces to bodies."""
        if not self.joint_edges:
            return np.zeros_like(self.force)
        if position is None or velocity is None:
            position, velocity = self._coupled_kinematics()
        nodal_xyz = self.joint_forces.evaluate(
            dt, position, velocity, coupled=coupled)
        nodal = np.zeros_like(self.force)
        nodal.reshape(-1, DOF_PER_NODE)[:, :3] = nodal_xyz
        return nodal

    def substep_plan(self, dt: float) -> tuple:
        """How many of its own substeps this takes out of the dt it was
        given -- the same shape as engine_cycle_sim._clutch_substep_plan
        and drivetrain_graph's stable sub-dt.

        THE DT IS NOT MINE TO CHOOSE. One outer step is handed to every
        subsystem in the scene and each subdivides THAT, by whatever its
        own stiffest term requires. Subsystems running on clocks of
        their own would drift apart from one another between exchanges,
        which is a different machine every frame."""
        n = max(1, int(math.ceil(dt / self.max_dt)))
        return n, dt / n

    def step(self, dt: float) -> None:
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("structure dt must be finite and nonnegative")
        if self._adaptive_gestalts:
            self.update_gestalt_residency(dt)
        beta, gamma = 0.25, 0.5
        remaining = float(dt)
        coupled_steps = 0
        minimum_h = math.inf
        minimum_limiter = None
        last_total = self.base_force + self.force + self.actuator_force
        while remaining > max(1.0e-15, abs(dt) * 1.0e-12):
            maximum_h = min(self.max_dt, remaining)
            position, velocity_xyz = self._coupled_kinematics()
            h = self.joint_forces.coupled_step_size(
                maximum_h, position, velocity_xyz)
            if not math.isfinite(h) or h <= 0.0:
                raise FloatingPointError(f"invalid coupled structure step {h!r}")
            joint_force = self._joint_forces(
                h, coupled=True, position=position, velocity=velocity_xyz)
            self._last_joint_force = joint_force
            changing_force = self.force + self.actuator_force + joint_force
            total = self.base_force + changing_force
            q_predict = (self.reference_q + h * self.reference_qdot
                         + h * h * (0.5 - beta) * self.reference_qddot)
            qdot_predict = (self.reference_qdot
                            + h * (1.0 - gamma)
                            * self.reference_qddot)
            force_free = changing_force[self.free]
            nonzero = np.flatnonzero(force_free)
            modal_force = self.base_modal_force.copy()
            if nonzero.size:
                modal_force += (self.reference_basis_free[nonzero].T
                                @ force_free[nonzero])
            modal_rhs = (modal_force
                         - self.reference_damping * qdot_predict
                         - self.reference_omega ** 2 * q_predict)
            denominator = (1.0 + gamma * h * self.reference_damping
                           + beta * h * h * self.reference_omega ** 2)
            qddot_new = modal_rhs / denominator
            self.reference_q = q_predict + beta * h * h * qddot_new
            self.reference_qdot = qdot_predict + gamma * h * qddot_new
            self.reference_qddot = qddot_new
            last_total = total
            remaining = max(0.0, remaining - h)
            coupled_steps += 1
            if h < minimum_h:
                minimum_limiter = dict(
                    self.joint_forces.last_step_limiter or {})
            minimum_h = min(minimum_h, h)
            # A runaway used to look like a frozen window.  This guard changes
            # no accepted trajectory; it turns an impossible integration into
            # an explicit diagnostic instead of burning forever.
            if coupled_steps > 1_000_000:
                raise RuntimeError(
                    "coupled structure step exceeded 1,000,000 subdivisions")
        # Materialise the complete physical state once per caller step for
        # rendering, strain recovery and public diagnostics. No basis column
        # is omitted; only the timing of this algebra has changed.
        physical_state = self.reference_basis_free @ np.column_stack((
            self.reference_q, self.reference_qdot, self.reference_qddot))
        self.dynamic_displacement[self.free] = physical_state[:, 0]
        self.velocity[self.free] = physical_state[:, 1]
        self.acceleration[self.free] = physical_state[:, 2]
        # Exact physical residual from complete modal coordinates. For a
        # mass-normal complete basis, f = M Phi (Phi.T f). This avoids two
        # dense physical K/C products without changing the vector at all.
        final_force_free = last_total[self.free]
        nonzero = np.flatnonzero(final_force_free)
        final_modal_force = (
            self.reference_basis_free[nonzero].T @ final_force_free[nonzero]
            if nonzero.size else np.zeros(len(self.free)))
        modal_static_residual = (
            final_modal_force
            - self.reference_damping * self.reference_qdot
            - self.reference_omega ** 2 * self.reference_q)
        self.last_static_residual_free = (
            self.mass_free
            * (self.reference_basis_free @ modal_static_residual))
        self.last_coupled_steps = coupled_steps
        self.minimum_coupled_dt_s = minimum_h
        self.minimum_coupled_limiter = minimum_limiter
        self.elapsed_s += float(dt)
        self.solve_sequence += 1

    def displacement(self) -> np.ndarray:
        return self.u_static + self.dynamic_displacement

    def member_response(self) -> dict[str, np.ndarray]:
        """Vectorised strain, stress and utilisation for every beam member.

        Four einsums over the whole structure: the per-member loop that
        the static recovery uses is right for one solve and is sixty
        times too slow for sixty frames a second."""
        u = self.displacement()
        master = u[self.dof]                              # (n, 2, 6)
        U = np.einsum("neij,nej->nei", self.member_endpoint_transform,
                      master)
        da = np.einsum("nij,nj->ni", self.rot, U[:, 0, 0:3])
        ra = np.einsum("nij,nj->ni", self.rot, U[:, 0, 3:6])
        db = np.einsum("nij,nj->ni", self.rot, U[:, 1, 0:3])
        rb = np.einsum("nij,nj->ni", self.rot, U[:, 1, 3:6])
        axial = np.where(self.free_axial, 0.0,
                         (db[:, 0] - da[:, 0]) / self.length)
        curv_y = (rb[:, 1] - ra[:, 1]) / self.length
        curv_z = (rb[:, 2] - ra[:, 2]) / self.length
        # Maximum corner-fibre strain. For a round tube both extents are
        # equal and the exact radial maximum is the curvature magnitude;
        # shaped sections use the conservative rectangular envelope.
        shaped = np.abs(self.outer_y - self.outer_z) > 1.0e-12
        curv = np.where(
            shaped,
            np.abs(curv_y) * self.outer_z
            + np.abs(curv_z) * self.outer_y,
            np.hypot(curv_y, curv_z) * self.outer_y)
        bending = np.where(self.free_bend, 0.0, curv)
        shear_y = (db[:, 1] - da[:, 1]) / self.length \
            - (ra[:, 2] + rb[:, 2]) / 2.0
        shear_z = (db[:, 2] - da[:, 2]) / self.length \
            + (ra[:, 1] + rb[:, 1]) / 2.0
        twist = (rb[:, 0] - ra[:, 0]) / self.length
        shear = np.sqrt(shear_y ** 2 + shear_z ** 2
                        + (twist * np.maximum(
                            self.outer_y, self.outer_z)) ** 2)
        shear = np.where(self.free_shear, 0.0, shear)
        normal_stress = (np.abs(axial) + np.abs(bending)) * self.e_pa
        shear_stress = np.abs(shear) * self.g_pa
        stress = np.sqrt(normal_stress ** 2 + 3.0 * shear_stress ** 2)
        utilisation = np.where(
            self.material_capacity_member,
            stress / np.maximum(self.yield_pa, 1.0), 0.0)
        return {"axial_strain": axial,
                "bending_surface_strain": bending,
                "shear_strain": shear,
                "normal_stress_pa": normal_stress,
                "shear_stress_pa": shear_stress,
                "stress_pa": stress,
                "utilisation": utilisation}

    def utilisation(self) -> np.ndarray:
        """Every member's fraction of its own yield, this instant."""
        return self.member_response()["utilisation"]

    def energy_state(self) -> dict[str, float]:
        """Mechanical energy visible at the beam/reference boundary."""
        dynamic_u = self.dynamic_displacement
        dynamic_energy = 0.5 * float(
            (self.reference_omega * self.reference_q)
            @ (self.reference_omega * self.reference_q))
        cross_energy = float(
            self._static_stiffness_modal @ self.reference_q)
        return {
            "kinetic_j": 0.5 * float(
                self.reference_qdot @ self.reference_qdot),
            "beam_elastic_j": (self._static_elastic_energy_j
                               + cross_energy + dynamic_energy),
            "dynamic_beam_elastic_j": dynamic_energy,
            "reference_beam_elastic_j": self._static_elastic_energy_j,
            # Potential change from the constant residual load about the
            # reference pose. Nonconservative joint work remains owned by the
            # constitutive bank and is intentionally not relabelled as stored
            # beam energy.
            "constant_load_potential_change_j": -float(
                self.base_force @ dynamic_u),
        }

    def inject_solver_stats(self, response: dict[str, np.ndarray] | None = None
                            ) -> dict:
        """Attach the current authoritative solve to graph node/edge objects.

        These dictionaries are the same objects consumed by the renderer and
        controls.  Keeping telemetry on them makes a probe, HCU controller or
        later baked-scale comparison inspect one state instead of joining an
        unrelated table by hand.
        """
        response = response or self.member_response()
        displacement, dynamic, velocity, acceleration = \
            self.solver.endpoint_motion_batch(np.vstack((
                self.displacement(), self.dynamic_displacement,
                self.velocity, self.acceleration)))
        position = self.solver.position + displacement[:, :3]
        for i, node in enumerate(self.solver.document["nodes"]):
            node["solver_stats"] = {
                "sequence": self.solve_sequence,
                "time_s": self.elapsed_s,
                "position_m": position[i].tolist(),
                "displacement_m": displacement[i, :3].tolist(),
                "rotation_rad": displacement[i, 3:].tolist(),
                "dynamic_displacement_m": dynamic[i, :3].tolist(),
                "velocity_m_s": velocity[i, :3].tolist(),
                "angular_velocity_rad_s": velocity[i, 3:].tolist(),
                "acceleration_m_s2": acceleration[i, :3].tolist(),
                "angular_acceleration_rad_s2": acceleration[i, 3:].tolist(),
                "speed_m_s": float(np.linalg.norm(velocity[i, :3])),
                "structural_participation": bool(
                    self.solver.structural_node[i]),
            }
        for adaptive in self._adaptive_gestalts.values():
            representation = ("rigid-gestalt" if self.gestalt_dormant
                              and adaptive["partition"].identity
                              in self._dormant_partition_ids
                              else "elastic-local")
            for node_i in adaptive["partition"].node_indices:
                stats = self.solver.document["nodes"][int(node_i)]["solver_stats"]
                stats["adaptive_component"] = adaptive["partition"].identity
                stats["solver_representation"] = representation
        # A condensed machine's internal objects have no independent beam
        # coordinates.  They inherit their one gestalt body's rigid small-
        # displacement kinematics for rendering and inspection.
        for i, node in enumerate(self.solver.document["nodes"]):
            target = node.get("solver_condensed_into")
            if not target:
                continue
            j = self.solver.index[target]
            offset = self.solver.position[i] - self.solver.position[j]
            parent_u = displacement[j, :3]
            parent_r = displacement[j, 3:]
            parent_dynamic_u = dynamic[j, :3]
            parent_dynamic_r = dynamic[j, 3:]
            parent_v = velocity[j, :3]
            parent_w = velocity[j, 3:]
            parent_a = acceleration[j, :3]
            parent_alpha = acceleration[j, 3:]
            inherited_u = parent_u + np.cross(parent_r, offset)
            inherited_dynamic_u = (parent_dynamic_u
                                   + np.cross(parent_dynamic_r, offset))
            inherited_v = parent_v + np.cross(parent_w, offset)
            inherited_a = (parent_a + np.cross(parent_alpha, offset)
                           + np.cross(parent_w, np.cross(parent_w, offset)))
            stats = node["solver_stats"]
            stats.update({
                "position_m": (self.solver.position[i] + inherited_u).tolist(),
                "displacement_m": inherited_u.tolist(),
                "rotation_rad": parent_r.tolist(),
                "dynamic_displacement_m": inherited_dynamic_u.tolist(),
                "velocity_m_s": inherited_v.tolist(),
                "angular_velocity_rad_s": parent_w.tolist(),
                "acceleration_m_s2": inherited_a.tolist(),
                "angular_acceleration_rad_s2": parent_alpha.tolist(),
                "speed_m_s": float(np.linalg.norm(inherited_v)),
                "structural_participation": False,
                "condensed_into": target,
            })
        edge_by_identity = self._edge_by_identity
        for i, identity in enumerate(self.identities):
            edge_by_identity[identity]["solver_stats"] = {
                "sequence": self.solve_sequence,
                "time_s": self.elapsed_s,
                "axial_strain": float(response["axial_strain"][i]),
                "bending_surface_strain": float(
                    response["bending_surface_strain"][i]),
                "shear_strain": float(response["shear_strain"][i]),
                "normal_stress_pa": float(response["normal_stress_pa"][i]),
                "shear_stress_pa": float(response["shear_stress_pa"][i]),
                "stress_pa": float(response["stress_pa"][i]),
                "utilisation": float(response["utilisation"][i]),
            }
        for identity, force in self.joint_forces.last_element_force.items():
            edge = edge_by_identity[identity]
            stats = edge.setdefault("solver_stats", {
                "sequence": self.solve_sequence, "time_s": self.elapsed_s})
            stats["constitutive_force_n"] = float(force)
        maximum = float(np.max(response["utilisation"])) \
            if len(response["utilisation"]) else 0.0
        summary = {
            "sequence": self.solve_sequence,
            "time_s": self.elapsed_s,
            "coupled_steps": self.last_coupled_steps,
            "minimum_coupled_dt_s": self.minimum_coupled_dt_s,
            "minimum_coupled_limiter": self.minimum_coupled_limiter,
            "maximum_utilisation": maximum,
            "maximum_speed_m_s": float(np.max(np.linalg.norm(
                velocity[:, :3], axis=1))),
            "static_residual_norm_n": float(np.linalg.norm(
                self.last_static_residual_free)),
            "energy": self.energy_state(),
            "gestalt_dormant": self.gestalt_dormant,
            "adaptive_components": tuple(sorted(self._adaptive_gestalts)),
            "dormant_components": tuple(sorted(self._dormant_partition_ids)),
        }
        self.solver.document["solver_stats"] = summary
        return {**summary, "member_utilisation": response["utilisation"]}

    def positions(self) -> np.ndarray:
        u = self.solver.endpoint_motion(self.displacement())
        return self.solver.position + u[:, :3]


# =====================================================================
def main(argv) -> None:
    import pygame
    import turret_production as tp
    import machines
    import firing_frame_view as ffv

    print("building the production graph ...", flush=True)
    graph = tp.balanced_station(bore_mm=20.0)
    doc = graph.as_document()

    # THE GAME ENGINE'S OWN MACHINE TICK, over this graph.
    sim = machines.MachineSim(machine=machines.get("gimbal-cannon-station"),
                              graph=doc)
    print("assembling the structure ...", flush=True)
    live = LiveStructure(doc)
    _n60, _h60 = live.substep_plan(1.0 / 60.0)
    print(f"  {len(live.members)} members, {len(live.omega)} modes, "
          f"{live.omega[0] / (2 * math.pi):.2f} .. "
          f"{live.omega[-1] / (2 * math.pi):.1f} Hz", flush=True)
    print(f"  implicit full-beam solve -> {_n60} substeps of "
          f"{_h60 * 1000:.3f} ms inside a 60 Hz frame",
          flush=True)

    trial = ffv.Trial(doc, width=1440, height=920, hidden=False)
    # triangle ranges per member, once, so recolouring is a scatter
    ranges = []
    from articulation import _flat
    for ident in live.identities:
        ranges.append(trial.part_range.get("edge_" + _flat(ident)))
    owner = np.concatenate([np.full(r[1] - r[0], i, np.int32)
                            for i, r in enumerate(ranges) if r]) \
        if any(ranges) else np.zeros(0, np.int32)
    tris = np.concatenate([np.arange(r[0], r[1]) for r in ranges if r]) \
        if any(ranges) else np.zeros(0, np.int64)
    bands = np.asarray([lim for lim, _c, _l in fft.UTILISATION_BANDS])
    band_mat = np.asarray([trial.band_ids[c]
                           for _l, c, _n in fft.UTILISATION_BANDS])
    base_mat = trial.rest_material_ids

    clock = pygame.time.Clock()
    azimuth, elevation, zoom = 0.62, 0.40, 1.0
    dragging = False
    running = True
    t0 = time.perf_counter()
    frames = 0
    recoil_left = 0.0
    colour_mode = "assembly"
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    recoil_left = 0.012
                    live.apply("turret.breech", (0.0, 0.0, -110_000.0))
                elif ev.key == pygame.K_g:
                    live.apply("turret.breech", (0.0, 0.0, -15_000.0))
                elif ev.key == pygame.K_r:
                    live.reset()
                elif ev.key == pygame.K_1:
                    colour_mode = "yield"
                    print("  colour: YIELD UTILISATION", flush=True)
                elif ev.key == pygame.K_2:
                    colour_mode = "assembly"
                    print("  colour: ASSEMBLY / MATERIAL", flush=True)
                elif ev.key == pygame.K_z:
                    zoom = max(0.25, zoom * 0.9)
                elif ev.key == pygame.K_x:
                    zoom = min(4.0, zoom * 1.1)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    dragging = True
                elif ev.button == 4:
                    zoom = max(0.25, zoom * 0.9)
                elif ev.button == 5:
                    zoom = min(4.0, zoom * 1.1)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                azimuth -= ev.rel[0] * 0.008
                elevation = max(-0.9, min(1.4, elevation + ev.rel[1] * 0.005))
            elif ev.type == pygame.KEYUP and ev.key == pygame.K_g:
                live.clear()

        dt = min(clock.tick(60) / 1000.0, 0.05)
        # THE GAME ENGINE TICKS FIRST: supply, actuators, damage, holes.
        sim.step(dt)
        # then the structure, at the same dt, from where it actually is
        if recoil_left > 0.0:
            recoil_left -= dt
            if recoil_left <= 0.0:
                live.clear()
        live.step(dt)

        util = live.utilisation()
        mat = ffv.material_ids_for_colour_mode(
            base_mat, tris, owner, util, bands, band_mat, colour_mode)
        trial.mesh.material_ids = mat
        trial.articulated.displace(live.positions())
        trial.view.restage_static_mesh()

        trial.view._elevation = elevation
        trial.view._zoom = zoom
        trial.present(azimuth)
        pygame.display.flip()
        frames += 1
        if frames % 60 == 0:
            hot = int((util > 0.85).sum())
            print(f"  t={time.perf_counter() - t0:6.1f}s  "
                  f"{clock.get_fps():5.1f} fps   peak "
                  f"{util.max() * 100:6.1f}% of yield   {hot} members hot",
                  flush=True)
    pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
