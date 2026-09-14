import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sled_reference as sled
import station_reference as station
import structure_native
from graph_columns import structural, structural_participation_report
from graph_physics import GraphJointForces
from frame_solver import FrameSolver
from live_scene import LiveStructure
from projectiles import ProjectileField
from turret_production import ProductionGraph


class StationReferenceMotionTests(unittest.TestCase):
    def test_plate_render_cells_end_exactly_at_authored_sheet_edges(self):
        from surfaces import Plate, emit_plate
        from vehicle_mesh import build_drivetrain_solid_parts

        graph = ProductionGraph("exact-sheet-boundary-witness")
        corner = np.asarray((1.0, 2.0, 3.0))
        span_u = np.asarray((2.0, 0.0, 1.0))
        span_v = np.asarray((-0.5, 0.0, 1.0))
        plate = Plate("sheet", tuple(corner), tuple(span_u), tuple(span_v),
                      thickness_m=0.012, nu=2, nv=2)
        emitted = emit_plate(graph, plate)
        document = graph.as_document()
        wanted = {"node_" + identity.replace(".", "_")
                  for identity in emitted["nodes"].values()}
        parts = [part for part in build_drivetrain_solid_parts(document)
                 if part.name in wanted]
        self.assertEqual(len(parts), 9)
        vertices = np.concatenate([part.vertices for part in parts])
        u = span_u / np.linalg.norm(span_u)
        v = span_v / np.linalg.norm(span_v)
        normal = np.cross(u, v)
        relative = vertices - corner
        np.testing.assert_allclose(
            (relative @ u).min(), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            (relative @ u).max(), np.linalg.norm(span_u), atol=1e-12)
        np.testing.assert_allclose(
            (relative @ v).min(), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            (relative @ v).max(), np.linalg.norm(span_v), atol=1e-12)
        np.testing.assert_allclose(
            (relative @ normal).min(), -plate.thickness_m / 2.0,
            atol=1e-12)
        np.testing.assert_allclose(
            (relative @ normal).max(), plate.thickness_m / 2.0,
            atol=1e-12)

    def test_platforms_have_structural_sheets_and_distributed_contact(self):
        from sled import (PlatformStage, emit_platform_stage,
                          emit_platform_deck_contacts)
        graph = ProductionGraph("two-sheet-contact-witness")
        anchors = ()
        names = []
        for tag, xyz in (("aft.a", (-1.0, 1.0, -1.0)),
                         ("fwd.a", (-1.0, 1.0, 1.0)),
                         ("aft.b", (1.0, 1.0, -1.0)),
                         ("fwd.b", (1.0, 1.0, 1.0))):
            name = f"anchor.{tag}"
            graph.node(name, xyz, "load-bearing-structure", fixed_to="world")
            names.append(name)
        anchors = tuple(names)
        lower = emit_platform_stage(
            graph, PlatformStage("lower", anchors, 1.0,
                                 rises=-1.0, rest_angle_deg=0.0))
        upper = emit_platform_stage(
            graph, PlatformStage("upper", lower["anchors_for_next"], 1.0,
                                 rises=1.0, rest_angle_deg=60.0,
                                 inboard_m=0.1))
        contacts = emit_platform_deck_contacts(graph, lower, upper)

        self.assertEqual(len(lower["deck_sheet"]["nodes"]), 9)
        self.assertEqual(len(upper["deck_sheet"]["nodes"]), 9)
        lower_sheet_nodes = [next(node for node in graph.nodes
                                  if node["identity"] == identity)
                             for identity in
                             lower["deck_sheet"]["nodes"].values()]
        lower_corner_y = next(node["reference_position"][1]
                              for node in graph.nodes
                              if node["identity"] ==
                              lower["corners"]["aft.a"])
        self.assertTrue(all(node["mounting_face"] == "top-of-support-pipe"
                            and node["overhang_m"] == 0.0
                            for node in lower_sheet_nodes))
        self.assertAlmostEqual(
            min(node["reference_position"][1]
                - node["thickness_m"] / 2.0
                for node in lower_sheet_nodes),
            lower_corner_y + lower["stage"].beam_radius_m)
        self.assertGreaterEqual(len(contacts), 4)
        made = {edge["identity"]: edge for edge in graph.edges}
        self.assertTrue(all(made[name]["part_role"]
                            == "platform-sheet-contact" for name in contacts))
        self.assertTrue(all(made[name]["slide_axis"] == (0.0, 1.0, 0.0)
                            for name in contacts))
        document = graph.as_document()
        distributor = GraphJointForces(document)
        xyz = np.asarray([node["reference_position"]
                          for node in document["nodes"]], float)
        velocity = np.zeros_like(xyz)
        witness = made[contacts[0]]
        ia = distributor.index[witness["a"]]
        ib = distributor.index[witness["b"]]
        xyz[ib, 1] = xyz[ia, 1] + 0.006
        force = distributor.evaluate(1.0e-4, xyz, velocity)
        self.assertGreater(distributor.last_element_force[contacts[0]], 0.0)
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-8)

    def test_live_colour_mode_switches_between_authored_and_yield_palette(self):
        from firing_frame_view import material_ids_for_colour_mode
        base = np.asarray((10, 11, 12, 13, 14), dtype=np.int32)
        triangles = np.asarray((1, 3, 4), dtype=np.int64)
        owner = np.asarray((0, 1, 0), dtype=np.int32)
        utilisation = np.asarray((.05, .90))
        bands = np.asarray((.10, .85, 1.0))
        palette = np.asarray((20, 21, 22), dtype=np.int32)

        assembly = material_ids_for_colour_mode(
            base, triangles, owner, utilisation, bands, palette, "assembly")
        yielded = material_ids_for_colour_mode(
            base, triangles, owner, utilisation, bands, palette, "yield")

        np.testing.assert_array_equal(assembly, base)
        np.testing.assert_array_equal(yielded, (10, 20, 12, 22, 20))
        with self.assertRaises(ValueError):
            material_ids_for_colour_mode(
                base, triangles, owner, utilisation, bands, palette, "dead")

    def test_explicit_nonstructural_node_flag_controls_solver_pass(self):
        graph = ProductionGraph("beam-flag-witness")
        graph.node("a", (0.0, 0.0, 0.0), "load-bearing-structure")
        graph.node("b", (1.0, 0.0, 0.0), "machine-subpart",
                   structural_participation=False)
        graph.edge("visual_mount", "a", "b", "rigid-distance",
                   radius=.02, beam_solvable=False)
        document = graph.as_document()

        # The legacy edge column still contains exact rigid kinematic ties;
        # the node participation pass is the unambiguous exclusion boundary.
        self.assertEqual(len(structural(document)), 1)
        solver = FrameSolver(document, {})
        self.assertEqual(len(solver.members), 0)
        self.assertFalse(solver.structural_node[solver.index["a"]])
        self.assertFalse(solver.structural_node[solver.index["b"]])

    def test_explicit_nonstructural_edge_is_not_a_beam(self):
        graph = ProductionGraph("edge-flag-witness")
        graph.node("a", (0.0, 0.0, 0.0), "load-bearing-structure")
        graph.node("b", (1.0, 0.0, 0.0), "load-bearing-structure")
        graph.edge("drawing_tie", "a", "b", "rigid-distance",
                   radius=.02, structural_participation=False)
        document = graph.as_document()

        self.assertEqual(len(structural(document)), 0)
        self.assertEqual(len(FrameSolver(document, {}).members), 0)

    def test_participation_audit_separates_rigid_condensation_from_exclusion(self):
        graph = ProductionGraph("participation-outcome-witness")
        graph.node("body", (0.0, 0.0, 0.0), "load-bearing-structure")
        graph.node("body.skin", (0.0, 0.2, 0.0), "machine-subpart",
                   solver_condensed_into="body", solver_condensed_mass=True)
        graph.node("body.port", (0.0, 0.4, 0.0), "structural-mount-port",
                   solver_condensed_into="body")
        graph.node("drawing", (1.0, 0.0, 0.0), "machine-subpart",
                   structural_participation=False)
        graph.edge("condensed_structure", "body.skin", "body.port",
                   "rigid-distance", radius=.02)
        graph.edge("drawing_edge", "body", "drawing", "rigid-distance",
                   radius=.02)
        report = structural_participation_report(graph.as_document())

        self.assertEqual(report["rigid_condensed_nodes"], 2)
        self.assertEqual(report["rigid_condensed_edges"], 1)
        self.assertEqual(report["excluded_by_nonstructural_node"], 1)
        self.assertEqual(report["deformable_beam_edges"], 0)

    def test_disengaged_lock_is_not_structurally_assembled(self):
        graph = ProductionGraph("lock-state-witness")
        graph.node("a", (0.0, 0.0, 0.0), "load-bearing-structure")
        graph.node("b", (1.0, 0.0, 0.0), "load-bearing-structure")
        graph.edge("positive_lock", "a", "b", "rigid-distance",
                   radius=.02, lock_engaged=False)
        document = graph.as_document()

        self.assertEqual(len(structural(document)), 0)
        self.assertEqual(len(FrameSolver(document, {}).members), 0)

    def test_live_authored_settle_uses_frame_solvers_gravity_vector(self):
        graph = ProductionGraph("runtime-gravity-witness")
        graph.node("fixed", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=100.0)
        graph.edge("beam", "fixed", "body", "rigid-distance",
                   radius=.05)
        document = graph.as_document()
        solver = FrameSolver(document, {})
        live = LiveStructure(document, solver=solver,
                             settle_from_authored=True)

        np.testing.assert_allclose(live.u_static, 0.0)
        self.assertLess(live.base_force[solver.index["body"] * 6 + 1], 0.0)
        live.step(1.0 / 240.0)
        self.assertLess(live.displacement()[solver.index["body"] * 6 + 1], 0.0)

    def test_live_utilisation_uses_shear_and_ignores_ideal_constraints(self):
        graph = ProductionGraph("live-capacity-witness")
        graph.node("fixed", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure")
        graph.edge("beam", "fixed", "body", "rigid-distance", radius=.02)
        graph.edge("ideal_contact", "fixed", "body", "rigid-distance",
                   radius=.02, rigid=True)
        document = graph.as_document()
        live = LiveStructure(document)
        live.dynamic_displacement[live.solver.index["body"] * 6 + 1] = .001
        response = live.member_response()
        beam = live.identities.index("beam")
        ideal = live.identities.index("ideal_contact")

        self.assertGreater(response["shear_stress_pa"][beam], 0.0)
        self.assertGreater(response["utilisation"][beam], 0.0)
        self.assertEqual(response["utilisation"][ideal], 0.0)

    def test_external_actuator_force_enters_through_edge_endpoints(self):
        graph = ProductionGraph("actuator-force-witness")
        graph.node("fixed", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=10.0)
        graph.edge("beam", "fixed", "body", "rigid-distance", radius=.04)
        graph.edge("ram", "fixed", "body", "linear-hydraulic-actuator",
                   radius=.02)
        document = graph.as_document()
        live = LiveStructure(document)
        live.set_edge_axial_forces({"ram": 1000.0})

        self.assertGreater(live.actuator_force[
            live.solver.index["body"] * 6], 0.0)
        self.assertLess(live.actuator_force[
            live.solver.index["fixed"] * 6], 0.0)

    def test_vectorized_actuator_force_preserves_condensed_port_moment(self):
        graph = ProductionGraph("actuator-condensed-port-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=20.0)
        graph.node("body.port", (1.0, 0.5, 0.0),
                   "structural-mount-port", wrench_point=True,
                   surface_of="body", solver_condensed_into="body",
                   solver_condensed_mass=False)
        graph.edge("mount", "fixed", "body", "rigid-distance", radius=.03)
        graph.edge("ram", "fixed", "body.port", "linear-hydraulic-actuator",
                   radius=.02, structural_participation=False)
        document = graph.as_document()
        live = LiveStructure(document)

        live.set_edge_axial_forces({"ram": 1234.0})
        expected = np.zeros_like(live.actuator_force)
        position = live.positions()
        ia = live.solver.index["fixed"]
        ib = live.solver.index["body.port"]
        axis = position[ib] - position[ia]
        axis /= np.linalg.norm(axis)
        wrench = np.r_[axis * 1234.0, np.zeros(3)]
        live.solver._add_endpoint_wrench(expected, ia, -wrench)
        live.solver._add_endpoint_wrench(expected, ib, wrench)

        np.testing.assert_allclose(live.actuator_force, expected,
                                   rtol=1.0e-13, atol=1.0e-12)
        body_moment = live.actuator_force[
            live.solver.index["body"] * 6 + 3:
            live.solver.index["body"] * 6 + 6]
        self.assertGreater(np.linalg.norm(body_moment), 0.0)

    def test_component_owned_assembly_matches_whole_reference_matrix(self):
        graph = ProductionGraph("component-assembly-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world",
                   mass_kg=2.0)
        graph.node("middle", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=3.0)
        graph.node("tip", (2.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=4.0)
        graph.edge("left", "fixed", "middle", "rigid-distance", radius=.03)
        graph.edge("right", "middle", "tip", "rigid-distance", radius=.04)
        document = graph.as_document()
        solver = FrameSolver(document, gravity=False)
        reference = solver.modes()
        component = solver.assemble_component_matrices(
            {"left", "right"}, node_mass_fractions={
                "fixed": 1.0, "middle": 1.0, "tip": 1.0})

        dofs = component["physical_dofs"]
        np.testing.assert_allclose(
            component["stiffness_matrix"],
            reference["stiffness_matrix"][np.ix_(dofs, dofs)])
        np.testing.assert_allclose(
            np.diag(component["mass_matrix"]),
            reference["lumped_mass"][dofs])

    def test_component_partitions_are_generic_topology_not_part_names(self):
        from component_mode_atlas import (AdaptiveGestaltPolicy,
                                          assemble_component_partition,
                                          build_component_stress_envelope,
                                          build_component_orchestration_graph,
                                          build_global_gestalt_subspace,
                                          discover_component_partitions,
                                          merge_welded_partition_cluster)

        graph = ProductionGraph("generic-component-partition-witness")
        graph.assembly = "alpha"
        graph.node("fixed", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world", mass_kg=2.0)
        graph.node("shared", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=3.0)
        graph.edge("alpha.member", "fixed", "shared", "rigid-distance",
                   radius=.03)
        graph.assembly = "beta"
        graph.node("tip", (2.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=4.0)
        graph.edge("beta.member", "shared", "tip", "rigid-distance",
                   radius=.03)
        graph.assembly = "arbitrary-free-machine"
        graph.node("free.a", (0.0, 2.0, 0.0), "load-bearing-structure",
                   mass_kg=2.0)
        graph.node("free.b", (1.0, 2.0, 0.0), "load-bearing-structure",
                   mass_kg=2.0)
        graph.edge("free.member", "free.a", "free.b", "rigid-distance",
                   radius=.03)
        graph.assembly = "controller"
        graph.edge("crossing.ram", "tip", "free.a",
                   "linear-hydraulic-actuator", radius=.02,
                   structural_participation=False)
        solver = FrameSolver(graph.as_document(), gravity=False)
        partitions = {part.identity: part for part in
                      discover_component_partitions(solver)}

        self.assertEqual(set(partitions),
                         {"alpha", "beta", "arbitrary-free-machine"})
        self.assertEqual(
            [solver.document["nodes"][i]["identity"]
             for i in partitions["beta"].interface_node_indices],
            ["shared", "tip"])
        self.assertEqual(set(partitions["beta"].node_mass_fractions), {"tip"})
        self.assertFalse(
            partitions["arbitrary-free-machine"].floating_reference_added)
        self.assertEqual(
            [solver.document["nodes"][i]["identity"] for i in
             partitions["arbitrary-free-machine"].interface_node_indices],
            ["free.a"])
        assembled = assemble_component_partition(solver, partitions["alpha"])
        self.assertEqual(len(assembled["interface_local_dofs"]), 12)
        envelope = build_component_stress_envelope(
            solver, partitions["arbitrary-free-machine"])
        self.assertEqual(envelope.probe_identities, ("free.member",))
        self.assertEqual(envelope.influence_pa_per_wrench.shape, (1, 6))
        self.assertTrue(np.all(envelope.influence_pa_per_wrench >= 0.0))
        self.assertGreater(envelope.bound(np.array([1., 0., 0., 0., 0., 0.])),
                           0.0)
        reference = solver.modes()
        subspace = build_global_gestalt_subspace(
            solver, reference["stiffness_matrix"], reference["lumped_mass"],
            reference["free"], [partitions["beta"]])
        self.assertEqual(subspace.reduced_size, len(reference["free"]) - 6)
        reduced = np.linspace(-.2, .3, subspace.reduced_size)
        np.testing.assert_allclose(subspace.project(subspace.recover(reduced)),
                                   reduced, atol=1e-11)
        arbitrary = np.linspace(-.4, .5, len(reference["free"]))
        projected = subspace.recover(subspace.project(arbitrary))
        np.testing.assert_allclose(
            subspace.transformation.T @ subspace._mass_matrix
            @ (arbitrary - projected), 0.0, atol=1e-11)
        diagonal = subspace.mass_orthonormal_basis(1.0e-6)
        basis = diagonal["basis_free"]
        np.testing.assert_allclose(
            basis.T @ subspace._mass_matrix @ basis,
            np.eye(subspace.reduced_size), atol=1e-10)
        reduced_k = basis.T @ reference["stiffness_matrix"][
            np.ix_(reference["free"], reference["free"])] @ basis
        np.testing.assert_allclose(
            reduced_k, np.diag(diagonal["omega"] ** 2),
            rtol=1e-8, atol=2e-6)
        live = LiveStructure(solver.document, solver=solver,
                             reference=reference)
        live.configure_gestalt_subspace(subspace)
        physical = subspace.recover(reduced)
        physical_velocity = subspace.recover(reduced * .25)
        live.reference_q = (live.reference_basis_free.T
                            @ (live.mass_free * physical))
        live.reference_qdot = (live.reference_basis_free.T
                               @ (live.mass_free * physical_velocity))
        live.set_gestalt_dormant(True)
        self.assertTrue(live.gestalt_dormant)
        self.assertEqual(len(live.reference_q), subspace.reduced_size)
        np.testing.assert_allclose(
            live.reference_basis_free @ live.reference_q, physical,
            atol=1e-10)
        np.testing.assert_allclose(
            live.reference_basis_free @ live.reference_qdot,
            physical_velocity, atol=1e-10)
        live.set_gestalt_dormant(False)
        self.assertFalse(live.gestalt_dormant)
        np.testing.assert_allclose(
            live.reference_basis_free @ live.reference_q, physical,
            atol=1e-10)
        automatic = LiveStructure(solver.document, solver=solver,
                                  reference=reference)
        automatic.configure_adaptive_partition(
            partitions["arbitrary-free-machine"],
            AdaptiveGestaltPolicy(
                sleep_stress_bound_pa=1.0, wake_stress_bound_pa=2.0,
                sleep_internal_kinetic_bound_j=1.0, sleep_dwell_s=.01),
            envelope)
        self.assertEqual(automatic.update_gestalt_residency(.01), "slept")
        self.assertTrue(automatic.gestalt_dormant)
        automatic.apply("free.a", (1000.0, 0.0, 0.0))
        self.assertEqual(automatic.update_gestalt_residency(.01), "woke")
        self.assertFalse(automatic.gestalt_dormant)
        orchestration = build_component_orchestration_graph(solver)
        self.assertEqual(orchestration.weld_neighbors("alpha"), ("beta",))
        self.assertEqual(
            orchestration.coupling_neighbors("beta"),
            ("arbitrary-free-machine",))
        self.assertEqual(
            orchestration.welded_clusters(
                {"alpha", "beta", "arbitrary-free-machine"}),
            (("alpha", "beta"), ("arbitrary-free-machine",)))
        merged = merge_welded_partition_cluster(
            solver, orchestration, {"alpha", "beta"})
        self.assertEqual(set(merged.member_identities),
                         {"alpha.member", "beta.member"})
        self.assertEqual(
            [solver.document["nodes"][i]["identity"]
             for i in merged.interface_node_indices],
            ["fixed", "tip"])
        beta_envelope = build_component_stress_envelope(
            solver, partitions["beta"])
        automatic_many = LiveStructure(
            solver.document, solver=solver, reference=reference)
        quiet_policy = AdaptiveGestaltPolicy(
            sleep_stress_bound_pa=1.0, wake_stress_bound_pa=2.0,
            sleep_internal_kinetic_bound_j=1.0, sleep_dwell_s=.01)
        automatic_many.configure_adaptive_orchestration(
            orchestration, {
                "beta": (quiet_policy, beta_envelope),
                "arbitrary-free-machine": (quiet_policy, envelope),
            })
        transitions = automatic_many.update_gestalt_residency(.01)
        self.assertEqual(transitions,
                         {"beta": "slept",
                          "arbitrary-free-machine": "slept"})
        self.assertEqual(automatic_many._dormant_partition_ids,
                         {"beta", "arbitrary-free-machine"})
        automatic_many.apply("tip", (1000.0, 0.0, 0.0))
        transitions = automatic_many.update_gestalt_residency(.01)
        self.assertEqual(transitions["beta"], "woke")
        self.assertEqual(automatic_many._dormant_partition_ids,
                         {"arbitrary-free-machine"})

    def test_arch_load_paths_use_their_real_beam_sections(self):
        graph, *_ = sled.build()
        edges = {edge["identity"]: edge for edge in graph.as_document()["edges"]}
        physical = [identity for identity in edges
                    if identity.startswith(("arch.leg.", "arch.crown.",
                                            "arch.pin_boss.", "arch.tie."))]

        self.assertTrue(physical)
        self.assertTrue(all(edges[identity].get("beam_solvable")
                            for identity in physical))
        self.assertTrue(all(not edges[identity].get("rigid", False)
                            for identity in physical))
        self.assertTrue(all("damage" in edges[identity]
                            for identity in physical))

    def test_arch_is_low_deep_and_has_locked_hydraulic_crown_rise(self):
        graph, *_ = sled.build()
        document = graph.as_document()
        nodes = {node["identity"]: node for node in document["nodes"]}
        lifts = [edge for edge in document["edges"]
                 if edge.get("part_role") == "arch-crown-lift-actuator"]
        locks = [edge for edge in document["edges"]
                 if edge.get("part_role") ==
                 "arch-crown-positive-height-lock"]
        crown = [edge for edge in document["edges"]
                 if edge.get("load_path") ==
                 "deep-peaked-arch-crown-segment"]

        self.assertEqual(len(lifts), 4)
        self.assertEqual(len(locks), 4)
        self.assertEqual(len(crown), 4)
        self.assertTrue(all(edge["stroke_m"] == 1.2 for edge in lifts))
        self.assertTrue(all(edge.get("lock_engaged") for edge in locks))
        for side in ("a", "b"):
            apex = nodes[f"arch.crown.{side}.apex"]
            self.assertAlmostEqual(apex["reference_position"][1],
                                   sled.DECK_Y + 1.8)
            self.assertEqual(apex["maximum_vertical_travel_m"], 1.2)

        high = sled.arch_height_document(document, 1.2)
        high_nodes = {node["identity"]: node for node in high["nodes"]}
        high_edges = {edge["identity"]: edge for edge in high["edges"]}
        for side in ("a", "b"):
            self.assertAlmostEqual(
                high_nodes[f"arch.crown.{side}.apex"]["reference_position"][1],
                nodes[f"arch.crown.{side}.apex"]["reference_position"][1]
                + 1.2)
        self.assertAlmostEqual(
            high_nodes["turret.breech"]["reference_position"][1],
            nodes["turret.breech"]["reference_position"][1] + 1.2)
        self.assertEqual(high["arch_crown_state"]["fraction"], 1.0)
        self.assertTrue(all(high_edges[edge["identity"]]["rest_length"] ==
                            edge["rest_length"] + 1.2 for edge in lifts))

    def test_unlocked_audit_releases_crown_and_uses_narrower_second_level_pins(self):
        graph, *_ = sled.build()
        document = graph.as_document()
        nodes = {node["identity"]: node for node in document["nodes"]}
        level_one = [node for node in document["nodes"]
                     if node.get("part_role") == "arch-pin"]
        level_two = [node for node in document["nodes"]
                     if node.get("part_role") == "platform-pin"
                     and node.get("stage") == "gun"]
        self.assertEqual(len(level_one), 4)
        self.assertEqual(len(level_two), 4)
        self.assertTrue(all(node["pin_level"] == 1 and node["shape"] == "drum"
                            for node in level_one))
        self.assertTrue(all(node["pin_level"] == 2 and node["shape"] == "drum"
                            for node in level_two))
        self.assertLess(max(node["drum_radius_m"] for node in level_two),
                        min(node["drum_radius_m"] for node in level_one))
        self.assertLess(max(node["drum_length_m"] for node in level_two),
                        min(node["drum_length_m"] for node in level_one))
        self.assertTrue(all(abs(node["reference_position"][0]) <
                            abs(nodes[f"dangling.corner.{node['corner']}"]
                                ["reference_position"][0])
                            for node in level_two))

        unlocked = station.unlocked_motion_document(document)
        locks = [edge for edge in unlocked["edges"]
                 if edge.get("part_role") ==
                 "arch-crown-positive-height-lock"]
        self.assertEqual(len(locks), 4)
        self.assertTrue(all(not edge["lock_engaged"]
                            and edge["structural_participation"] is False
                            for edge in locks))
        self.assertTrue(all(edge.get("preload_n") == 0.0
                            for edge in unlocked["edges"]
                            if edge.get("part_role") == "platform-actuator"))

    def test_planted_pose_has_retracted_inner_and_extended_angled_jacks(self):
        import stand
        graph = ProductionGraph("stand-state-witness")
        spec = stand.Stand()
        stand.emit_stand(graph, spec)
        document = graph.as_document()
        legs = {edge["identity"]: edge for edge in document["edges"]
                if edge.get("part_role") == "outrigger-leg"}

        self.assertEqual(len(legs), 8)
        for identity, edge in legs.items():
            if ".outer." in identity:
                self.assertEqual(edge["actuator_state"],
                                 "rake-stage-fully-extended-service-stage-retracted")
                self.assertEqual(edge["rake_stage_extension_frac"], 1.0)
                self.assertEqual(edge["service_stage_extension_frac"], 0.0)
                self.assertAlmostEqual(edge["rest_length"], spec.leg_length_m)
            else:
                self.assertEqual(edge["actuator_state"], "fully-retracted")
                self.assertEqual(edge["service_stage_extension_frac"], 0.0)
                self.assertAlmostEqual(edge["rest_length"],
                                       spec.lower_room_clear_height_m)
            expected_stroke = (spec.corner_leg_stroke_m if ".outer." in identity
                               else spec.trailer_clearance_m)
            self.assertAlmostEqual(edge["stroke_m"], expected_stroke, places=4)

        runtime = stand.outrigger_set(spec, carried_kg=20_000.0)
        self.assertEqual(len(runtime.legs), 8)
        for identity, leg in runtime.legs.items():
            expected = spec.rake_extension_m if identity.startswith("outer.") else 0.0
            self.assertAlmostEqual(leg.extension_m, expected)
            self.assertAlmostEqual(
                leg.physical_length_m,
                spec.leg_length_m if identity.startswith("outer.")
                else spec.lower_room_clear_height_m)
            if identity.startswith("outer."):
                self.assertAlmostEqual(leg.cylinder.bore_m, 0.160)
                self.assertAlmostEqual(leg.cylinder.rod_m, 0.100)
                self.assertEqual(leg.cylinder.stages, 3)
            else:
                self.assertAlmostEqual(leg.cylinder.bore_m, 0.140)
                self.assertAlmostEqual(leg.cylinder.rod_m, 0.090)
                self.assertEqual(leg.cylinder.stages, 2)

    def test_support_oil_volume_bounds_follow_pumped_actuator_position(self):
        import stand
        graph = ProductionGraph("support-oil-volume-witness")
        spec = stand.Stand()
        stand.emit_stand(graph, spec)
        document = station.support_leveling_document(graph.as_document())
        joints = GraphJointForces(document)

        boundaries = [edge for edge in joints.bump_stop_edges
                      if edge.get("part_role") ==
                      "outrigger-hydraulic-volume-boundary"]
        self.assertEqual(len(boundaries), 16)
        self.assertEqual(
            {edge["contact_side"] for edge in boundaries},
            {"minimum-separation", "maximum-separation"})

        class StructureWitness:
            joint_forces = joints
            applied = None

            def set_edge_axial_forces(self, values):
                self.applied = dict(values)

        runtime = stand.outrigger_set(spec, carried_kg=20_000.0)
        reading = {"legs": {
            name: {"extension_m": leg.extension_m + 0.025,
                   "force_n": 1234.0}
            for name, leg in runtime.legs.items()}}
        witness = StructureWitness()
        station.apply_support_hydraulic_forces(witness, reading)
        self.assertEqual(len(witness.applied), 8)
        self.assertTrue(all(value == 1234.0
                            for value in witness.applied.values()))
        self.assertTrue(all(abs(edge["clearance_m"] - 0.025) < 1e-12
                            for edge in boundaries))

        position = np.asarray([node["reference_position"]
                               for node in document["nodes"]], float)
        velocity = np.zeros_like(position)
        leg = joints.edge_by_id["stand.leg.inner.fr"]
        ia, ib = joints.index[leg["a"]], joints.index[leg["b"]]
        axis = np.asarray(leg["slide_axis"], float)
        # Exactly the pumped coordinate is unloaded. Moving ahead of its oil
        # volume engages the extension face; moving behind it engages the
        # retract face. Both reactions are balanced on the same real nodes.
        at_volume = position.copy()
        at_volume[ib] += axis * 0.025
        force = joints.evaluate(1e-4, at_volume, velocity)
        pair = [edge for edge in boundaries
                if edge["support_name"] == "inner.fr"]
        self.assertTrue(all(joints.last_element_force[e["identity"]] == 0.0
                            for e in pair))
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1e-7)

        ahead = at_volume.copy()
        ahead[ib] += axis * 0.010
        joints.evaluate(1e-4, ahead, velocity)
        extension_face = next(e for e in pair
                              if e["contact_side"] == "maximum-separation")
        self.assertLess(joints.last_element_force[extension_face["identity"]],
                        0.0)

        behind = at_volume.copy()
        behind[ib] -= axis * 0.010
        joints.evaluate(1e-4, behind, velocity)
        retract_face = next(e for e in pair
                            if e["contact_side"] == "minimum-separation")
        self.assertGreater(joints.last_element_force[retract_face["identity"]],
                           0.0)

    def test_pumped_oil_coordinate_physically_holds_a_live_support(self):
        graph = ProductionGraph("live-support-oil-column-witness")
        graph.node("fixed", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=1000.0, half_extent_m=(0.2, 0.2, 0.2))
        graph.edge("guide", "fixed", "body", "single-axis-slider",
                   radius=0.04, slide_axis=(1.0, 0.0, 0.0))
        graph.edge("stand.leg.inner.fr", "fixed", "body",
                   "linear-hydraulic-actuator", radius=0.04,
                   part_role="outrigger-leg", support_name="inner.fr",
                   actuator_extension_m=0.0,
                   initial_actuator_extension_m=0.0,
                   slide_axis=(1.0, 0.0, 0.0))
        common = dict(
            kind="bump-stop", in_view=False,
            structural_participation=False, beam_solvable=False,
            part_role="outrigger-hydraulic-volume-boundary",
            support_name="inner.fr",
            coupled_slide_edges=("stand.leg.inner.fr",),
            slide_load_share=(1.0,),
            coupled_reference_separation_m=(1.0,),
            coupled_motion_sign=(1.0,), slide_axis=(1.0, 0.0, 0.0),
            clearance_m=0.0, linear_stiffness_n_per_m=2.5e8,
            cubic_stiffness_n_per_m3=2.0e12,
            compression_damping_n_s_per_m=7.5e5,
            hydraulic_position_offset_m=0.0)
        graph.edge("oil.retract", "fixed", "body", "bump-stop-contact",
                   contact_side="minimum-separation", **common)
        graph.edge("oil.extend", "fixed", "body", "bump-stop-contact",
                   contact_side="maximum-separation", **common)
        document = graph.as_document()
        solver = FrameSolver(document, loads={}, gravity=False)
        live = LiveStructure(document, solver=solver,
                             reference=solver.modes(count=1))
        target = 0.001
        station.apply_support_hydraulic_forces(live, {"legs": {
            "inner.fr": {"extension_m": target, "force_n": 100_000.0}}})
        for _ in range(20):
            live.step(1.0 / 240.0)
        travel = live.positions()[solver.index["body"], 0] - 1.0
        self.assertTrue(np.isfinite(travel))
        self.assertLess(abs(travel - target), 0.003)
        self.assertLess(abs(travel), 0.01)
        self.assertLess(
            live.joint_forces.last_element_force["oil.extend"], 0.0)

    def test_raised_firing_pose_keeps_all_eight_pads_grounded(self):
        import stand
        graph = ProductionGraph("raised-stand-witness")
        spec = stand.Stand()
        stand.emit_stand(graph, spec)
        raised = stand.raised_firing_document(graph.as_document(), spec)
        nodes = {node["identity"]: node for node in raised["nodes"]}

        ground_y = -spec.lower_room_clear_height_m
        pads = [node for node in raised["nodes"]
                if node.get("part_role") == "outrigger-pad"]
        self.assertEqual(len(pads), 8)
        np.testing.assert_allclose(
            [node["reference_position"][1] for node in pads], ground_y,
            atol=1e-12)
        self.assertEqual(raised["stand_configuration"][
            "deck_height_above_ground_m"], 4.2)
        for tag in ("outer.a", "outer.b", "outer.c", "outer.d"):
            top = np.asarray(nodes[f"stand.leg.{tag}.top"]["reference_position"])
            pad = np.asarray(nodes[f"stand.leg.{tag}.pad"]["reference_position"])
            lug = np.asarray(nodes[f"stand.leg.{tag}.lug"]["reference_position"])
            self.assertAlmostEqual(np.linalg.norm(pad - top),
                                   spec.raised_raked_leg_length_m)
            self.assertAlmostEqual(np.linalg.norm(lug - top), 1.6)

    def test_corner_deploy_lug_is_part_of_upper_leg_not_rigid_triangle(self):
        import stand
        graph = ProductionGraph("deploy-lug-boundary-witness")
        spec = stand.Stand()
        stand.emit_stand(graph, spec)
        document = graph.as_document()
        nodes = {node["identity"]: node for node in document["nodes"]}
        edges = {edge["identity"]: edge for edge in document["edges"]}

        for tag in ("outer.a", "outer.b", "outer.c", "outer.d"):
            lug = nodes[f"stand.leg.{tag}.lug"]
            self.assertEqual(lug["solver_condensed_into"],
                             f"stand.leg.{tag}.top")
            self.assertTrue(lug["solver_condensed_mass"])
            self.assertFalse(edges[f"stand.deploy_lug.{tag}"][
                "structural_participation"])
            self.assertFalse(edges[f"stand.deploy_lug_foot.{tag}"][
                "structural_participation"])
            self.assertEqual(edges[f"stand.deploy_ram.{tag}"]["b"],
                             f"stand.leg.{tag}.lug")

    def test_lower_room_and_engine_lowering_frames_hang_from_deck(self):
        import stand
        graph = ProductionGraph("lower-room-framing-witness")
        spec = stand.Stand()
        site = stand.emit_stand(graph, spec)
        document = graph.as_document()
        nodes = {node["identity"]: node for node in document["nodes"]}
        edges = {edge["identity"]: edge for edge in document["edges"]}

        self.assertEqual(len(site["lower_room"]), 4)
        self.assertEqual({edge.get("part_role") for edge in edges.values()
                          if edge["identity"].startswith(
                              "stand.lower_room.cross.")},
                         {"lower-room-end-crossmember"})
        hangers = [edge for edge in edges.values()
                   if edge.get("part_role") in {
                       "lower-room-frame-hanger",
                       "engine-lowering-frame-hanger"}]
        self.assertEqual(len(hangers), 12)
        self.assertTrue(all(edge.get("always_down") for edge in hangers))
        self.assertTrue(all(edge.get("not_ground_support") for edge in hangers))
        self.assertTrue(all(not nodes[edge["b"]].get("fixed_to")
                            for edge in hangers))
        floor_nodes = [nodes[identity] for identity in
                       site["floor_plate"]["nodes"].values()]
        floor_x = [node["reference_position"][0] for node in floor_nodes]
        floor_z = [node["reference_position"][2] for node in floor_nodes]
        self.assertAlmostEqual(max(floor_x), spec.half - 0.20)
        self.assertAlmostEqual(min(floor_x), -spec.half + 0.20)
        self.assertAlmostEqual(max(floor_z), spec.half - 0.20)
        self.assertAlmostEqual(min(floor_z), -spec.half + 0.20)
        pads = [node for node in document["nodes"]
                if node.get("part_role") == "outrigger-pad"]
        self.assertTrue(all(not (min(floor_x) <= node["reference_position"][0]
                                 <= max(floor_x)
                                 and min(floor_z)
                                 <= node["reference_position"][2]
                                 <= max(floor_z)) for node in pads))
        welds = [edge for edge in document["edges"]
                 if edge.get("part_role") ==
                 "tank-service-floor-perimeter-weld"]
        self.assertEqual(len(welds), 4)
        self.assertTrue(all(edge.get("fastening") ==
                            "continuous-fillet-weld" for edge in welds))
        plate_strips = [edge for edge in document["edges"]
                        if edge.get("plate") ==
                        "stand.lower_room.floor"]
        self.assertTrue(plate_strips)
        self.assertTrue(all(edge["damage"]["second_moment_y_m4"]
                            != edge["damage"]["second_moment_z_m4"]
                            for edge in plate_strips))
        spines = [edge for edge in document["edges"]
                  if edge.get("part_role") ==
                  "grand-floor-longitudinal"]
        crossmembers = [edge for edge in document["edges"]
                        if edge.get("part_role") ==
                        "stout-floor-crossmember"]
        self.assertEqual(len(spines), 8)
        self.assertEqual(len(crossmembers), 20)
        self.assertTrue(all(edge.get("section_shape") == "welded-i"
                            and edge["damage"]["material"] == "hy80"
                            for edge in spines + crossmembers))
        for side in ("port", "starboard"):
            self.assertEqual(len(site["lower_bays"][side]), 4)
            self.assertEqual(len([edge for edge in edges.values()
                                  if edge.get("bay") == side and edge.get(
                                      "part_role") ==
                                  "engine-lowering-end-crossmember"]), 2)

    def test_oriented_i_section_uses_strong_and_weak_axes(self):
        from milspec import WeldedISection
        section = WeldedISection(.40, .22, .014, .022)

        def tip_deflection(section_up):
            graph = ProductionGraph("oriented-section-witness")
            graph.node("root", (0.0, 0.0, 0.0),
                       "load-bearing-structure", fixed_to="world")
            graph.node("tip", (2.0, 0.0, 0.0),
                       "load-bearing-structure")
            graph.edge(
                "beam", "root", "tip", "rigid-distance",
                radius=section.flange_width_m / 2.0,
                alloy=section.material, section_shape="welded-i",
                section_up=section_up,
                section_depth_m=section.depth_m,
                section_flange_width_m=section.flange_width_m,
                section_web_thickness_m=section.web_thickness_m,
                section_flange_thickness_m=section.flange_thickness_m,
                section_properties=section.graph_properties())
            solver = FrameSolver(graph.as_document(), gravity=False,
                                 loads={"tip": (0.0, -10_000.0, 0.0)})
            return abs(solver.solve()["displacement"][solver.index["tip"], 1])

        strong = tip_deflection((0.0, 1.0, 0.0))
        weak = tip_deflection((0.0, 0.0, 1.0))
        self.assertGreater(weak, strong * 4.0)

    def test_welded_i_section_is_rendered_as_web_and_flange_plates(self):
        from milspec import WeldedISection
        from vehicle_mesh import build_drivetrain_mesh
        section = WeldedISection(.40, .22, .014, .022)
        graph = ProductionGraph("i-section-mesh-witness")
        graph.node("a", (0.0, 0.0, 0.0), "load-bearing-structure")
        graph.node("b", (2.0, 0.0, 0.0), "load-bearing-structure")
        graph.edge(
            "beam", "a", "b", "rigid-distance",
            section_shape="welded-i", section_up=(0.0, 0.0, 1.0),
            section_depth_m=section.depth_m,
            section_flange_width_m=section.flange_width_m,
            section_web_thickness_m=section.web_thickness_m,
            section_flange_thickness_m=section.flange_thickness_m,
            section_properties=section.graph_properties())

        vertices, _normals, name = build_drivetrain_mesh(
            graph.as_document())[0]
        self.assertEqual(name, "edge_beam")
        self.assertEqual(len(vertices), 3 * 36)
        self.assertAlmostEqual(np.ptp(vertices[:, 1]), .22)
        self.assertAlmostEqual(np.ptp(vertices[:, 2]), .40)

    def test_drum_has_four_columns_into_floor_spines(self):
        graph, _plat, _box, _spec, _stand, site = station.build()
        document = graph.as_document()
        nodes = {node["identity"]: node for node in document["nodes"]}
        edges = {edge["identity"]: edge for edge in document["edges"]}

        self.assertEqual(set(site["drum_columns"]), {"fl", "fr", "rl", "rr"})
        for tag, path in site["drum_columns"].items():
            column = edges[f"stand.drum_column.{tag}"]
            cap = edges[f"stand.drum_cap.{tag}"]
            self.assertEqual(column["a"], path["head"])
            self.assertEqual(column["b"], path["floor"])
            self.assertEqual(cap["a"], path["seat"])
            self.assertEqual(cap["b"], path["head"])
            self.assertEqual(column["section_shape"], "welded-i")
            self.assertEqual(column["damage"]["material"], "hy80")
            self.assertAlmostEqual(
                nodes[path["head"]]["reference_position"][0],
                nodes[path["floor"]]["reference_position"][0])
            self.assertAlmostEqual(
                nodes[path["head"]]["reference_position"][2],
                nodes[path["floor"]]["reference_position"][2])

    def test_first_lift_phase_drives_only_four_inner_pillars(self):
        import stand
        spec = stand.Stand()
        runtime = stand.outrigger_set(spec, carried_kg=20_000.0)
        outer_before = {name: leg.extension_m for name, leg in runtime.legs.items()
                        if name.startswith("outer.")}
        commands = {name: (1.0 if name.startswith("inner.") else 0.0)
                    for name in runtime.legs}
        loads = {name: (20_000.0 * 9.80665 / 4.0
                        if name.startswith("inner.") else 0.0)
                 for name in runtime.legs}

        def supply(demand, pressure):
            return {"pressure_pa": pressure, "delivered_l_min": demand,
                    "shaft_load_w": pressure * demand / 60_000.0 / .85}

        result = runtime.step(.05, commands, supply=supply, loads_n=loads)

        self.assertTrue(all(leg.extension_m > 0.0 for name, leg in runtime.legs.items()
                            if name.startswith("inner.")))
        self.assertEqual(outer_before,
                         {name: leg.extension_m for name, leg in runtime.legs.items()
                          if name.startswith("outer.")})
        self.assertGreater(result["pump"]["shaft_load_w"], 0.0)

    def test_outrigger_valve_command_meters_flow_exactly_once(self):
        import stand
        spec = stand.Stand()
        runtime = stand.outrigger_set(spec, carried_kg=20_000.0)
        commands = {name: (0.25 if name.startswith("inner.") else 0.0)
                    for name in runtime.legs}
        loads = {name: (20_000.0 * 9.80665 / 4.0
                        if name.startswith("inner.") else 0.0)
                 for name in runtime.legs}

        def supply(demand, pressure):
            return {"pressure_pa": pressure, "delivered_l_min": demand,
                    "shaft_load_w": pressure * demand / 60_000.0 / .85}

        runtime.step(1.0, commands, supply=supply, loads_n=loads)
        # OutriggerSet defines full spool as 60 mm/s; quarter spool must be
        # 15 mm/s, not 3.75 mm/s from applying 0.25 twice.
        for name, leg in runtime.legs.items():
            if name.startswith("inner."):
                self.assertAlmostEqual(leg.extension_m, 0.015, places=6)

    def test_each_powerplant_has_four_beam_mounts_on_its_bay_platform(self):
        graph, _platform, _box, _spec, _stand, site = station.build()
        document = graph.as_document()
        for side in ("port", "starboard"):
            engine = f"engine.{side}"
            engine_node = next(node for node in document["nodes"]
                               if node["identity"] == engine)
            expected = set(engine_node["structural_mount_targets"])
            self.assertTrue(all(target.startswith(
                f"powerplant.{side}.floor.node.") for target in expected))
            mounts = [edge for edge in document["edges"]
                      if edge["a"] == engine
                      and edge.get("part_role") == "engine-mount"]
            self.assertEqual(len(mounts), 4)
            self.assertEqual({edge["b"] for edge in mounts}, expected)
            self.assertTrue(all(edge.get("beam_solvable") for edge in mounts))

            pallet_nodes = [node for node in document["nodes"]
                            if node["identity"].startswith(
                                f"powerplant.{side}.floor.node.")]
            pallet_xyz = np.asarray([node["reference_position"]
                                     for node in pallet_nodes])
            self.assertAlmostEqual(np.ptp(pallet_xyz[:, 0]), 1.48)
            self.assertAlmostEqual(np.ptp(pallet_xyz[:, 2]), 1.36)

            package = site["powerplants"][side]
            skid_mounts = [edge for edge in document["edges"]
                           if edge["a"] == package["skid"]
                           and edge.get("part_role") == "utility-skid-mount"]
            self.assertEqual(len(skid_mounts), 4)
            self.assertEqual({edge["b"] for edge in skid_mounts}, expected)
            self.assertTrue(all(edge.get("beam_solvable") for edge in skid_mounts))

        winches = [edge for edge in document["edges"]
                   if edge.get("part_role") ==
                   "four-corner-powerplant-chain-winch"]
        self.assertEqual(len(winches), 8)
        self.assertTrue(all(edge.get("tension_only")
                            and edge.get("load_share") == 0.25
                            and edge["maximum_rest_length_m"]
                            > edge["minimum_rest_length_m"]
                            for edge in winches))
        locks = [edge for edge in document["edges"]
                 if edge.get("part_role") ==
                 "powerplant-pallet-positive-lock"]
        self.assertEqual(len(locks), 16)
        self.assertEqual(sum(edge.get("lock_engaged", False)
                             for edge in locks), 8)
        self.assertTrue(all(edge.get("structural_participation") is False
                            for edge in locks
                            if edge.get("lock_position") == "lower"))

        service_edges = [edge for edge in document["edges"]
                         if edge["identity"].startswith("station.service.")]
        # Fuel is no longer a generic central ballast tank.  Hydraulic,
        # pneumatic and refrigerant retain four half-to-bay pigtails plus
        # two metaconduit-to-machine pigtails apiece.
        self.assertEqual(len(service_edges), 18)
        self.assertTrue(all(edge.get("routed") for edge in service_edges))
        self.assertTrue(all(not edge.get("beam_solvable") for edge in service_edges))
        self.assertEqual(len(site["storage"]["parts"]), 6)
        drums = [node for node in document["nodes"]
                 if node.get("part_role") ==
                 "post-mounted-50-gallon-fuel-drum"]
        self.assertEqual(len(drums), 4)
        self.assertTrue(all(abs(node["fluid_volume_l"] - 189.2705892)
                            < 1.0e-6 for node in drums))
        self.assertEqual({node["fluid"] for node in drums},
                         {"ultra-low-sulfur-diesel", "jet-a-kerosene"})
        by_identity = {node["identity"]: node for node in document["nodes"]}
        for drum in drums:
            side, index = drum["identity"].split(".")[2], int(
                drum["identity"].split(".")[-1])
            rear_post = (site["lower_bays"][side]["ir"],
                         site["lower_bays"][side]["or"])[index]
            post = by_identity[rear_post]
            self.assertLess(
                drum["reference_position"][2] + drum["body_half_extent_m"][2],
                post["reference_position"][2] - post["body_half_extent_m"][2])
            self.assertEqual(drum["location_role"],
                             "exterior-back-post-rack")
            self.assertEqual(drum["shape"], "drum")
        pumps = [node for node in document["nodes"]
                 if node.get("part_role") ==
                 "auto-off-drop-in-barrel-fuel-pump"]
        self.assertEqual(len(pumps), 4)
        self.assertTrue(all(node["quick_disconnect"] and node["auto_off"]
                            for node in pumps))
        engines = [node for node in document["nodes"]
                   if node.get("part_role") == "managed-engine-object"]
        self.assertEqual(len(engines), 2)
        self.assertTrue(all(node["render_primitive"] is False
                            and node["rendered_by"] == "engine-owned-baked-mesh"
                            for node in engines))
        fuel_ports = [node for node in document["nodes"]
                      if node.get("part_role") ==
                      "engine-external-fuel-intake-port"]
        self.assertEqual(len(fuel_ports), 2)
        fuel_lines = [edge for edge in document["edges"]
                      if edge.get("circuit_identity", "").startswith(
                          "station.fuel.")]
        fuel_pairs = {(edge["a"], edge["b"]) for edge in fuel_lines}
        for side, fuel in (("port", "ultra-low-sulfur-diesel"),
                           ("starboard", "jet-a-kerosene")):
            port = by_identity[f"engine.{side}.external_fuel_in"]
            self.assertEqual(port["solver_condensed_into"], f"engine.{side}")
            self.assertEqual(port["accepted_fuel"], fuel)
            self.assertIn((f"station.fuel.manifold.{fuel}",
                           port["identity"]), fuel_pairs)
        pickups = [edge for edge in fuel_lines
                   if edge["identity"].endswith(".pickup")]
        removable = [edge for edge in fuel_lines
                     if edge["identity"].endswith(".to_manifold")]
        self.assertEqual(len(pickups), 4)
        self.assertTrue(all(edge["fluid_route_class"] == "rigid-orthogonal"
                            and edge["route_obstacle_aware"]
                            for edge in pickups))
        self.assertEqual(len(removable), 4)
        self.assertTrue(all(edge["fluid_route_class"] == "flexible-hose"
                            and edge["route_length_m"] > edge["rest_length"]
                            for edge in removable))
        jackets = [edge for edge in document["edges"]
                   if edge.get("part_role") ==
                   "thermally-coupled-isolated-service-bundle"]
        self.assertEqual(len(jackets), 2)
        self.assertTrue(all(not edge.get("beam_solvable") for edge in jackets))
        self.assertTrue(all("coolant-supply" in edge["contained_channels"]
                            and "exhaust" in edge["contained_channels"]
                            for edge in jackets))

        exchanger = next(node for node in document["nodes"] if node["identity"] ==
                         "station.cooling.double_wall_water_to_coolant_hx")
        self.assertTrue(exchanger["fluids_isolated"])
        self.assertTrue(exchanger["interstitial_leak_detection"])
        tank_mounts = [edge for edge in document["edges"]
                       if edge.get("part_role") == "water-tank-cradle-mount"]
        self.assertEqual(len(tank_mounts), 4)
        self.assertTrue(all(edge.get("load_share") == 0.5
                            for edge in tank_mounts))
        self.assertTrue(all(edge["a"].startswith(
                                "station.cooling.potable_tank.")
                            and ".foot." in edge["a"]
                            and edge["b"].startswith(
                                "station.cooling.floor_rail.")
                            for edge in tank_mounts))
        tanks = [node for node in document["nodes"]
                 if node.get("part_role") == "chilled-potable-thermal-store"]
        self.assertEqual(len(tanks), 2)
        self.assertTrue(all(node.get("installed_under_counter")
                            and node.get("clear_central_aisle_m", 0.0) > 1.4
                            for node in tanks))
        plate_node = next(node for node in document["nodes"]
                          if node.get("surface_role") ==
                          "tank-service-floor-plate")
        floor_y = (float(plate_node["reference_position"][1])
                   + float(plate_node["thickness_m"]) / 2.0)
        for node in tanks:
            radius = float(node["drum_radius_m"])
            centre_y = float(node["reference_position"][1])
            self.assertAlmostEqual(centre_y - radius, floor_y)
            self.assertLessEqual(centre_y + radius, floor_y + 0.92)
        coolant_mounts = [edge for edge in document["edges"]
                          if edge.get("part_role") == "cooling-plant-mount"]
        self.assertEqual(len(coolant_mounts), 16)
        self.assertTrue(all(edge["b"].startswith("stand.lower_room.")
                            for edge in coolant_mounts))
        nitrogen = [node for node in document["nodes"]
                    if str(node.get("part_role", "")).endswith(
                        "-nitrogen-torpedo")]
        self.assertEqual(len(nitrogen), 6)
        self.assertTrue(all(node.get("installed_on") ==
                            "tank-service-floor-plate"
                            and str(node.get("solver_condensed_into", ""))
                            .startswith("stand.lower_room.floor.node.")
                            for node in nitrogen))
        bottle_clamps = [edge for edge in document["edges"]
                         if edge.get("part_role") ==
                         "secured-nitrogen-bottle-floor-clamp"]
        self.assertEqual(len(bottle_clamps), 12)
        self.assertTrue(all(edge.get("structural_participation") is False
                            for edge in bottle_clamps))
        bottles = [node for node in document["nodes"]
                   if str(node.get("part_role", "")).endswith(
                       "-nitrogen-torpedo")]
        self.assertEqual(len(bottles), 6)
        self.assertTrue(all(node.get("installed_on") ==
                            "tank-service-floor-plate"
                            and node.get("clamp_type") ==
                            "two-saddle-through-bolted"
                            and str(node.get("solver_condensed_into", ""))
                            .startswith("stand.lower_room.floor.node.")
                            for node in bottles))
        clamps = [edge for edge in document["edges"]
                  if edge.get("part_role") ==
                  "secured-nitrogen-bottle-floor-clamp"]
        self.assertEqual(len(clamps), 12)
        self.assertTrue(all(edge.get("structural_participation") is False
                            for edge in clamps))

    def test_gun_and_upper_mount_are_balanced_at_platform_travel_midpoint(self):
        graph, _platform, _lower, upper = sled.build()
        document = graph.as_document()
        moved = [node for node in document["nodes"]
                 if "gun_platform_balance_shift_m" in node
                 and float(node.get("mass_kg", 0.0)) > 0.0]
        masses = np.asarray([float(node["mass_kg"]) for node in moved])
        positions = np.asarray([node["reference_position"] for node in moved])
        cg = (positions * masses[:, None]).sum(axis=0) / masses.sum()

        self.assertAlmostEqual(cg[2], upper["gun_travel_midpoint_m"][2],
                               places=10)
        pitch = next(node for node in document["nodes"]
                     if node["identity"] == "turret.pitch")
        platform_y = np.mean([
            next(node["reference_position"][1] for node in document["nodes"]
                 if node["identity"] == upper["corners"][tag])
            for tag in ("aft.a", "fwd.a", "aft.b", "fwd.b")])
        self.assertAlmostEqual(
            pitch["reference_position"][1] - platform_y,
            upper["installed_trunnion_clearance_m"], places=10)
        feet = [node for node in document["nodes"]
                if node["identity"].startswith("fine.base.")]
        self.assertTrue(feet)
        self.assertTrue(all("gun_platform_balance_shift_m" not in node
                            for node in feet))
        positions_by_id = {node["identity"]: np.asarray(
            node["reference_position"], float) for node in document["nodes"]}
        supports = [edge for edge in document["edges"]
                    if edge.get("balanced_mount")]
        self.assertEqual(len(supports), 6)
        self.assertEqual(sorted(edge["radius"] for edge in supports),
                         [0.18, 0.18, 0.22, 0.22, 0.22, 0.22])
        self.assertTrue(all(edge.get("provisional_overbuilt_test_support")
                            and not edge.get("rigid", False)
                            for edge in supports))
        for edge in supports:
            installed = np.linalg.norm(positions_by_id[edge["b"]]
                                       - positions_by_id[edge["a"]])
            self.assertAlmostEqual(edge["rest_length"], installed, places=10)
        self.assertFalse(any(node.get("assembly") in {
            "fine-rig-platform", "fine-rig-mount"}
            for node in document["nodes"]))

    def test_service_tank_geometry_scales_from_declared_capacity(self):
        from station_cooling import CylindricalServiceTank
        small = CylindricalServiceTank("small", "potable-water", 250.0, 200.0)
        large = CylindricalServiceTank("large", "potable-water", 500.0, 400.0)

        self.assertGreater(large.inner_length_m, small.inner_length_m)
        self.assertAlmostEqual(large.inner_length_m,
                               2.0 * small.inner_length_m)
        self.assertGreater(large.mass_kg, small.mass_kg)
        self.assertEqual(large.graph_attributes()["capacity_l"], 500.0)

    def test_live_beam_keeps_every_dof_and_accepts_force_and_moment(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        live = LiveStructure(graph.as_document(), modes=1)

        # `modes=1` is deliberately hostile: it was formerly sufficient to
        # discard almost the whole structure. It is now only a compatibility
        # argument and the live state spans every free physical DOF.
        self.assertEqual(
            len(live.free), len(live.omega) + live.mechanism_shapes.shape[1])
        self.assertGreater(len(live.omega), 1)

        live.apply_wrench("turret.breech", moment=(1000.0, 0.0, 0.0))
        live.step(1.0 / 240.0)
        state = live.dynamic_displacement.reshape(-1, 6)
        self.assertGreater(np.max(np.abs(state[:, 3:])), 0.0)
        self.assertTrue(np.isfinite(live.utilisation()).all())

        live.reset()
        live.apply("turret.breech", (0.0, 0.0, -15_000.0))
        live.step(2.0 / 240.0)
        self.assertGreater(np.max(np.linalg.norm(
            live.dynamic_displacement.reshape(-1, 6)[:, :3], axis=1)), 0.0)
        self.assertNotEqual(
            live.joint_forces.last_element_force[
                "turret.absorber.magnetorheological"], 0.0)

    def test_baked_reference_inverse_is_complete_not_reduced(self):
        graph = ProductionGraph("reference-inverse-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("tip", (1.0, 0.0, 0.0), "chassis-load-node")
        graph.edge("cantilever", "fixed", "tip", "rigid-distance",
                   radius=0.03)
        live = LiveStructure(graph.as_document(), modes=1)

        # At dt=0 the effective operator is exactly M.  A complete baked
        # factorisation therefore reproduces M^-1 on every free coordinate.
        rhs = np.linspace(-1.0, 1.0, len(live.free))
        np.testing.assert_allclose(
            live.apply_baked_reference_inverse(rhs, 0.0),
            rhs / live.mass_free,
            rtol=1.0e-7, atol=1.0e-8)
        self.assertEqual(live.reference_basis_free.shape,
                         (len(live.free), len(live.free)))

    def test_live_reuses_reference_assembly_and_modal_energy_is_exact(self):
        graph = ProductionGraph("reference-reuse-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("tip", (1.0, 0.0, 0.0), "chassis-load-node",
                   mass_kg=20.0)
        graph.edge("cantilever", "fixed", "tip", "rigid-distance",
                   radius=0.03)
        document = graph.as_document()
        solver = FrameSolver(document, loads={})
        reference = solver.modes(count=1)
        with patch.object(solver, "solve",
                          side_effect=AssertionError("duplicate assembly")):
            live = LiveStructure(document, solver=solver,
                                 reference=reference)
        live.apply("tip", (0.0, -1000.0, 0.0))
        live.step(1.0 / 240.0)
        energy = live.energy_state()
        displacement = live.displacement()
        direct = 0.5 * float(
            displacement @ (live.stiffness @ displacement))
        self.assertAlmostEqual(energy["beam_elastic_j"], direct, places=8)

        batch = solver.endpoint_motion_batch(np.vstack((
            displacement, live.velocity, live.acceleration)))
        for i, state in enumerate((displacement, live.velocity,
                                   live.acceleration)):
            np.testing.assert_allclose(batch[i], solver.endpoint_motion(state))

    def test_live_solve_injects_node_and_edge_stats_into_graph_objects(self):
        graph = ProductionGraph("solver-object-stats-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("tip", (1.0, 0.0, 0.0), "chassis-load-node",
                   mass_kg=12.0)
        graph.edge("cantilever", "fixed", "tip", "rigid-distance",
                   radius=0.03)
        document = graph.as_document()
        live = LiveStructure(document, modes=1)
        live.apply("tip", (0.0, -1_000.0, 0.0))
        live.step(1.0 / 240.0)
        summary = live.inject_solver_stats()

        tip = next(node for node in document["nodes"]
                   if node["identity"] == "tip")
        beam = next(edge for edge in document["edges"]
                    if edge["identity"] == "cantilever")
        self.assertEqual(tip["solver_stats"]["sequence"], 1)
        self.assertGreater(tip["solver_stats"]["speed_m_s"], 0.0)
        self.assertIn("axial_strain", beam["solver_stats"])
        self.assertIn("bending_surface_strain", beam["solver_stats"])
        self.assertIn("stress_pa", beam["solver_stats"])
        self.assertEqual(document["solver_stats"]["coupled_steps"],
                         live.last_coupled_steps)
        self.assertGreater(summary["energy"]["kinetic_j"], 0.0)
        self.assertGreaterEqual(summary["energy"]["beam_elastic_j"], 0.0)
        self.assertEqual(summary["member_utilisation"].shape, (1,))

    def test_baked_subobject_mass_and_motion_are_condensed_into_gestalt(self):
        graph = ProductionGraph("gestalt-condensation-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("machine", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=20.0)
        graph.node("machine.internal", (1.0, 0.2, 0.0), "machine-subpart",
                   mass_kg=5.0, structural_participation=False,
                   solver_condensed_into="machine", solver_condensed_mass=True)
        graph.node("machine.port", (1.0, 0.0, 0.2), "fluid-port",
                   mass_kg=0.1)
        graph.edge("mount", "fixed", "machine", "rigid-distance",
                   radius=.03)
        graph.edge("route", "machine.internal", "machine.port",
                   "coolant-line", radius=.01)
        document = graph.as_document()
        live = LiveStructure(document, modes=1)

        machine_i = live.solver.index["machine"]
        internal_i = live.solver.index["machine.internal"]
        port_i = live.solver.index["machine.port"]
        self.assertEqual(live.solver.solver_node_mass[machine_i], 25.0)
        self.assertFalse(live.solver.structural_node[internal_i])
        self.assertFalse(live.solver.structural_node[port_i])
        self.assertEqual(len(live.free), 6)
        live.apply("machine", (0.0, -1000.0, 0.0))
        live.step(1.0e-3)
        live.inject_solver_stats()
        internal = document["nodes"][internal_i]["solver_stats"]
        self.assertEqual(internal["condensed_into"], "machine")
        self.assertGreater(internal["speed_m_s"], 0.0)

    def test_condensed_surface_port_retains_offset_wrench_and_member(self):
        graph = ProductionGraph("surface-port-condensation-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=20.0, half_extent_m=(0.2, 0.3, 0.4))
        graph.node("body.port", (1.0, 0.5, 0.0), "structural-mount-port",
                   mass_kg=0.4, wrench_point=True, surface_of="body",
                   solver_condensed_into="body", solver_condensed_mass=False)
        graph.edge("external_member", "fixed", "body.port",
                   "rigid-distance", radius=0.03)
        document = graph.as_document()
        solver = FrameSolver(document, gravity=False)

        self.assertEqual([edge["identity"] for edge in solver.members],
                         ["external_member"])
        self.assertTrue(solver.structural_node[solver.index["body"]])
        self.assertFalse(solver.structural_node[solver.index["body.port"]])
        self.assertEqual(len(document["nodes"]) * 6
                         - len(set(solver._fixed_dofs())), 6)

        load = np.zeros(len(document["nodes"]) * 6)
        solver._add_endpoint_wrench(
            load, solver.index["body.port"], (10.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0))
        body = solver.index["body"] * 6
        np.testing.assert_allclose(load[body:body + 6],
                                   (10.0, 0.0, 0.0, 0.0, 0.0, -5.0))

    def test_live_recovery_materialises_condensed_surface_port_motion(self):
        graph = ProductionGraph("surface-port-live-recovery-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=20.0, half_extent_m=(0.2, 0.3, 0.4))
        graph.node("body.port", (1.0, 0.5, 0.0), "structural-mount-port",
                   wrench_point=True, surface_of="body",
                   solver_condensed_into="body", solver_condensed_mass=False)
        graph.edge("external_member", "fixed", "body.port",
                   "rigid-distance", radius=0.03)
        document = graph.as_document()
        solver = FrameSolver(document, loads={"body": (0.0, -1000.0, 0.0)},
                             gravity=False)
        reference = solver.modes(count=1)
        live = LiveStructure(document, solver=solver, reference=reference)
        recovered = solver.solve()
        response = live.member_response()

        self.assertAlmostEqual(
            response["axial_strain"][0], recovered["axial_strain"][0],
            places=12)
        self.assertAlmostEqual(
            response["shear_strain"][0], recovered["shear_strain"][0],
            places=12)
        port_i = solver.index["body.port"]
        np.testing.assert_allclose(
            live.positions()[port_i],
            solver.position[port_i]
            + solver._endpoint_motion(live.displacement(), port_i)[:3])

    def test_complete_reference_coordinates_match_physical_newmark_step(self):
        graph = ProductionGraph("complete-coordinate-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("tip", (1.0, 0.0, 0.0), "chassis-load-node",
                   mass_kg=12.0)
        graph.edge("cantilever", "fixed", "tip", "rigid-distance",
                   radius=0.03)
        live = LiveStructure(graph.as_document(), modes=1)
        live.apply_wrench("tip", force=(17.0, -31.0, 9.0),
                          moment=(2.0, -3.0, 5.0))
        dt = 1.0 / 1000.0

        expected_acceleration = live.apply_baked_reference_inverse(
            live.force[live.free], dt)
        expected_u = 0.25 * dt * dt * expected_acceleration
        expected_v = 0.5 * dt * expected_acceleration
        expected_static_residual = (
            live.force[live.free]
            - live.stiffness_free @ expected_u
            - live.damping_free @ expected_v)
        live.step(dt)

        np.testing.assert_allclose(
            live.dynamic_displacement[live.free], expected_u,
            rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(
            live.velocity[live.free], expected_v,
            rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(
            live.last_static_residual_free, expected_static_residual,
            rtol=2e-9, atol=2e-10)

    def test_static_nullspace_residual_drives_released_mechanism(self):
        graph = ProductionGraph("loaded-slider-nullspace-witness")
        graph.node("fixed", (0.0, 0.0, 0.0),
                   "structural-body-pin-frame-foot", fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=10.0)
        graph.edge("slide", "fixed", "body", "single-axis-slider",
                   radius=0.03)
        document = graph.as_document()
        solver = FrameSolver(document, loads={"body": (1000.0, 0.0, 0.0)},
                             gravity=False)
        reference = solver.modes(count=1)
        live = LiveStructure(document, solver=solver, reference=reference)
        body_x = solver.index["body"] * 6

        self.assertEqual(reference["mechanism_count"], 1)
        self.assertAlmostEqual(live.base_force[body_x], 1000.0)
        live.step(1.0e-3)
        self.assertGreater(live.dynamic_displacement[body_x], 0.0)

    def test_support_leveling_opens_only_outrigger_length_locks(self):
        original = {"nodes": [], "edges": [
            {"identity": "support.lock", "part_role":
             "outrigger-positive-length-lock", "lock_engaged": True},
            {"identity": "arch.lock", "part_role":
             "arch-crown-positive-height-lock", "lock_engaged": True},
        ], "_graph_columns": object()}
        leveled = station.support_leveling_document(original)
        original_edges = {edge["identity"]: edge
                          for edge in original["edges"]}
        leveled_edges = {edge["identity"]: edge
                         for edge in leveled["edges"]}
        changed = {identity for identity in original_edges
                   if original_edges[identity].get("lock_engaged")
                   != leveled_edges[identity].get("lock_engaged")}

        self.assertTrue(changed)
        self.assertTrue(all(
            leveled_edges[identity].get("part_role")
            == "outrigger-positive-length-lock"
            for identity in changed))
        self.assertTrue(all(not leveled_edges[identity]["lock_engaged"]
                            for identity in changed))

    def test_space_path_creates_round_at_current_posed_muzzle(self):
        field = ProjectileField()
        muzzle = np.array([1.25, 3.5, -0.75])
        direction = np.array([0.0, 0.0, 1.0])

        shot = station.fire_viewer_projectile(
            field, 120.0, muzzle, direction)

        self.assertIs(field.rounds[0], shot)
        np.testing.assert_allclose(shot.position, muzzle)
        np.testing.assert_allclose(
            shot.velocity / shot.speed_m_s, direction)
        self.assertGreater(shot.speed_m_s, 0.0)

    def test_projectile_visual_is_a_reusable_mesh_span(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        render = station.projectile_visual_document(document)

        self.assertEqual(len(render["nodes"]), len(document["nodes"]) + 2)
        self.assertEqual(len(render["edges"]), len(document["edges"]) + 1)
        edge = render["edges"][-1]
        self.assertEqual(edge["a"], "viewer.projectile.tail")
        self.assertEqual(edge["b"], "viewer.projectile.round")
        self.assertGreater(edge["radius"], 0.0)

    def test_attachment_stations_are_link_metadata_not_fake_members(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()

        self.assertFalse(any(
            node["identity"].startswith("station.")
            for node in document["nodes"]))
        self.assertFalse(any(
            edge["identity"].startswith("station.")
            for edge in document["edges"]))
        link = next(edge for edge in document["edges"]
                    if edge["identity"] == "gun.link.fwd.a")
        self.assertEqual(
            [station["fraction"]
             for station in link["attachment_stations"]],
            list(sled.STATIONS),
        )

    def test_oleo_bank_excludes_guides_lockups_and_commanded_actuators(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()

        force_edges = structure_native.force_joints(document)
        constraints = {edge["constraint"] for edge in force_edges}

        self.assertEqual(constraints,
                         {"spring-damper", "oleo-recoil-slide",
                          "belleville-preload-stack",
                          "linear-hydraulic-actuator"})
        self.assertIn("turret.absorber.spring",
                      {edge["identity"] for edge in force_edges})
        self.assertNotIn("turret.journal.lower.bear.0",
                         {edge["identity"] for edge in force_edges})
        self.assertNotIn("turret.recoil_slide",
                         {edge["identity"] for edge in force_edges})
        platform_dampers = [edge for edge in force_edges
                            if edge.get("part_role") == "platform-damper"]
        platform_rams = [edge for edge in force_edges
                         if edge.get("part_role") == "platform-actuator"]
        self.assertEqual(len(platform_dampers), 8)
        self.assertEqual(len(platform_rams), 4)
        self.assertTrue(all(edge in structure_native.linear_spring_joints(document)
                            for edge in platform_dampers + platform_rams))
        solver = FrameSolver(document, gravity=False)
        beam_identities = {edge["identity"] for edge in solver.members}
        self.assertTrue(all(edge["identity"] not in beam_identities
                            for edge in platform_dampers + platform_rams))
        distributor = GraphJointForces(document, linear_embedded=True)
        distributor.command_platform_preload(1.0)
        self.assertTrue(all(
            edge["commanded_preload_n"] == edge["maximum_preload_n"]
            for edge in distributor.linear_edges
            if edge.get("part_role") == "platform-actuator"))
        distributor.command_platform_preload(0.0)
        self.assertTrue(all(
            edge["commanded_preload_n"] == 0.0
            for edge in distributor.linear_edges
            if edge.get("part_role") == "platform-actuator"))
        self.assertLess(len(force_edges),
                        len(structure_native.travelling_joints(document)))

    def test_plain_linear_equilibrator_is_not_rewritten_as_an_oleo(self):
        graph = ProductionGraph("linear-equilibrator-witness")
        graph.node("anchor", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure")
        graph.edge("equilibrator", "anchor", "body", "spring-damper",
                   stiffness_n_per_m=8_000.0, preload_force_n=400.0,
                   compression_damping_n_s_per_m=300.0,
                   rebound_damping_n_s_per_m=420.0)
        document = graph.as_document()
        distributor = GraphJointForces(document)

        self.assertEqual([edge["identity"] for edge in distributor.linear_edges],
                         ["equilibrator"])
        self.assertEqual(distributor.bank.n, 0)

        position = np.asarray(
            [node["reference_position"] for node in document["nodes"]], float)
        velocity = np.zeros_like(position)
        velocity[distributor.index["body"], 0] = -1.0
        compressed = position.copy()
        compressed[distributor.index["body"], 0] -= 0.1
        force = distributor.evaluate(1.0 / 240.0, compressed, velocity)

        self.assertAlmostEqual(
            distributor.last_element_force["equilibrator"], 1_500.0)
        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-12)

        velocity[distributor.index["body"], 0] = 1.0
        distributor.evaluate(1.0 / 240.0, position, velocity)
        self.assertAlmostEqual(
            distributor.last_element_force["equilibrator"], -20.0)

    def test_plain_spring_preload_and_rate_are_in_static_frame_equilibrium(self):
        graph = ProductionGraph("preloaded-frame-witness")
        graph.node("fixed", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world")
        graph.node("body", (1.0, 0.0, 0.0), "load-bearing-structure",
                   mass_kg=10.0)
        graph.edge("guide", "fixed", "body", "single-axis-slider",
                   radius=0.02)
        graph.edge("spring", "fixed", "body", "spring-damper",
                   stiffness_n_per_m=8_000.0, preload_force_n=400.0,
                   compression_damping_n_s_per_m=300.0,
                   rebound_damping_n_s_per_m=420.0)

        solver = FrameSolver(graph.as_document(), gravity=False)
        solved = solver.solve()
        displacement = solved["displacement"][solver.index["body"], 0]

        self.assertAlmostEqual(displacement, 400.0 / 8_000.0, places=9)
        modes = solver.modes(count=20)
        self.assertGreater(modes["first_hz"], 0.0)

    def test_oleo_and_body_advance_on_the_same_coupled_substeps(self):
        graph = ProductionGraph("coupled-oleo-witness")
        graph.node("anchor", (0.0, 0.0, 0.0), "load-bearing-structure",
                   fixed_to="world")
        graph.node("body", (0.0, 0.0, 0.5), "recoiling-weapon-mass",
                   mass_kg=171.3, half_extent_m=(0.1, 0.1, 0.2))
        graph.edge("recoil", "anchor", "body", "oleo-recoil-slide",
                   radius=0.036, slide_axis=(0.0, 0.0, 1.0),
                   piston_area_m2=np.pi * 0.05 ** 2,
                   gas_volume_m3=0.0085, charge_pressure_pa=5.0e5,
                   orifice_area_m2=np.pi * 0.0215 ** 2 / 4.0 * 0.7,
                   stroke_m=0.420)
        live = LiveStructure(graph.as_document(), modes=1)

        live.apply("body", (0.0, 0.0, -120_000.0))
        live.step(0.010)

        self.assertTrue(np.isfinite(live.dynamic_displacement).all())
        self.assertTrue(np.isfinite(live.velocity).all())
        self.assertGreater(abs(live.joints.compression[0]), 0.0)
        self.assertGreater(abs(live.dynamic_displacement[
            live.solver.index["body"] * 6 + 2]), 0.0)

        live.joints.arrays["velocity_m_s"][0] = 100.0
        count, stable_h = live.joints.coupled_substep_plan(
            1.0 / 240.0, live.joint_forces.passive_effective_mass_kg)
        travel_count, travel_h = live.joints.substep_plan(1.0 / 240.0)
        self.assertGreater(count, travel_count)
        self.assertLess(stable_h, travel_h)

    def test_recoil_absorber_span_can_physically_contain_its_stroke(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        positions = {
            node["identity"]: np.asarray(node["reference_position"], float)
            for node in document["nodes"]
        }
        absorber = next(edge for edge in document["edges"]
                        if edge["identity"] == "turret.absorber.spring")
        span = np.linalg.norm(
            positions[absorber["b"]] - positions[absorber["a"]])

        self.assertGreaterEqual(span, absorber["recoil_stroke_m"])

    def test_triple_recoil_pack_is_physical_and_force_balanced(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        distributor = GraphJointForces(document)
        position = np.asarray(
            [node["reference_position"] for node in document["nodes"]],
            float)
        velocity = np.zeros_like(position)
        index = distributor.index
        velocity[index["turret.spine.0"], 2] = -1.0

        force = distributor.evaluate(1.0e-4, position, velocity)

        np.testing.assert_allclose(force.sum(axis=0), 0.0, atol=1.0e-8)
        self.assertTrue(np.isfinite(force).all())
        self.assertNotEqual(
            distributor.last_element_force[
                "turret.absorber.magnetorheological"], 0.0)

        edges = {edge["identity"]: edge for edge in document["edges"]}
        self.assertEqual(edges["turret.recoil_slide"]["constraint"],
                         "single-axis-slider")
        self.assertEqual(
            {edge["identity"] for edge in distributor.linear_edges
             if edge["identity"].startswith("turret.absorber.")},
            {"turret.absorber.spring",
             "turret.absorber.recuperator.orifice",
             "turret.absorber.recuperator.magnetorheological"})
        self.assertEqual(
            {edge["identity"] for edge in distributor.passive_edges
             if edge["identity"].startswith("turret.absorber.")},
            {"turret.absorber.orifice"})
        rates = [float(edge["spring_rate_n_per_m"])
                 for edge in distributor.linear_edges
                 if edge["identity"].startswith("turret.absorber.")]
        self.assertEqual(len(rates), 3)
        self.assertAlmostEqual(max(rates), min(rates), places=9)
        self.assertEqual(
            {edges[name]["constraint"] for name in (
                "turret.journal.lower.bear.0", "turret.journal.upper.bear.0",
                "turret.journal.side.bear.0", "turret.journal.side.bear.1")},
            {"single-axis-slider"})
        self.assertEqual(sum(node.get("section") ==
                             "square-journal-with-three-bores"
                             for node in document["nodes"]), 6)
        self.assertEqual(sum(node.get("part_role") ==
                             "barrel-jacket-chamber"
                             for node in document["nodes"]), 80)
        barrel_subobjects = [node for node in document["nodes"]
                             if (node.get("part_role") ==
                                 "barrel-jacket-chamber"
                                 or node["identity"] ==
                                 "turret.filler_sleeve")]
        self.assertTrue(all(node.get("solver_condensed_into") ==
                            "turret.outer_barrel"
                            for node in barrel_subobjects))
        filler = next(node for node in barrel_subobjects
                      if node["identity"] == "turret.filler_sleeve")
        self.assertTrue(filler["solver_condensed_mass"])
        self.assertFalse(any(node["identity"].startswith("weapon.")
                             for node in document["nodes"]))

    def test_recoil_guides_release_the_declared_bore_axis(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        solver = FrameSolver(document, gravity=False)
        edge = next(edge for edge in document["edges"]
                    if edge["identity"] == "turret.collar_guide")
        ia, ib = solver.index[edge["a"]], solver.index[edge["b"]]
        from frame_solver import _element_frame, _released_dofs
        frame = _element_frame(edge, solver.position[ia], solver.position[ib])

        np.testing.assert_allclose(frame[0], (0.0, 0.0, 1.0), atol=1e-12)
        self.assertIn(6, _released_dofs(edge))

    def test_thrust_seat_is_after_not_across_the_recoil_slide(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        edges = {edge["identity"]: edge for edge in document["edges"]}

        self.assertEqual(
            {edges["turret.thrust_seat"]["a"],
             edges["turret.thrust_seat"]["b"]},
            {"turret.journal.lower", "turret.pitch"})
        for k in range(3):
            self.assertEqual(
                {edges[f"turret.seat_preload.{k}"]["a"],
                 edges[f"turret.seat_preload.{k}"]["b"]},
                {"turret.journal.lower", "turret.pitch"})

    def test_rear_bump_stop_bodies_have_real_braced_mounts(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        for side in ("left", "right"):
            for body in ("buffer", "striker"):
                name = f"turret.bump_stop.{side}.{body}"
                mounts = [edge for edge in document["edges"]
                          if name in (edge["a"], edge["b"])
                          and edge["constraint"] == "rigid-distance"]
                self.assertEqual(len(mounts), 2)

    def test_each_swing_has_equalized_forward_and_aft_stops(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        stops = {edge["identity"]: edge for edge in document["edges"]
                 if edge.get("part_role") in {
                     "adjustable-forward-preload-stop",
                     "fixed-full-aft-maintenance-stop"}}
        self.assertEqual(set(stops), {
            "dangling.stop.forward", "dangling.stop.aft",
            "gun.stop.forward", "gun.stop.aft"})
        for edge in stops.values():
            self.assertEqual(len(edge["coupled_slide_edges"]), 4)
            self.assertAlmostEqual(sum(edge["slide_load_share"]), 1.0)
            self.assertTrue(edge["equalized_across_bays"])
        for stage in ("dangling", "gun"):
            forward = stops[f"{stage}.stop.forward"]
            self.assertEqual(forward["clearance_m"], 0.0)
            self.assertGreater(forward["installed_stage_preload_n"], 0.0)
            self.assertGreater(stops[f"{stage}.stop.aft"]["clearance_m"], 0.0)

    def test_platform_preload_can_be_commanded_in_total_stage_newtons(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        distributor = GraphJointForces(graph.as_document())
        applied = distributor.command_platform_preload_force(
            30_000.0, stage="dangling")
        self.assertEqual(applied, {"dangling": 30_000.0})
        rams = [edge for edge in distributor.linear_edges
                if edge.get("part_role") == "platform-actuator"
                and edge.get("stage") == "dangling"]
        self.assertEqual(len(rams), 2)
        self.assertTrue(all(edge["commanded_preload_n"] == 15_000.0
                            for edge in rams))
        stop = next(edge for edge in distributor.bump_stop_edges
                    if edge["identity"] == "dangling.stop.forward")
        self.assertEqual(stop["clearance_m"], stop["nominal_clearance_m"])
        distributor.command_platform_preload_force(0.0, stage="dangling")
        self.assertLess(stop["clearance_m"], stop["nominal_clearance_m"])

    def test_recoil_bump_stops_are_open_then_balance_end_force(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        distributor = GraphJointForces(document)
        position = np.asarray(
            [node["reference_position"] for node in document["nodes"]],
            float)
        velocity = np.zeros_like(position)
        recoil_stops = [edge for edge in distributor.bump_stop_edges
                        if edge["identity"].startswith("turret.bump_stop.")]
        self.assertEqual(len(recoil_stops), 1)
        self.assertEqual(len(distributor._paths(recoil_stops[0])),
                         2)

        open_force = distributor.evaluate(1.0e-4, position, velocity)
        for edge in distributor.bump_stop_edges:
            self.assertEqual(distributor.last_element_force[edge["identity"]],
                             0.0)

        np.testing.assert_allclose(open_force.sum(axis=0), 0.0,
                                   atol=1.0e-8)

    def test_force_push_evolves_the_complete_unlocked_rig_while_it_rings(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        live = LiveStructure(document, modes=1)
        _barrel, axis, _offset = station.gun_muzzle_mount(document)

        stopped = station.push_barrel_until_structure_resists(
            live, document, axis, force_n=400_000.0,
            dt=1.0 / 60.0, timeout_s=0.05)

        self.assertTrue(stopped["oscillation_permitted"])
        self.assertIn("globally_still", stopped)
        self.assertTrue(np.isfinite([
            stopped["max_linear_velocity_m_s"],
            stopped["max_angular_velocity_rad_s"],
            stopped["max_linear_acceleration_m_s2"],
            stopped["max_angular_acceleration_rad_s2"],
            stopped["static_residual_ratio"]]).all())
        for identity in ("turret.absorber.spring",
                         "turret.absorber.orifice",
                         "turret.absorber.magnetorheological"):
            self.assertGreater(stopped["peak_joint_force_n"][identity], 0.0)

    def test_recoil_reactions_feed_the_beam_solve_from_drooped_geometry(self):
        graph, _platform, _dangling, _gun = sled.build(fold=0.0)
        document = graph.as_document()
        gravity_solver = FrameSolver(document=document, loads={}, gravity=True)
        drooped = gravity_solver.settled_position(gravity_solver.solve())
        distributor = GraphJointForces(document)
        velocity = np.zeros_like(drooped)
        velocity[distributor.index["turret.spine.0"], 2] = -0.5

        joint_force = distributor.evaluate(1.0e-4, drooped, velocity)
        np.testing.assert_allclose(joint_force.sum(axis=0), 0.0, atol=1.0e-8)
        loads = {
            identity: tuple(float(v) for v in joint_force[i])
            for identity, i in distributor.index.items()
            if np.linalg.norm(joint_force[i]) > 0.0
        }
        loaded = FrameSolver(
            document=document, loads=loads, gravity=True).solve()

        self.assertTrue(np.isfinite(loaded["displacement"]).all())
        self.assertTrue(np.isfinite(loaded["reaction"]).all())
        self.assertGreater(loaded["max_deflection_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
