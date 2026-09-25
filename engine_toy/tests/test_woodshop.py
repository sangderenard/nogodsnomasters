"""First woodshop slice: objects stay Machines and cutting stays in dt."""

import numpy as np
import pytest

import honorary_engine_equation_catalogue as catalogue
from material_topology import (
    MaterialForm,
    SdfMaterialBody,
    WorldObjectGraph,
    chip_batch,
    disincorporate_sdf_body,
)
from machines import Machine, MachinePose, MachineSim, MachineSystem
from sdf_geometry import CapsuleSdf
from woodworking_joints import (
    CompressionSleeveInterface,
    WoodworkingJoint,
    WoodworkingJointInterface,
    bar_clamp,
    pipe_clamp,
    resin_sawhorse_bracket,
)
from woodshop import (
    KILN_DRY_PINE_STUD,
    NOMINAL_STUD_LENGTH_M,
    STARTING_SAWHORSE_BRACKET_COUNT,
    STARTING_STUD_COUNT,
    STUD_THICKNESS_M,
    STUD_WIDTH_M,
    ToothPattern,
    WoodshopSimulation,
    WoodshopWorldRules,
    hand_saw,
    kiln_dried_pine_2x4x8,
    part_box_mesh,
    project_crosscut,
    translate_machine_3d,
)


def test_nominal_two_by_four_is_one_real_machine_part_on_z_zero_plane():
    stock = kiln_dried_pine_2x4x8()

    assert len(stock.parts) == 1
    part = stock.parts[0]
    assert part.part_role == "workpiece"
    assert 2.0 * part.half_extent_m[0] == pytest.approx(NOMINAL_STUD_LENGTH_M)
    assert 2.0 * part.half_extent_m[1] == pytest.approx(STUD_WIDTH_M)
    assert 2.0 * part.half_extent_m[2] == pytest.approx(STUD_THICKNESS_M)
    assert part.position[2] - part.half_extent_m[2] == pytest.approx(0.0)
    assert part.mass_kg == pytest.approx(
        NOMINAL_STUD_LENGTH_M * STUD_WIDTH_M * STUD_THICKNESS_M
        * KILN_DRY_PINE_STUD.density_kg_m3
    )
    assert np.all(np.linalg.eigvalsh(KILN_DRY_PINE_STUD.compliance_matrix()) > 0.0)


def test_starting_world_has_four_brackets_and_a_pile_of_true_eight_foot_studs():
    world = WoodshopSimulation()
    studs = [item for identity, item in world.items.items()
             if identity.startswith("stock.pine-stud.")]
    brackets = [identity for identity in world.items
                if identity.startswith("jig.sawhorse-bracket.")]

    assert len(studs) == STARTING_STUD_COUNT == 8
    assert len(brackets) == STARTING_SAWHORSE_BRACKET_COUNT == 4
    assert all(2.0 * item.sim.machine.parts[0].half_extent_m[0]
               == pytest.approx(8.0 * 0.3048) for item in studs)
    assert len({round(float(item.center_xyz()[2]), 5)
                for item in studs[1:]}) >= 4


def test_woodshop_honorary_family_contains_the_executable_cutting_chain():
    equations = catalogue.equations_by_engine("Woodshop")
    requested = tuple(f"eq_WO4_{number}" for number in range(1, 6))
    pieces = catalogue.law_pieces(("Woodshop",), law_ids=requested)

    assert all(f"eq_WO4_{number}" in equations for number in range(1, 6))
    assert all(f"eq_WO4_{number}" in pieces for number in range(1, 6))
    from src.compiler.native_law_kernels import LLVMPiece
    assert all(isinstance(piece, LLVMPiece) for piece in pieces.values())
    assert all(piece.batch == 1 for piece in pieces.values())
    assert catalogue.ENGINE_SETS["woodshop_slice"] == (
        "Newton", "Timoshenko", "Bragg", "Woodshop"
    )
    assert catalogue.check_law_declarations() == []


def test_world_rules_select_existing_newton_gravity_and_contact_laws():
    world = WoodshopSimulation()

    assert world.world_rules.HONORARY_LAWS == (
        "eq_N1_1", "eq_N1_2", "eq_N4_1",
        "eq_N5_1", "eq_N5_2", "eq_N5_3", "eq_N5_4", "eq_N5_5",
        "eq_N5_6", "eq_N5_7",
    )
    assert isinstance(world.world_rules, WoodshopWorldRules)
    from src.compiler.native_law_kernels import LLVMPiece
    assert isinstance(world.world_rules._laws["eq_N5_7"], LLVMPiece)
    assert world.world_rules._laws["eq_N5_7"].entry.endswith("__eq_N5_7")


def test_newton_world_motion_is_one_batched_sequential_dt_system():
    world = WoodshopSimulation()
    world.advance(0.001)

    rules = world.world_rules
    assert rules.newton_dt_graph.label == "woodshop-newton"
    assert rules.newton_dt_graph.schedule == "sequential"
    assert [child.label for child in rules.newton_dt_graph.children] == [
        "N4.1 gravity", "N1.2 momentum", "N1.1 position",
    ]
    assert len(rules._newton_dt_pieces) == 3
    assert all(piece.batch == len(world.items)
               for piece in rules._newton_dt_pieces)
    assert rules.newton_dt_state.wall_cost_ledger.calls == [1, 1, 1]
    assert not {"eq_N1_1", "eq_N1_2", "eq_N4_1"}.intersection(rules._laws)


def test_newton_gravity_falls_and_n5_floor_contact_prevents_penetration():
    world = WoodshopSimulation()
    identity = "stock.pine-stud.001"
    item = world.items[identity]
    center = item.center_xyz()
    translate_machine_3d(item.sim.machine,
                         (float(center[0]), float(center[1]), 0.55))
    start_z = float(item.center_xyz()[2])

    world.advance(0.05)

    assert item.center_xyz()[2] < start_z
    assert item.linear_momentum_kg_m_s[2] < 0.0
    for _ in range(80):
        world.advance(0.01)
    lo, _hi = item.bounds_xyz()
    assert lo[2] >= -1.0e-10
    assert world.state_table.get(
        "woodshop", "world_rules", "honorary_laws"
    ) == world.world_rules.HONORARY_LAWS


def test_n5_machine_contact_resolves_overlap_and_impulse():
    world = WoodshopSimulation()
    stock = world.items["stock.pine-stud.001"]
    saw = world.items["tool.hand-saw.001"]
    for identity, item in world.items.items():
        if identity not in {"stock.pine-stud.001", "tool.hand-saw.001"}:
            item.custody = "inventory"
    _stock_lo, stock_hi = stock.bounds_xyz()
    saw_lo, saw_hi = saw.bounds_xyz()
    saw_bottom_from_center = float(saw.center_xyz()[2] - saw_lo[2])
    translate_machine_3d(
        saw.sim.machine,
        (0.0, 0.0,
         float(stock_hi[2]) + saw_bottom_from_center - 0.004),
    )
    saw.linear_momentum_kg_m_s = (0.0, 0.0, -saw.mass_kg)

    world.advance(0.001)

    contact = next(contact for contact in world.world_rules.contacts
                   if {contact.a, contact.b} == {
                       "stock.pine-stud.001", "tool.hand-saw.001"
                   })
    assert contact.penetration_m > 0.0
    assert contact.normal_impulse_n_s > 0.0
    stock_lo, stock_hi = stock.bounds_xyz()
    saw_lo, saw_hi = saw.bounds_xyz()
    assert (min(stock_hi[2], saw_hi[2]) - max(stock_lo[2], saw_lo[2])
            <= 1.0e-10)


def test_inventory_machine_is_not_advanced_by_world_gravity():
    world = WoodshopSimulation()
    identity = "stock.pine-stud.001"
    world.pickup(identity, "left")
    before = world.items[identity].center_xyz().copy()

    world.advance(0.1)

    assert world.items[identity].center_xyz() == pytest.approx(before)
    assert world.items[identity].linear_momentum_kg_m_s == (0.0, 0.0, 0.0)


def test_hand_saw_is_two_shapes_one_work_edge_and_a_held_machine_action():
    saw = hand_saw()
    graph = saw.build_graph()

    assert {part.part_role for part in saw.parts} == {"blade", "handle"}
    assert len(saw.work_edges) == 1
    edge = saw.work_edges[0]
    assert edge.tooth_pattern == ToothPattern.CROSSCUT_ALTERNATING_BEVEL.value
    assert edge.thickness_m > 0.0
    assert edge.wave_amplitude_m > 0.0
    assert graph["work_edges"][0]["identity"] == edge.identity
    assert graph["actions"] == [{
        "identity": "tool.hand-saw.001.actions.saw",
        "gesture": "held-primary",
        "operation": "advance-kerf",
        "destination": edge.identity,
        "release_operation": "stop-kerf-work",
    }]
    assert [pose["name"] for pose in graph["interaction_poses"]] == [
        "selected", "used-1", "used-2"
    ]


def test_machine_hand_pose_has_a_generic_selected_fallback():
    pose = Machine("plain", "plain object").interaction_pose("used-2")

    assert isinstance(pose, MachinePose)
    assert pose.name == "selected"
    assert pose.offset_hand_m[2] > 0.0


def test_resin_sawhorse_bracket_is_one_part_with_three_general_sleeve_ports():
    bracket = resin_sawhorse_bracket()
    graph = bracket.build_graph()

    assert len(bracket.parts) == 1
    assert bracket.parts[0].attributes["one_physical_part"] is True
    ports = bracket.parts[0].ports
    assert {port.role for port in ports} == {
        "top-beam", "leg-left", "leg-right"
    }
    assert all(port.compression_sleeve.accepts(
        (STUD_THICKNESS_M, STUD_WIDTH_M)) for port in ports)
    assert all(len(port.fastener_areas) == 2 for port in ports)
    assert all(port.glue_surfaces[0].area_m2 > 0.0 for port in ports)
    rendered_ports = graph["nodes"][0]["ports"]
    assert all(port["compression_sleeve"]["max_penetration_m"] > 0.0
               for port in rendered_ports)
    assert all(port["fastener_areas"] and port["glue_surfaces"]
               for port in rendered_ports)


def test_held_stud_click_initializes_breakable_rigid_sleeve_fit():
    world = WoodshopSimulation()
    stud = "stock.pine-stud.001"
    bracket = "jig.sawhorse-bracket.001"
    world.pickup(stud, "left")

    joint = world.begin_sleeve_fit_action("left", (-0.75, -1.05))

    assert joint is not None
    assert joint.sleeve_object == bracket
    assert joint.penetration_m == pytest.approx(joint.max_penetration_m)
    assert joint.holding_force_n == pytest.approx(850.0 * 0.46)
    assert world.hotbar.held("left") is None
    assert world.items[stud].custody == "world"
    edge = world.object_graph.edges[joint.edge_identity]
    assert edge.kind == "compression-sleeve-fit"
    assert edge.attributes["rigid"] is True
    assert edge.attributes["condense_rigid_membership"] is True
    assert edge.attributes["fastener_areas"]
    assert edge.attributes["glue_surfaces"]
    assert world.support_quality(stud) == pytest.approx(0.70)

    release = np.asarray(joint.release_vector)
    assert not world.compression_sleeves.apply_force(
        joint.identity, release * (joint.holding_force_n * 0.99))
    assert world.compression_sleeves.apply_force(
        joint.identity, release * (joint.holding_force_n * 1.01))
    assert joint.edge_identity not in world.object_graph.edges
    assert joint.released is True


def test_world_snapshot_restores_objects_graph_joints_hands_and_dt_state():
    world = WoodshopSimulation()
    stud = "stock.pine-stud.001"
    world.pickup(stud, "left")
    joint = world.begin_sleeve_fit_action("left", (-0.75, -1.05))
    assert joint is not None
    world.items[stud].linear_momentum_kg_m_s = (1.0, 2.0, 3.0)
    world.state_table.set("woodshop", "test", "marker", {"present": True})
    saved_center = world.items[stud].center_xyz().copy()
    checkpoint = world.snapshot()

    translate_machine_3d(world.items[stud].sim.machine, (9.0, 9.0, 9.0))
    world.items[stud].linear_momentum_kg_m_s = (0.0, 0.0, 0.0)
    world.object_graph.remove_edge(joint.edge_identity)
    world.compression_sleeves.joints.clear()
    world.state_table.set("woodshop", "test", "marker", None)

    world.restore(checkpoint)

    assert world.items[stud].center_xyz() == pytest.approx(saved_center)
    assert world.items[stud].linear_momentum_kg_m_s == (1.0, 2.0, 3.0)
    assert joint.edge_identity in world.object_graph.edges
    assert list(world.compression_sleeves.joints) == [joint.identity]
    assert world.state_table.get("woodshop", "test", "marker") == {
        "present": True
    }
    world.advance(0.001)


def test_world_disk_save_load_round_trip(tmp_path):
    world = WoodshopSimulation()
    saw = "tool.hand-saw.001"
    world.pickup(saw, "right")
    world.items[saw].orientation_deg_xyz = (12.0, 34.0, 56.0)
    world.items[saw].pose_state = "used-2"
    save_path = tmp_path / "woodshop-world.pkl"

    assert world.save_world(save_path) == save_path
    world.items[saw].orientation_deg_xyz = (0.0, 0.0, 0.0)
    world.items[saw].pose_state = "selected"
    world.hotbar.right_slot = None

    assert world.load_world(save_path) == save_path
    assert world.hotbar.held("right") == saw
    assert world.items[saw].orientation_deg_xyz == (12.0, 34.0, 56.0)
    assert world.items[saw].pose_state == "used-2"

def test_compression_sleeve_accepts_same_profile_at_arbitrary_length():
    objects = WorldObjectGraph()
    bracket = resin_sawhorse_bracket()
    short = kiln_dried_pine_2x4x8("stock.short")
    short.parts[0].half_extent_m = (
        0.31, short.parts[0].half_extent_m[1],
        short.parts[0].half_extent_m[2],
    )
    objects.add_node(bracket.identity, bracket)
    objects.add_node(short.identity, short)
    interface = CompressionSleeveInterface(objects)

    assert len(interface.compatible_ports(bracket.identity, short.identity)) == 3


def test_crosscut_is_projected_onto_the_top_mesh_face():
    stock = kiln_dried_pine_2x4x8()
    part = stock.parts[0]
    mesh = part_box_mesh(part)
    projection = project_crosscut(
        stock, (0.22, 0.0), support_quality=1.0,
        attempt_identity="supported",
    )

    top = float(mesh.vertices[:, 2].max())
    assert projection.intended_a[2] == pytest.approx(top)
    assert projection.intended_b[2] == pytest.approx(top)
    assert projection.intended_b[1] - projection.intended_a[1] == pytest.approx(
        STUD_WIDTH_M
    )
    assert projection.actual_a == pytest.approx(projection.intended_a)
    assert projection.actual_b == pytest.approx(projection.intended_b)


def test_two_hands_hold_machine_identities_not_inventory_substitutes():
    world = WoodshopSimulation()
    stock = "stock.pine-stud.001"
    saw = "tool.hand-saw.001"

    stock_slot = world.pickup(stock, "left")
    saw_slot = world.pickup(saw, "right")

    assert world.hotbar.held("left") == stock
    assert world.hotbar.held("right") == saw
    assert world.hotbar.slots == {stock_slot: stock, saw_slot: saw}
    assert isinstance(world.items[stock].sim, MachineSim)
    assert isinstance(world.machine_systems[stock], MachineSystem)


def test_default_try_put_in_hand_uses_free_hand_and_refuses_when_both_are_full():
    world = WoodshopSimulation()

    assert world.try_put_in_hand("stock.pine-stud.001") == ("right", 1)
    assert world.try_put_in_hand("tool.hand-saw.001") == ("left", 2)
    assert world.try_put_in_hand("tool.bar-clamp.001") is None


def test_drop_gimbal_commits_orientation_and_stands_stud_on_narrow_edge():
    world = WoodshopSimulation()
    identity = "stock.pine-stud.001"
    world.pickup(identity, "right")
    world.sync_hands((0.0, 0.0), (1.0, 0.0))
    placement = world.begin_drop("right", (0.8, 0.2))
    assert placement is not None

    world.adjust_drop((90.0, 0.0, 0.0))
    dropped = world.confirm_drop()

    assert dropped == identity
    item = world.items[identity]
    assert item.custody == "world"
    assert item.orientation_deg_xyz[0] == pytest.approx(90.0)
    x0, y0, x1, y1 = item.bounds_xy()
    assert y1 - y0 == pytest.approx(STUD_THICKNESS_M)
    rotation = item.rotation_matrix()
    part = item.sim.machine.parts[0]
    center = item.center_xyz()
    lowest = min(
        (center + rotation @ ((np.asarray(part.position) - center)
                              + np.asarray(part.half_extent_m) * signs))[2]
        for signs in np.asarray([(sx, sy, sz) for sx in (-1, 1)
                                 for sy in (-1, 1) for sz in (-1, 1)])
    )
    assert lowest == pytest.approx(0.0)


def test_held_saw_action_advances_a_kerf_through_honorary_laws_and_dt_state():
    world = WoodshopSimulation()
    world.pickup("stock.pine-stud.001", "left")
    world.pickup("tool.hand-saw.001", "right")
    world.sync_hands((0.0, 0.0), (1.0, 0.0))
    kerf = world.begin_saw_action("right", (0.5, 0.1))

    assert kerf is not None
    assert kerf.support_quality == pytest.approx(0.12)
    assert abs(kerf.longitudinal_error_m if hasattr(kerf, "longitudinal_error_m") else
               kerf.actual_a[0] - kerf.intended_a[0]) > 0.001

    initial_mass = world.items["stock.pine-stud.001"].sim.machine.parts[0].mass_kg
    for _ in range(120):
        world.advance(1.0 / 120.0)

    advanced = world.cutting.state.kerfs[0]
    assert 0.0 < advanced.depth_m < advanced.through_depth_m
    assert advanced.complete is False
    assert "stock.pine-stud.001" in world.items
    chip_identity = world.chip_products[advanced.identity]
    assert world.items[chip_identity].sim.machine.parts[0].mass_kg == pytest.approx(
        advanced.removed_mass_kg
    )
    assert advanced.removed_volume_m3 == pytest.approx(
        advanced.removed_mass_kg / KILN_DRY_PINE_STUD.density_kg_m3)
    assert advanced.removed_mass_kg == pytest.approx(
        advanced.removed_volume_m3 * KILN_DRY_PINE_STUD.density_kg_m3
    )
    assert world.items["stock.pine-stud.001"].sim.machine.parts[0].mass_kg == pytest.approx(
        initial_mass - advanced.removed_mass_kg
    )
    assert advanced.work_j > 0.0
    assert world.state_table.get("woodshop", "cutting", "kerfs")[0].depth_m == pytest.approx(
        advanced.depth_m
    )
    assert world.items["stock.pine-stud.001"].sim.elapsed_s == pytest.approx(1.0)


def test_offhand_fixture_is_far_less_accurate_than_supported_stock():
    stock = kiln_dried_pine_2x4x8()
    held = project_crosscut(
        stock, (0.0, 0.0), support_quality=0.12, attempt_identity="same-attempt"
    )
    supported = project_crosscut(
        stock, (0.0, 0.0), support_quality=1.0, attempt_identity="same-attempt"
    )

    held_error = abs(held.actual_a[0] - held.intended_a[0])
    supported_error = abs(supported.actual_a[0] - supported.intended_a[0])
    assert held_error > 0.001
    assert supported_error == pytest.approx(0.0)


def test_cutting_snapshot_restores_kerf_and_removed_stock_mass():
    world = WoodshopSimulation()
    world.pickup("stock.pine-stud.001", "left")
    world.pickup("tool.hand-saw.001", "right")
    world.sync_hands((0.0, 0.0), (1.0, 0.0))
    world.begin_saw_action("right", (0.5, 0.1))
    checkpoint = world.cutting.snapshot()
    initial_mass = world.items["stock.pine-stud.001"].sim.machine.parts[0].mass_kg

    world.advance(0.1)
    assert world.items["stock.pine-stud.001"].sim.machine.parts[0].mass_kg < initial_mass

    world.cutting.restore(checkpoint)
    assert world.cutting.state.kerfs[0].depth_m == 0.0
    assert world.items["stock.pine-stud.001"].sim.machine.parts[0].mass_kg == pytest.approx(
        initial_mass
    )


def test_sdf_kerf_removes_mass_before_separation_mints_two_solids():
    stock = kiln_dried_pine_2x4x8()
    part = stock.parts[0]
    body = SdfMaterialBody.from_part(
        part, KILN_DRY_PINE_STUD.density_kg_m3, tool_width_m=0.01,
    )
    mass_before = body.mass_kg
    radius = 0.005
    for z in np.arange(STUD_THICKNESS_M, -radius, -radius):
        body.subtract([CapsuleSdf(
            (0.0, -STUD_WIDTH_M / 2.0, float(z)),
            (0.0, STUD_WIDTH_M / 2.0, float(z)), radius,
        )])
    part.mass_kg = body.mass_kg
    _labels, island_count = body.material_islands()
    assert island_count == 2
    chips = chip_batch(
        stock, part, identity="stock.pine-stud.001.chips.kerf-001",
        position=(0.0, 0.0, 0.0), mass_kg=body.removed_mass_kg,
        density_kg_m3=KILN_DRY_PINE_STUD.density_kg_m3,
        transition_identity="kerf.001",
    )
    transition = disincorporate_sdf_body(
        stock, body, transition_identity="kerf.001", chips=chips,
    )

    assert transition.predecessor.identity == stock.identity
    assert len(transition.solid_descendants) == 2
    assert len(transition.material_products) == 1
    assert transition.mass_before_kg == pytest.approx(mass_before)
    assert transition.mass_after_kg == pytest.approx(mass_before)
    assert transition.mass_error_kg == pytest.approx(0.0, abs=1e-12)
    descendant_ids = {machine.identity for machine in transition.solid_descendants}
    assert stock.identity not in descendant_ids
    for child in transition.solid_descendants:
        child_part = child.parts[0]
        assert child_part.identity != part.identity
        assert child_part.attributes["material_form"] == MaterialForm.SOLID.value
        assert child_part.attributes["predecessor_part"] == part.identity
        assert child_part.attributes["sdf_occupied_cells"] > 0
        assert child_part.attributes["sdf_kernel"].occupied.any()
    chips = transition.material_products[0].parts[0]
    assert chips.attributes["material_form"] == MaterialForm.CHIPS.value
    assert chips.mass_kg == pytest.approx(body.removed_mass_kg)
    assert chips.attributes["solid_volume_m3"] == pytest.approx(
        body.removed_volume_m3)
    assert chips.attributes["bulk_volume_m3"] > chips.attributes["solid_volume_m3"]


def test_world_fission_transfers_held_slot_and_publishes_new_identities():
    world = WoodshopSimulation()
    predecessor = "stock.pine-stud.001"
    slot = world.pickup(predecessor, "left")
    world.pickup("tool.hand-saw.001", "right")
    world.sync_hands((0.0, 0.0), (1.0, 0.0))
    kerf = world.begin_saw_action("right", (0.48, 0.0))
    assert kerf is not None

    source_part = world.items[predecessor].sim.machine.parts[0]
    body = world.cutting.material_bodies[source_part.identity]
    radius = kerf.width_m / 2.0
    top = source_part.position[2] + source_part.half_extent_m[2]
    bottom = source_part.position[2] - source_part.half_extent_m[2]
    for z in np.arange(top, bottom - radius, -radius):
        body.subtract([CapsuleSdf(
            (kerf.actual_a[0], kerf.actual_a[1], float(z)),
            (kerf.actual_b[0], kerf.actual_b[1], float(z)), radius,
        )])
    source_part.mass_kg = body.mass_kg
    kerf.depth_m = kerf.through_depth_m
    kerf.removed_volume_m3 = body.removed_volume_m3
    kerf.removed_mass_kg = body.removed_mass_kg
    _labels, island_count = body.material_islands()
    kerf.complete = island_count > 1
    assert kerf.complete
    world._sync_chip_product(kerf)
    transition = world.disincorporate_completed_kerf(kerf)

    retained = transition.solid_descendants[0].identity
    assert predecessor not in world.items
    assert predecessor in world.retired_objects
    assert predecessor in world.object_graph.lineage
    assert world.hotbar.slots[slot] == retained
    assert world.hotbar.held("left") == retained
    assert world.items[retained].custody == "inventory"
    assert world.items[transition.solid_descendants[1].identity].custody == "world"
    chip_identity = transition.material_products[0].identity
    assert world.items[chip_identity].custody == "world"
    assert chip_identity in world.machine_systems
    child_ids = tuple(machine.identity for machine in transition.solid_descendants)
    assert all(identity in world.object_graph.nodes for identity in child_ids)
    assert world.object_graph.edges_between(*child_ids) == ()
    assert world.object_graph.edges == {}
    assert world.object_graph.lineage[predecessor].successors == (
        child_ids + (chip_identity,)
    )
    published = [
        identity for identity in world.state_table.identity_registry.values()
        if identity.get("machine_identity") in transition.predecessor.successors
    ]
    assert len(published) == 3
    retired = [
        identity for identity in world.state_table.identity_registry.values()
        if identity.get("semantic_identity") == source_part.identity
    ]
    assert retired[0]["retired"] is True
    assert retired[0]["mass"] == pytest.approx(0.0)


def test_bar_and_pipe_clamps_are_real_machines_with_deploy_actions():
    for clamp in (bar_clamp(), pipe_clamp()):
        assert {"clamp-frame", "fixed-jaw", "moving-jaw", "spindle",
                "pressure-pad", "handle"} == {
                    part.part_role for part in clamp.parts
                }
        assert len(clamp.lines) == 4
        assert clamp.actions[0].operation == "apply-woodworking-clamp"
        assert clamp.build_graph()["actions"][0]["gesture"] == "auto-deploy"


def test_auto_deploy_uses_clamp_nodes_as_force_path_and_never_joins_members():
    objects = WorldObjectGraph()
    a = kiln_dried_pine_2x4x8("stock.a")
    b = kiln_dried_pine_2x4x8("stock.b")
    clamps = [bar_clamp("clamp.1"), bar_clamp("clamp.2")]
    for machine in (a, b, *clamps):
        objects.add_node(machine.identity, machine)
    interface = WoodworkingJointInterface(objects)
    joint = WoodworkingJoint(
        "joint.test", a.identity, b.identity,
        (-0.25, 0.0, 0.04), (0.25, 0.0, 0.04),
        (0.0, 1.0, 0.0), required_opening_m=0.10,
        required_throat_m=0.05, target_total_force_n=6_000.0,
        max_pad_pressure_pa=5.0e6, max_spacing_m=0.30,
    )

    deployments = interface.auto_deploy(
        joint, [machine.identity for machine in clamps]
    )

    assert len(deployments) == 2
    assert objects.edges_between(a.identity, b.identity) == ()
    assert len(objects.edges) == 4
    for deployment in deployments:
        assert deployment.force_n == pytest.approx(3_000.0)
        assert deployment.pad_pressure_pa <= joint.max_pad_pressure_pa
        assert {objects.edges[edge].kind for edge in deployment.edge_ids} == {
            "clamp-jaw-contact"
        }
        assert all(objects.edges[edge].attributes["temporary"]
                   for edge in deployment.edge_ids)
    interface.release(joint.identity)
    assert objects.edges == {}


def test_auto_deploy_selects_pipe_clamp_when_bar_opening_is_too_short():
    objects = WorldObjectGraph()
    a = kiln_dried_pine_2x4x8("stock.long-a")
    b = kiln_dried_pine_2x4x8("stock.long-b")
    bar = bar_clamp("clamp.bar")
    pipe = pipe_clamp("clamp.pipe")
    for machine in (a, b, bar, pipe):
        objects.add_node(machine.identity, machine)
    joint = WoodworkingJoint(
        "joint.wide", a.identity, b.identity,
        (0.0, 0.0, 0.04), (0.1, 0.0, 0.04), (0.0, 1.0, 0.0),
        required_opening_m=0.80, required_throat_m=0.05,
        target_total_force_n=2_000.0, max_pad_pressure_pa=5.0e6,
    )
    deployed = WoodworkingJointInterface(objects).auto_deploy(
        joint, [bar.identity, pipe.identity]
    )
    assert [item.clamp for item in deployed] == [pipe.identity]
