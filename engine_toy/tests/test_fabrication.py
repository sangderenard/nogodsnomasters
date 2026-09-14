import math
import unittest

import numpy as np

from fabrication import (WrapProfile, emit_shaped_iron_strap,
                         outside_perimeter, weld_touching)
from frame_solver import FrameSolver
from turret_production import ProductionGraph
from vehicle_mesh import build_drivetrain_mesh


class FabricationTests(unittest.TestCase):
    def _parents(self):
        graph = ProductionGraph("fabrication-witness")
        graph.node("tube", (0.0, 0.15, 0.0), "load-bearing-structure",
                   mass_kg=20.0, half_extent_m=(0.1, 0.1, 0.5))
        graph.node("spine", (0.0, -0.12, 0.0), "load-bearing-structure",
                   mass_kg=30.0, half_extent_m=(0.12, 0.12, 0.5))
        return graph

    def test_outside_perimeter_skips_the_interior_cleft(self):
        hull = outside_perimeter(
            (WrapProfile("tube", "circle", (0.0, 0.15), radius_m=0.10),
             WrapProfile("spine", "box", (0.0, -0.12),
                         half_extent_m=(0.12, 0.12))),
            clearance_m=0.004, facet_angle_deg=22.5)
        self.assertGreaterEqual(len(hull), 8)
        points = [p.point for p in hull]
        turns = []
        for i in range(len(points)):
            a = points[(i + 1) % len(points)] - points[i]
            b = points[(i + 2) % len(points)] - points[(i + 1) % len(points)]
            turns.append(a[0] * b[1] - a[1] * b[0])
        self.assertTrue(all(float(turn) > 0.0 for turn in turns))

    def test_strap_is_two_halves_with_two_tensioners_and_surface_contacts(self):
        graph = self._parents()
        result = emit_shaped_iron_strap(
            graph, "strap", (
                WrapProfile("tube", "circle", (0.0, 0.15), radius_m=0.10),
                WrapProfile("spine", "box", (0.0, -0.12),
                            half_extent_m=(0.12, 0.12))),
            station_m=0.0, set_torque_nm=160.0)
        screws = [e for e in graph.edges
                  if e.get("part_role") == "strap-tensioner"]
        contacts = [e for e in graph.edges
                    if e.get("part_role") == "preloaded-strap-contact"]
        self.assertEqual(len(screws), 2)
        self.assertEqual(result["closures"], 2)
        self.assertTrue(all(screw["constraint"] == "strap-tensioner"
                            for screw in screws))
        self.assertTrue(all(screw["preload_force_n"] > 0.0
                            for screw in screws))
        self.assertTrue(all(math.isclose(screw["set_torque_nm"], 160.0)
                            for screw in screws))
        self.assertTrue(all(screw["adjustable"] and screw["removable"]
                            for screw in screws))
        self.assertEqual(len(contacts), result["contact_count"])
        parents = {node.get("solver_condensed_into") for node in graph.nodes
                   if node.get("part_role") == "strap-contact-surface"}
        self.assertEqual(parents, {"tube", "spine"})
        self.assertFalse(any(p.startswith("strap.") and
                             "connection(s) -- underconstrained" in p
                             for p in graph.check()))
        names = {name for _v, _n, name in build_drivetrain_mesh(
            graph.as_document())}
        self.assertTrue(any("strap_a_segment" in name for name in names))

    def test_torqued_strap_closure_has_no_free_lug_mode(self):
        graph = self._parents()
        for node in graph.nodes:
            node["fixed_to"] = "world"
        emit_shaped_iron_strap(
            graph, "strap", (
                WrapProfile("tube", "circle", (0.0, 0.15), radius_m=0.10),
                WrapProfile("spine", "box", (0.0, -0.12),
                            half_extent_m=(0.12, 0.12))),
            station_m=0.0, width_m=.100, thickness_m=.020,
            bolt="M20", set_torque_nm=430.0)

        solver = FrameSolver(graph.as_document(), {})
        self.assertEqual(solver.modes(12)["mechanism_count"], 0)
        self.assertLess(solver.solve()["max_deflection_m"], 0.001)

    def test_production_spine_straps_use_heavy_forging_scale_section(self):
        from turret_production import build_gimbal_cannon_station

        graph = build_gimbal_cannon_station()
        straps = [edge for edge in graph.edges
                  if edge.get("part_role") == "shaped-iron-strap"]
        self.assertTrue(straps)
        self.assertTrue(all(edge["section_width_m"] == 0.100
                            and edge["section_thickness_m"] == 0.020
                            for edge in straps))
        passages = [edge for edge in graph.edges
                    if edge.get("part_role") == "evacuator-port"]
        self.assertTrue(passages)
        self.assertTrue(all(edge["constraint"] == "exhaust-flow-path"
                            and "damage" not in edge for edge in passages))
        self.assertTrue(all(edge["cleanliness_fraction"] == 1.0
                            and edge["fouling_mass_g"] == 0.0
                            and edge["occluded_area_fraction"] == 0.0
                            for edge in passages))

    def test_weld_touching_emits_filler_between_parent_surface_toes(self):
        graph = self._parents()
        made = weld_touching(
            graph, "seam", "tube", "spine",
            [((0.0, 0.050, 0.0), (0.0, 0.044, 0.0))], throat_m=0.004)
        self.assertEqual(made, ["seam.0"])
        edge = graph.edges[-1]
        self.assertEqual(edge["constraint"], "welded-joint-element")
        self.assertEqual(edge["parent_bodies"], ("tube", "spine"))
        self.assertEqual(edge["damage"]["model"],
                         "elastic-plastic-member-with-shear-fracture")
        self.assertLess(edge["rest_length"], edge["damage"]["geometric_span_m"])
        with self.assertRaisesRegex(ValueError, "not touching"):
            weld_touching(graph, "remote", "tube", "spine",
                          [((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))],
                          throat_m=0.004)


if __name__ == "__main__":
    unittest.main()
