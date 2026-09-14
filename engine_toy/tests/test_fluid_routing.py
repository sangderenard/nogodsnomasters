import math

import numpy as np

from fluid_routing import (FLEXIBLE_HOSE, RIGID_ORTHOGONAL, TRIVIAL,
                           annotate_fluid_routes, orthogonal_route)
from drivetrain_graph import _line_length_m
from vehicle_mesh import build_drivetrain_mesh


def _node(identity, position, half=(.08, .08, .08)):
    return {"identity": identity, "reference_position": list(position),
            "body_half_extent_m": list(half), "kind": "machine",
            "in_view": True}


def _edge(identity, a, b, route_class):
    return {"identity": identity, "a": a, "b": b,
            "constraint": "coolant-line", "circuit_identity": "coolant",
            "radius": .01, "fluid_route_class": route_class}


def test_rigid_line_is_manhattan_and_detours_around_body():
    graph = {
        "nodes": [_node("a", (0, 0, 0)), _node("b", (2, 0, 0)),
                  _node("block", (1, 0, 0), (.25, .25, .25))],
        "edges": [_edge("line", "a", "b", RIGID_ORTHOGONAL)]}
    annotate_fluid_routes(graph, clearance_m=.05)
    edge = graph["edges"][0]
    points = [np.array((0., 0., 0.)),
              *(np.array(p) for p in edge["waypoints"]),
              np.array((2., 0., 0.))]
    assert edge["route_fittings"]["interior_elbows_90_deg"] >= 2
    assert edge["route_length_m"] > 2.0
    for p, q in zip(points, points[1:]):
        assert np.count_nonzero(np.abs(q - p) > 1e-9) == 1
    # The detour clears the obstacle plus the requested clearance.
    assert any(abs(p[1]) >= .30 or abs(p[2]) >= .30 for p in points)


def test_flexible_hose_has_declared_slack_and_mesh_follows_it():
    graph = {"nodes": [_node("a", (0, 0, 0)), _node("b", (1, 0, 0))],
             "edges": [_edge("hose", "a", "b", FLEXIBLE_HOSE)]}
    graph["edges"][0]["hose_slack_fraction"] = .12
    annotate_fluid_routes(graph)
    edge = graph["edges"][0]
    assert len(edge["waypoints"]) >= 2
    assert math.isclose(edge["route_length_m"], 1.12, rel_tol=2e-3)
    vertices, _normals, _name = build_drivetrain_mesh(graph)[0]
    assert vertices[:, 1].min() < -.01


def test_trivial_connection_reports_manhattan_length_and_is_thin():
    graph = {"nodes": [_node("a", (0, 0, 0)), _node("b", (1, 2, 3))],
             "edges": [_edge("concept", "a", "b", TRIVIAL)]}
    annotate_fluid_routes(graph)
    edge = graph["edges"][0]
    assert edge["waypoints"] == []
    assert edge["route_length_m"] == 6.0
    assert edge["route_visual"] == "conceptual-thin"
    assert _line_length_m(edge, {n["identity"]: n for n in graph["nodes"]}) == 6.0


def test_three_fluid_edges_at_port_declare_tee():
    graph = {"nodes": [_node("m", (0, 0, 0)), _node("a", (1, 0, 0)),
                             _node("b", (0, 1, 0)), _node("c", (0, 0, 1))],
             "edges": [_edge("ma", "m", "a", RIGID_ORTHOGONAL),
                       _edge("mb", "m", "b", RIGID_ORTHOGONAL),
                       _edge("mc", "m", "c", RIGID_ORTHOGONAL)]}
    annotate_fluid_routes(graph)
    assert all(e["route_fittings"]["a"] == "tee" for e in graph["edges"])


def test_legacy_undeclared_hardline_is_marked_for_obstacle_audit():
    edge = {"identity": "old", "a": "a", "b": "b",
            "constraint": "oil-line", "circuit_identity": "oil"}
    graph = {"nodes": [_node("a", (0, 0, 0)), _node("b", (1, 1, 0))],
             "edges": [edge]}
    annotate_fluid_routes(graph)
    assert edge["fluid_route_class"] == RIGID_ORTHOGONAL
    assert edge["route_obstacle_aware"] is False
    assert edge["route_requires_obstacle_audit"] is True
