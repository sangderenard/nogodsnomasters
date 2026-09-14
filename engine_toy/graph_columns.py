"""THE GRAPH AS COLUMNS, so the hot paths stop walking it edge by edge.

Tokenising the constraints made the identity an integer. On its own that
bought nothing measurable -- CPython interns string literals, so `==` on
a spelling is already a pointer compare, and swapping it for a dict
lookup of an int measured 0.90x, which is to say slightly SLOWER. The
first claim made for the tokens was wrong and this module is the reason
they were worth doing anyway:

    membership over 11 kinds, 1739 edges, 200 sweeps
        per-edge loop over spellings        72 ms
        per-edge loop over tokens           79 ms      0.90x
        one gather on an int32 column        7 ms      10x

A string column cannot be a numpy array. An integer column is one, and
then every question of the form "which members are X" is a gather and a
mask rather than a loop -- and the same column is the only form that can
reach the native lane at all, where a Python string cannot go.

So the point was never the comparison. It was that the comparison stops
happening one edge at a time.

The columns are built once per document and cached on it, because the
document is the thing that changes and rebuilding them per query would
put the loop back one level up.
"""
from __future__ import annotations

import numpy as np

from joints import (MEMBER_CONSTRAINTS, ROUTED_TOKENS, SLIDING_TOKENS,
                    PINNED_TOKENS, SPRING_TOKENS, ZERO_LENGTH_TOKENS,
                    _TOKENS)


_CACHE_KEY = "_graph_columns"


def _token_of(edge) -> int:
    tok = edge.get("constraint_token")
    if tok is not None:
        return int(tok)
    mc = MEMBER_CONSTRAINTS.get(edge.get("constraint"))
    return mc.token if mc else -1


def columns(document: dict, *, rebuild: bool = False) -> dict:
    """Every per-edge scalar the solvers ask about, as arrays.

    Cached on the document. `rebuild=True` after mutating edges."""
    cached = document.get(_CACHE_KEY)
    if cached is not None and not rebuild \
            and cached["n"] == len(document["edges"]):
        return cached
    edges = document["edges"]
    n = len(edges)
    cols = {
        "n": n,
        "token": np.fromiter((_token_of(e) for e in edges), np.int32, n),
        "radius": np.fromiter((float(e.get("radius", 0.0)) for e in edges),
                              np.float64, n),
        "length": np.fromiter((float(e.get("rest_length", 0.0)) for e in edges),
                              np.float64, n),
        "rigid": np.fromiter((bool(e.get("rigid")) for e in edges), bool, n),
        "in_frame": np.fromiter(
            (bool((e.get("damage") or {}).get("model")) for e in edges),
            bool, n),
        "beam_solvable": np.fromiter(
            (bool(e.get("beam_solvable", True)) for e in edges),
            bool, n),
        "participates": np.fromiter(
            (e.get("structural_participation", True)
             not in (False, "nonstructural") for e in edges), bool, n),
        "lock_engaged": np.fromiter(
            (e.get("lock_engaged", True) is not False for e in edges),
            bool, n),
    }
    # MASKS OVER THE TOKEN SPACE, not over the edges. One bool per
    # declared constraint -- forty of them -- and then `mask[token]` is
    # a gather that answers the question for every edge at once.
    ntok = len(_TOKENS)
    for name, tokens in (("routed", ROUTED_TOKENS), ("sliding", SLIDING_TOKENS),
                         ("pinned", PINNED_TOKENS), ("spring", SPRING_TOKENS),
                         ("zero_length", ZERO_LENGTH_TOKENS)):
        m = np.zeros(ntok + 1, dtype=bool)
        if tokens:
            m[list(tokens)] = True
        cols[f"is_{name}"] = m[cols["token"]]
        cols[f"{name}_mask"] = m
    # slenderness, which decides what is a beam at all, for every member
    # in one expression instead of a call per edge
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["slenderness"] = np.where(cols["radius"] > 0.0,
                                       cols["length"] / (2.0 * cols["radius"]),
                                       0.0)
    document[_CACHE_KEY] = cols
    return cols


def of_kind(document: dict, token: int) -> np.ndarray:
    """Indices of every member of one declared kind."""
    return np.flatnonzero(columns(document)["token"] == int(token))


def structural(document: dict) -> np.ndarray:
    """Indices of members the frame solve should assemble: not routed,
    and carrying a damage model.

    ``beam_solvable=False`` is historically also used for exact rigid
    kinematic ties, so it cannot by itself mean "remove this edge". The node
    participation/condensation pass below is the unambiguous boundary.
    """
    c = columns(document)
    # Unlike the legacy ``beam_solvable`` field, participation is used only
    # for solver ownership.  It gives generators a precise way to retain a
    # render/ownership edge without inventing a beam.  All fields are already
    # columns, so do not return to a per-candidate Python loop here.
    return np.flatnonzero(~c["is_routed"] & c["in_frame"]
                          & c["participates"] & c["lock_engaged"])


def structural_participation_report(document: dict) -> dict:
    """Explain beam, rigid-condensed and nonstructural election separately."""
    c = columns(document)
    candidate = (~c["is_routed"] & c["in_frame"]
                 & c["participates"] & c["lock_engaged"])
    nodes = document["nodes"]
    master = {node["identity"]: str(
        node.get("solver_condensed_into") or node["identity"])
              for node in nodes}
    nonstructural_nodes = {
        node["identity"] for node in nodes
        if (not node.get("solver_condensed_into")
            and node.get("structural_participation")
            in (False, "nonstructural"))}
    rigid_condensed = np.fromiter(
        (bool(candidate[i]
              and master[edge["a"]] == master[edge["b"]])
         for i, edge in enumerate(document["edges"])), bool, c["n"])
    excluded_by_node = np.fromiter(
        (bool(candidate[i]
              and (edge["a"] in nonstructural_nodes
                   or edge["b"] in nonstructural_nodes))
         for i, edge in enumerate(document["edges"])), bool, c["n"])
    deformable = candidate & ~rigid_condensed & ~excluded_by_node
    return {
        "total_edges": int(c["n"]),
        "deformable_beam_edges": int(np.count_nonzero(deformable)),
        "rigid_condensed_edges": int(np.count_nonzero(rigid_condensed)),
        "rigid_condensed_nodes": sum(
            bool(node.get("solver_condensed_into")) for node in nodes),
        "routed_edges": int(np.count_nonzero(c["is_routed"])),
        "without_material_law": int(np.count_nonzero(
            ~c["is_routed"] & ~c["in_frame"])),
        "explicitly_nonstructural": int(np.count_nonzero(
            ~c["is_routed"] & c["in_frame"] & ~c["participates"])),
        "disengaged_locks": int(np.count_nonzero(
            ~c["is_routed"] & c["in_frame"] & c["participates"]
            & ~c["lock_engaged"])),
        "excluded_by_nonstructural_node": int(np.count_nonzero(
            excluded_by_node)),
        "deformable_selection_mask": deformable,
        "rigid_condensed_mask": rigid_condensed,
    }


def structural_nodes(document: dict) -> np.ndarray:
    """Boolean node mask for the frame/beam solve.

    Render and machine-state graphs legitimately contain ports, passages and
    baked subobjects that are not independent structural bodies.  Automatic
    participation requires incidence on a real frame member; explicit false
    and ``solver_condensed_into`` always remove the subobject's own DOFs.
    An edge incident on a condensed port activates its gestalt master: the
    point still carries load, it merely owns no independent six freedoms.
    """
    nodes = document["nodes"]
    index = {node["identity"]: i for i, node in enumerate(nodes)}
    active = np.zeros(len(nodes), dtype=bool)
    explicitly_out = {
        node["identity"] for node in nodes
        if node.get("solver_condensed_into")
        or node.get("structural_participation") in (False, "nonstructural")}

    def master(identity: str) -> str:
        node = nodes[index[identity]]
        return str(node.get("solver_condensed_into") or identity)

    def endpoints(edge: dict) -> tuple[str, str] | None:
        resolved = []
        for endpoint in (edge["a"], edge["b"]):
            node = nodes[index[endpoint]]
            if (not node.get("solver_condensed_into")
                    and node.get("structural_participation")
                    in (False, "nonstructural")):
                return None
            resolved.append(master(endpoint))
        return resolved[0], resolved[1]

    for edge_i in structural(document):
        edge = document["edges"][int(edge_i)]
        resolved = endpoints(edge)
        if resolved is None:
            continue
        for target in resolved:
            if target not in explicitly_out:
                active[index[target]] = True
    c = columns(document)
    for edge_i, edge in enumerate(document["edges"]):
        if (c["is_spring"][edge_i]
                or edge.get("kind") in {"bump-stop", "magnetorheological"}):
            resolved = endpoints(edge)
            if resolved is None:
                continue
            for target in resolved:
                if target not in explicitly_out:
                    active[index[target]] = True
    for i, node in enumerate(nodes):
        declared = node.get("structural_participation", "auto")
        if node.get("solver_condensed_into"):
            active[i] = False
        elif declared is True or declared == "structural":
            active[i] = True
        elif declared is False or declared == "nonstructural":
            active[i] = False
        if node.get("fixed_to"):
            active[i] = True
        node["solver_structural_participation"] = bool(active[i])
    return active


def beam_candidates(document: dict, min_slenderness: float = 6.0) -> np.ndarray:
    """Indices of members slender enough, and real enough, to ring."""
    c = columns(document)
    return np.flatnonzero(~c["is_routed"] & ~c["rigid"]
                          & (c["slenderness"] >= min_slenderness))
