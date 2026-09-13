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
    and carrying a damage model."""
    c = columns(document)
    return np.flatnonzero(~c["is_routed"] & c["in_frame"])


def beam_candidates(document: dict, min_slenderness: float = 6.0) -> np.ndarray:
    """Indices of members slender enough, and real enough, to ring."""
    c = columns(document)
    return np.flatnonzero(~c["is_routed"] & ~c["rigid"]
                          & (c["slenderness"] >= min_slenderness))
