"""Real crank-throw geometry, and an evaluate-and-evolve search over
firing order -- replacing the fixed front/back "wave" heuristic that
used to be the whole of firing-order choice.

Throw geometry isn't a free choice: for an even-firing four-stroke
crank with `per_bank` cylinder positions along its length, the throws
MUST sit at 720/per_bank crank-degrees apart -- that's the only
arrangement that fires evenly at all, and every bank shares the same
throw at a given position (offset only by the bank angle, not by
crank phase). What real engine designers actually choose is which
throw fires in which SLOT of the sequence, trading off torsional/
bearing loading between adjacent throws. That's a real combinatorial
optimization problem; this runs a small one.

`score_order` is the evaluation: it penalizes consecutive firing
events whose throws sit close together along the crank (the same
bearing pair getting loaded twice in quick succession, and a
concentrated rather than spread-out torsional pulse train).
`evolve_position_order` is the evolution: hill-climbing mutation
(swap two firing slots) with a small chance to accept a worse move
(so it doesn't just freeze at the first local minimum), run for a
fixed number of generations. Deterministic per seed, so a given
(cylinders, banks) combination always converges to the same order.
"""
from __future__ import annotations

import random


def throw_positions_deg(per_bank: int) -> list[float]:
    return [i * 720.0 / per_bank for i in range(per_bank)]


def crank_distance(a: int, b: int, per_bank: int) -> int:
    d = abs(a - b)
    return min(d, per_bank - d)


def score_order(position_order: list[int], per_bank: int) -> float:
    """Lower is better. Sum, over each consecutive pair of firing
    events, of a penalty that grows sharply as their throws sit closer
    together along the crank."""
    n = len(position_order)
    if n <= 1:
        return 0.0
    penalty = 0.0
    for i in range(n):
        a = position_order[i]
        b = position_order[(i + 1) % n]
        dist = crank_distance(a, b, per_bank)
        penalty += 1.0 / (1.0 + dist) ** 2
    return penalty


def evolve_position_order(per_bank: int, generations: int = 400, seed: int = 0) -> tuple[list[int], float]:
    if per_bank <= 1:
        return list(range(per_bank)), 0.0
    rng = random.Random(seed)
    current = list(range(per_bank))
    rng.shuffle(current)
    current_score = score_order(current, per_bank)
    best, best_score = current[:], current_score

    for _ in range(generations):
        candidate = current[:]
        i, j = rng.sample(range(per_bank), 2)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        cscore = score_order(candidate, per_bank)
        if cscore <= current_score or rng.random() < 0.05:
            current, current_score = candidate, cscore
            if cscore < best_score:
                best, best_score = candidate[:], cscore

    return best, best_score


def generate_radial_firing_order(cylinders_per_row: int) -> list[int]:
    """A radial row's firing order is NOT a search problem -- it's forced
    by the geometry. Every cylinder in a row shares one crank throw, so
    consecutive power strokes (720/N crank-degrees apart, same as any
    four-stroke) land on cylinders 720/N crank-degrees of throw rotation
    apart. Physically the cylinders sit 360/N degrees apart around the
    circle, exactly half that spacing -- which is *why* every real radial
    row has an odd cylinder count: with N odd, stepping by 2 physical
    positions each firing (1, 3, 5, ..., N, 2, 4, ..., N-1) walks every
    cylinder exactly once before repeating, landing each firing at the
    correct 720/N crank-degree interval. An even N can't do this at all
    (stepping by 2 splits into two separate cycles that never join up),
    which is the real mechanical reason single-row radials are always odd.
    """
    n = cylinders_per_row
    if n <= 0:
        return []
    if n % 2 == 0:
        raise ValueError(f"a single radial row needs an odd cylinder count, got {n}")
    order = []
    cyl = 1
    for _ in range(n):
        order.append(cyl)
        cyl = (cyl - 1 + 2) % n + 1
    return order


def generate_multirow_radial_firing_order(cylinders_per_row: int, rows: int) -> list[int]:
    """The firing order of a twin- or four-row radial.

    A multi-row radial is not a bigger single row: each row keeps its own
    forced 1,3,5,... walk (see above, it is geometry, not a search), and
    the rows fire ALTERNATELY so that consecutive power strokes land in
    different rows. That is what keeps the firing impulses spread around
    the crankshaft instead of hammering one throw, and it is why the rows
    are angularly staggered from each other.

    Numbering follows aircraft practice: consecutive cylinder numbers
    alternate rows, so a 14-cylinder twin row has its front row at
    1,3,5,7,9,11,13 and its rear row at 2,4,6,8,10,12,14. A 28-cylinder
    four-row Wasp Major walks all four rows in turn the same way.

    Each row still needs an odd cylinder count for its own walk to close,
    which is why every real multi-row radial is a stack of odd rows --
    14 is two sevens, 18 is two nines, 28 is four sevens.
    """
    per_row = generate_radial_firing_order(cylinders_per_row)
    rows = max(1, int(rows))
    if rows == 1:
        return per_row
    order: list[int] = []
    for step, position in enumerate(per_row):
        for row in range(rows):
            # rotate which row leads each time round so the alternation
            # advances rather than pairing the same two cylinders
            source = per_row[(step + row) % len(per_row)]
            order.append((source - 1) * rows + row + 1)
    return order[:cylinders_per_row * rows]


def generate_firing_order(cylinders: int, banks: int, generations: int = 400, seed: int = 0) -> tuple[list[int], float]:
    """Returns (firing order as cylinder numbers, final evaluation score).
    Cylinder numbering matches engine_geometry.cylinder_sites: cylinder
    k sits at bank (k-1) % banks, position (k-1) // banks."""
    if cylinders <= 0:
        return [], 0.0
    banks = max(1, banks)
    per_bank = -(-cylinders // banks)  # ceil
    position_order, score = evolve_position_order(per_bank, generations, seed)

    firing: list[int] = []
    for pos0 in position_order:
        pos = pos0 + 1
        for bank in range(banks):
            cyl = (pos - 1) * banks + bank + 1
            if cyl <= cylinders:
                firing.append(cyl)
    return firing, score
