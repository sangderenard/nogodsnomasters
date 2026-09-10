"""Real, disclosed manufacturing deviation applied to any declared spec
dataclass -- the difference between "as designed" and "as machined."

Not a fudge factor and not a per-engine special case: every deviation
is a real quantity against a real declared field (a bore reamed 0.3mm
oversize, a piston 8% heavy from a bad casting, a pinion radius off by
half a millimeter from a misaligned bore) using the field's own real
units, and the SAME downstream derivation every spec already has
(AtmosphericEngineSpec.natural_cycle_estimate, TurbineSpec.build,
GovernorSpec.build, ...) runs completely unchanged against the result.
A machined engine isn't a different code path -- it's a different, but
equally real, instance of the exact same spec class. This is why
`dataclasses.replace` is enough: nothing here needs to know what an
AtmosphericEngineSpec or a TurbineSpec actually means physically, only
that it's a frozen dataclass with named real fields.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class Deviation:
    """One real, disclosed manufacturing deviation on one declared
    field. `fraction` is a real multiplicative error (1.08 = 8% heavy
    from a bad casting, 0.97 = 3% under from wear-limit stock removal);
    `absolute` is a real additive offset in the field's own units (e.g.
    +0.0003 m of bore from a reboring mistake). Applied as
    value*fraction + absolute, so both compose in the same call.
    `note` records the real, disclosed cause -- required in spirit
    (not enforced), matching this toy's convention of never hiding an
    unexplained number."""
    field: str
    fraction: float = 1.0
    absolute: float = 0.0
    note: str = ""


def machine(spec, deviations: tuple[Deviation, ...]):
    """Returns a new spec instance with the given real deviations
    applied -- the "as machined" version of `spec`. Every field not
    named in `deviations` is copied unchanged. Raises if a deviation
    names a field the spec doesn't actually have, rather than silently
    creating a new attribute -- a real machinist can't modify a
    dimension the part doesn't have either."""
    changes: dict[str, float] = {}
    for d in deviations:
        if not hasattr(spec, d.field):
            raise ValueError(f"{type(spec).__name__} has no field {d.field!r}")
        current = changes.get(d.field, getattr(spec, d.field))
        if not isinstance(current, (int, float)):
            raise ValueError(f"{type(spec).__name__}.{d.field} is not a real-valued "
                              f"dimension (got {type(current).__name__}) -- can't machine it")
        changes[d.field] = current * d.fraction + d.absolute
    return dataclasses.replace(spec, **changes)
