"""Dissipative angular impulses for bearings, clutches and rolling contacts."""
from __future__ import annotations

import math


def friction_impulse(slip_rad_s: float, torque_limit_nm: float,
                     inverse_inertia_sum: float, dt: float) -> tuple[float, float]:
    """Return signed impulse and heat, stopping at equal shaft speeds.

    An externally driven shaft contributes zero inverse inertia. The heat is
    the work over the impulse's mean slip, including the finite inertia term;
    instantaneous torque times initial slip overstates it near sticking.
    """
    if dt <= 0.0 or slip_rad_s == 0.0:
        return 0.0, 0.0
    magnitude = max(0.0, torque_limit_nm) * dt
    if inverse_inertia_sum > 0.0:
        magnitude = min(magnitude, abs(slip_rad_s) / inverse_inertia_sum)
    impulse = math.copysign(magnitude, slip_rad_s)
    remaining_slip = max(0.0, abs(slip_rad_s) - magnitude * inverse_inertia_sum)
    heat_j = magnitude * (0.5 * abs(slip_rad_s) + 0.5 * remaining_slip)
    return impulse, heat_j
