"""Discrete throttle bodies/carburetor barrels as real objects, each
with its own real bore and its own real Linkage (linkage.py) to the
pedal -- replacing engine_cycle_sim.py's single-plate abstraction (one
effective bore, an ad-hoc "secondary surge gain" kick past a threshold)
with the actual real hardware for engines whose build genuinely has
more than one throttle plate moving on more than one real schedule: a
progressive double four-barrel, a set of individual throttle bodies on
an ITB (individual-throttle-body) intake, whatever the real build is.

engine.throttle_body is None by default -- every catalogue engine that
doesn't declare one keeps using the existing single-plate model
exactly as before (this is additive, not a replacement of the whole
catalogue's throttle response).

Real geometry, reused: the exact same butterfly-plate projected-open-
area relation engine_cycle_sim.throttle_plate_open_frac already uses
for the single-plate case (area_frac = (1-cos(angle))/(1-cos(max)),
the real geometric relation for a disc pivoting across a round bore),
applied per-barrel here instead of once for the whole engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from linkage import Linkage

BUTTERFLY_MAX_ANGLE_DEG = 78.0   # same real max-open angle engine_cycle_sim's single-plate model uses
_BUTTERFLY_MAX_ANGLE_RAD = math.radians(BUTTERFLY_MAX_ANGLE_DEG)
_BUTTERFLY_MAX_AREA_FRAC = 1.0 - math.cos(_BUTTERFLY_MAX_ANGLE_RAD)


@dataclass(frozen=True)
class ThrottleBarrel:
    """One real throttle bore -- a carburetor barrel or a discrete
    electronic throttle body."""
    bore_diameter_mm: float
    linkage: Linkage

    @property
    def bore_area_m2(self) -> float:
        r_m = self.bore_diameter_mm / 2000.0
        return math.pi * r_m * r_m

    def open_area_frac(self, pedal_frac: float) -> float:
        driven = self.linkage.driven_frac(pedal_frac)
        angle_rad = driven * _BUTTERFLY_MAX_ANGLE_RAD
        return (1.0 - math.cos(angle_rad)) / _BUTTERFLY_MAX_AREA_FRAC


@dataclass(frozen=True)
class ThrottleBodyAssembly:
    """One or more real ThrottleBarrels. The assembly's own effective
    open-area fraction is the real flow-area-weighted sum of every
    barrel's own open area over the assembly's total bore area -- a
    progressive secondary genuinely contributes exactly its own bore's
    real share of total capacity once its linkage starts moving, not
    an arbitrary gain number."""
    barrels: tuple[ThrottleBarrel, ...]

    @property
    def total_bore_area_m2(self) -> float:
        return sum(b.bore_area_m2 for b in self.barrels)

    def open_area_frac(self, pedal_frac: float) -> float:
        total = self.total_bore_area_m2
        if total <= 0.0:
            return 0.0
        return sum(b.bore_area_m2 * b.open_area_frac(pedal_frac) for b in self.barrels) / total

    def barrel_open_fracs(self, pedal_frac: float) -> list[float]:
        return [b.open_area_frac(pedal_frac) for b in self.barrels]


def progressive_double_four_barrel(primary_bore_mm: float = 44.0, secondary_bore_mm: float = 44.0,
                                   primary_end_frac: float = 0.65,
                                   secondary_start_frac: float = 0.55,
                                   secondary_cam_exponent: float = 1.8) -> ThrottleBodyAssembly:
    """Two real 4-barrel carburetors on a real dual-quad/tunnel-ram
    intake -- 8 real bores total, the actual hardware behind
    "progressive double four barrels": both carbs' PRIMARY pairs are
    ganged directly to the pedal (a real progressive dual-quad linkage
    bar ties both throttle shafts together, ordinary 1:1 response) and
    stop just past 2/3 pedal travel (a real primary throat is sized to
    cover idle through most of the driving range on its own); both
    carbs' SECONDARY pairs share one real progressive-cam linkage that
    doesn't begin opening until the primaries are most of the way open
    (secondary_start_frac), then opens through a real progressive cam
    profile (secondary_cam_exponent > 1: slow initial crack, then
    accelerating) rather than snapping open -- exactly the real reason
    a progressive secondary linkage exists (a sudden full-area dump of
    cold, unatomized secondary barrels right at tip-in would bog a real
    engine instead of adding power)."""
    primary = Linkage(engagement_start_frac=0.0, engagement_end_frac=primary_end_frac, cam_exponent=1.0)
    secondary = Linkage(engagement_start_frac=secondary_start_frac, engagement_end_frac=1.0,
                        cam_exponent=secondary_cam_exponent)
    barrels: list[ThrottleBarrel] = []
    for _carb in range(2):
        barrels.extend(ThrottleBarrel(bore_diameter_mm=primary_bore_mm, linkage=primary) for _ in range(2))
        barrels.extend(ThrottleBarrel(bore_diameter_mm=secondary_bore_mm, linkage=secondary) for _ in range(2))
    return ThrottleBodyAssembly(barrels=tuple(barrels))
