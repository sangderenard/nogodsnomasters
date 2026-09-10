"""A real mechanical linkage primitive -- translates one moving part's
own position (a throttle pedal, a boost-referenced diaphragm rod,
anything driving something else) into a DRIVEN element's own travel
fraction. Two real linkage shapes come out of one formula:

  ORDINARY  (cam_exponent=1.0, engagement spanning the whole real
            input range): ganged 1:1 to the input -- a plain carb
            primary or a direct-linked single throttle plate.

  PROGRESSIVE (engagement_start_frac > 0 and/or cam_exponent > 1.0):
            the real mechanism behind a progressive secondary
            carburetor linkage -- a cam-profiled follower that doesn't
            begin moving until the primary/pedal has already traveled
            past some real trip point (`engagement_start_frac`), then
            opens through its own real cam shape (`cam_exponent > 1`
            is a real progressive cam: slow initial opening,
            accelerating through the rest of its travel -- the actual
            geometry a period progressive-secondary cam plate gives,
            not an arbitrary "surge" number bolted onto a linear
            curve).

Reused wherever this catalogue needs a real driven-vs-driving
relationship: throttle_body.py's own carburetor barrels are the
concrete first use, but the same primitive is the right one for any
future cam-actuated accessory (a wastegate rod, a progressive brake
proportioning valve, ...).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Linkage:
    engagement_start_frac: float = 0.0   # input fraction where this linkage begins moving
    engagement_end_frac: float = 1.0     # input fraction where it reaches full travel
    cam_exponent: float = 1.0            # 1.0 = ordinary linear; >1.0 = a real progressive cam

    def driven_frac(self, input_frac: float) -> float:
        """0..1 travel of the DRIVEN element for this INPUT fraction."""
        span = max(1e-9, self.engagement_end_frac - self.engagement_start_frac)
        t = max(0.0, min(1.0, (input_frac - self.engagement_start_frac) / span))
        return t ** self.cam_exponent
