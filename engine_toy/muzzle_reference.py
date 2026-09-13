"""Where the muzzle is actually pointing, which is not where the sight is.

THE BARREL IS NOT STRAIGHT AND IT DOES NOT STAY THE SAME SHAPE. A gun
is aimed by moving its trunnion, and the shot leaves from the muzzle
five metres away at the end of a steel tube that bends. Two separate
things bend it and they need completely different answers:

  GRAVITY DROOP is large and predictable. A 5.28 m outer barrel droops
  14 mm at the muzzle, which is 3.6 mrad -- nearly three metres of miss
  at 800 m. But it is a KNOWN function of elevation: at the horizontal
  the whole weight bends the tube, straight up it bends none of it, and
  in between it goes with the cosine. So it is boresighted out and then
  corrected by angle, and it never needs measuring in the field.

  THERMAL BEND is small, unpredictable, and it moves DURING THE FIGHT.
  Sun on one side, rain on the other, or -- worst -- the heat of your
  own firing soaking unevenly into a tube that is also being cooled by
  a water jacket on the top half. A 40 K difference top to bottom
  curves this tube like a bimetallic strip, and nothing about it can be
  predicted from a table. It has to be MEASURED.

THAT IS WHAT A MUZZLE REFERENCE SYSTEM IS. A mirror on the muzzle, a
collimator at the sight, and a light path between them: the sight looks
at its own reflection and sees exactly how far the muzzle has wandered
off the line it was boresighted to. Every real tank gun of any
seriousness has one, and this mount needs it more than most because the
water jacket makes the thermal gradient WORSE -- a cooled top and a hot
bottom is precisely the bimetal case.

AND IT IS ALSO THE HONEST ANSWER TO "WHY DID THAT MISS". A shot that
lands long when the tube is hot is not a fire-control failure or a bad
charge; it is a barrel that was pointing somewhere else. Without an MRS
you cannot tell those apart, and you will spend the afternoon adjusting
the wrong thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

STEEL_ALPHA_PER_K = 12.0e-6


@dataclass
class BarrelBend:
    """The tube's shape, from its own geometry and its own temperature."""
    outer_diameter_m: float = 0.164
    wall_m: float = 0.022
    length_m: float = 5.28
    youngs_pa: float = 2.05e11
    density_kg_m3: float = 7850.0
    #: how much of the tube's weight is carried forward of the trunnion
    overhang_m: float = None

    def __post_init__(self) -> None:
        if self.overhang_m is None:
            self.overhang_m = self.length_m

    @property
    def second_moment_m4(self) -> float:
        d, t = self.outer_diameter_m, self.wall_m
        return math.pi / 64.0 * (d ** 4 - (d - 2 * t) ** 4)

    @property
    def area_m2(self) -> float:
        d, t = self.outer_diameter_m, self.wall_m
        return math.pi / 4.0 * (d ** 2 - (d - 2 * t) ** 2)

    @property
    def mass_kg(self) -> float:
        return self.area_m2 * self.density_kg_m3 * self.length_m

    # ------------------------------------------------------------------
    def gravity_droop_rad(self, elevation_deg: float = 0.0) -> float:
        """Muzzle pointing error from the tube's own weight.

        A uniformly loaded cantilever: the SLOPE at the tip is what
        aims the shot, not the deflection. And only the component of
        gravity perpendicular to the bore bends it, so this goes to
        zero pointing straight up -- which is why droop correction is a
        function of elevation and not a constant."""
        w = (self.area_m2 * self.density_kg_m3 * 9.81
             * math.cos(math.radians(elevation_deg)))
        return (w * self.overhang_m ** 3
                / (6.0 * self.youngs_pa * self.second_moment_m4))

    def thermal_bend_rad(self, delta_t_k: float) -> float:
        """Bend from a temperature difference across the tube.

        The hot side is longer than the cold side, so the tube curves
        like a bimetallic strip: curvature is alpha dT / diameter, and
        the muzzle's pointing error is that curvature times the length.
        """
        curvature = STEEL_ALPHA_PER_K * delta_t_k / self.outer_diameter_m
        return curvature * self.overhang_m


@dataclass
class MuzzleReferenceSystem:
    """A mirror at the muzzle and a collimator at the sight.

    It measures the thing that cannot be predicted, and it measures it
    as an ANGLE relative to the boresight -- which is the quantity fire
    control actually needs, rather than a temperature it would have to
    convert."""
    identity: str = "turret.mrs"
    barrel: BarrelBend = field(default_factory=BarrelBend)
    #: the instrument's own limits: it is a real optical measurement
    resolution_rad: float = 20.0e-6          # 20 urad, a good collimator
    noise_rad: float = 15.0e-6
    #: how many readings it averages before it will call a figure good
    settle_readings: int = 8
    #: boresight: what the sight was told "straight" is
    boresight_rad: tuple = (0.0, 0.0)
    #: live
    last_reading_rad: tuple = (0.0, 0.0)
    readings: int = 0
    valid: bool = False
    obscured: bool = False

    def true_bend_rad(self, elevation_deg: float, delta_t_k: float,
                      cross_delta_t_k: float = 0.0) -> tuple:
        """What the muzzle is really doing. The sim knows; the crew does
        not, which is the entire reason the instrument exists."""
        return (self.barrel.thermal_bend_rad(cross_delta_t_k),
                -self.barrel.gravity_droop_rad(elevation_deg)
                + self.barrel.thermal_bend_rad(delta_t_k))

    def read(self, elevation_deg: float, delta_t_k: float,
             cross_delta_t_k: float = 0.0, *, rng=None) -> dict:
        """One look down the collimator."""
        import random
        r = rng or random
        if self.obscured:
            self.valid = False
            self.readings = 0
            return {"valid": False, "why": "muzzle mirror obscured -- "
                                           "smoke, mud, or a hit"}
        true = self.true_bend_rad(elevation_deg, delta_t_k, cross_delta_t_k)
        meas = tuple(
            round((t + r.gauss(0.0, self.noise_rad)) / self.resolution_rad)
            * self.resolution_rad - b
            for t, b in zip(true, self.boresight_rad))
        self.last_reading_rad = meas
        self.readings += 1
        self.valid = self.readings >= self.settle_readings
        return {"valid": self.valid, "bend_rad": meas,
                "readings": self.readings,
                "correction_rad": tuple(-m for m in meas)}

    def boresight(self, elevation_deg: float = 0.0) -> None:
        """Zero it against a cold, level tube. Everything measured after
        this is DEPARTURE from that, which is what can be corrected."""
        self.boresight_rad = self.true_bend_rad(elevation_deg, 0.0, 0.0)
        self.readings = 0
        self.valid = False

    def describe(self) -> list[str]:
        b = self.barrel
        return [
            f"{self.identity}: mirror at the muzzle, collimator at the sight",
            f"  tube {b.length_m:.2f} m, {b.outer_diameter_m * 1000:.0f} mm OD, "
            f"{b.mass_kg:.0f} kg, I = {b.second_moment_m4 * 1e8:.0f} cm4",
            f"  gravity droop level {b.gravity_droop_rad(0.0) * 1000:.2f} mrad, "
            f"at 45 deg {b.gravity_droop_rad(45.0) * 1000:.2f}, "
            f"at 80 deg {b.gravity_droop_rad(80.0) * 1000:.2f}",
            f"  thermal bend {b.thermal_bend_rad(10.0) * 1000:.2f} mrad per "
            f"10 K across the tube",
            f"  instrument resolves {self.resolution_rad * 1e6:.0f} urad "
            f"= {self.resolution_rad * 800 * 1000:.0f} mm at 800 m",
        ]
