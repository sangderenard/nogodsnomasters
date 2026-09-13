"""A telescopic sight, with the compensation math inside it.

A scope is not magnification. Magnification is the least interesting
thing about it. A scope is a device for converting a known trajectory
into an aiming correction, and everything below is that conversion done
honestly rather than by a lookup table someone typed in.

THE THREE FACTS THE MATH RESTS ON

  A SIGHT SITS ABOVE THE BORE. The optic is mounted forty to fifty
  millimetres over the barrel's axis, so the bullet starts BELOW the
  line of sight and has to be launched at a slight upward angle to come
  up and cross it. That is what "zeroing" is, and it is why a rifle
  zeroed at three hundred metres shoots HIGH at one hundred and fifty:
  the trajectory crosses the line of sight twice, once on the way up
  and once on the way down.

  ANGLE IS THE ONLY UNIT THAT TRAVELS. A scope cannot know a distance,
  so its adjustments are angular and their effect grows with range. One
  milliradian subtends one metre at one thousand metres, ten
  centimetres at one hundred, by definition. One minute of angle is a
  sixtieth of a degree, which is 0.2909 mrad, or 2.908 cm at 100 m --
  the familiar "one inch at a hundred yards" is that number rounded,
  and the rounding is where a lot of confusion comes from. This module
  keeps the exact value.

  DRIFT IS NOT A SIDEWAYS PUSH. Wind deflection comes out of the drag
  integration acting on velocity relative to the air, which the
  trajectory law already does. A scope's windage correction is then
  just that lateral miss expressed as an angle.

WHAT THIS OBJECT DOES

  `zero()` solves the launch angle that puts the trajectory through the
  line of sight at the zero range -- a real solve against the compiled
  trajectory law, not an approximation.

  `solution()` gives the firing solution for a target at some range in
  some wind: the drop and drift relative to the LINE OF SIGHT, those
  same numbers as angles, the number of clicks to dial, and the aim
  direction to hand a gun.

  `reticle_marks()` says what the marks in the glass actually cover at
  a given range, which is how you hold over without touching a turret.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Exact, not the familiar rounding. One minute of angle is a sixtieth
# of a degree; in milliradians that is pi/(180*60)*1000.
MOA_IN_MRAD = math.pi / (180.0 * 60.0) * 1000.0        # 0.29089 mrad
MRAD_CM_AT_100M = 10.0                                  # by definition
MOA_CM_AT_100M = MOA_IN_MRAD * MRAD_CM_AT_100M          # 2.9089 cm


@dataclass(frozen=True)
class Reticle:
    """The marks in the glass, and what they subtend.

    A FIRST focal plane reticle scales with magnification, so its
    subtensions are true at every power. A SECOND focal plane reticle
    stays the same apparent size, so its subtensions are only true at
    one stated magnification -- which is the single most common way a
    hold is got wrong, and why it is declared here rather than
    assumed."""
    kind: str = "mil-dot"                 # "mil-dot" | "moa"
    spacing: float = 1.0                  # mrad or MOA between marks
    marks: int = 5                        # marks below centre
    focal_plane: str = "first"            # "first" | "second"
    true_at_magnification: float = 10.0   # second focal plane only

    @property
    def spacing_mrad(self) -> float:
        return self.spacing if self.kind == "mil-dot" else self.spacing * MOA_IN_MRAD

    def subtension_m(self, range_m: float, magnification: float | None = None) -> float:
        """What one mark covers at this range."""
        scale = 1.0
        if self.focal_plane == "second" and magnification:
            # the marks only mean what they say at the stated power
            scale = self.true_at_magnification / max(magnification, 1e-6)
        return self.spacing_mrad * scale * range_m / 1000.0


@dataclass
class FiringSolution:
    range_m: float
    drop_m: float                 # below the line of sight, positive = below
    drift_m: float                # lateral miss, positive = downwind
    elevation_mrad: float
    windage_mrad: float
    elevation_clicks: int
    windage_clicks: int
    arrival_speed_m_s: float
    arrival_energy_j: float
    time_of_flight_s: float
    aim_point: tuple = (0.0, 0.0, 0.0)

    @property
    def elevation_moa(self) -> float:
        return self.elevation_mrad / MOA_IN_MRAD

    @property
    def windage_moa(self) -> float:
        return self.windage_mrad / MOA_IN_MRAD

    def lines(self) -> list[str]:
        return [
            f"  {self.range_m:5.0f} m   drop {self.drop_m * 100:7.1f} cm   "
            f"drift {self.drift_m * 100:6.1f} cm",
            f"          elevation {self.elevation_mrad:5.2f} mrad "
            f"({self.elevation_moa:5.2f} MOA, {self.elevation_clicks:3d} clicks up)",
            f"          windage   {self.windage_mrad:5.2f} mrad "
            f"({self.windage_moa:5.2f} MOA, {self.windage_clicks:3d} clicks)",
            f"          arrives {self.arrival_speed_m_s:5.0f} m/s, "
            f"{self.arrival_energy_j / 1000:5.2f} kJ, after {self.time_of_flight_s:4.2f} s",
        ]


@dataclass
class Scope:
    """One sight on one rifle, zeroed for one cartridge."""
    magnification: float = 10.0
    sight_height_m: float = 0.045        # optic axis above bore axis
    zero_range_m: float = 100.0
    click_mrad: float = 0.1              # 0.1 mrad, or MOA_IN_MRAD/4 for quarter-MOA
    reticle: Reticle = field(default_factory=Reticle)
    # solved by zero(); the launch angle above the line of sight
    zero_elevation_rad: float = 0.0
    _trace: object = field(default=None, repr=False)

    # ---------------- the trajectory it is solving against ----------------
    def bind(self, trace_fn) -> "Scope":
        """Give the scope the trajectory it is sighting.

        `trace_fn(elevation_rad, azimuth_rad, range_m, wind)` must return
        `(drop_m, drift_m, speed, energy, time)` relative to the line of
        sight. Passed in rather than imported so the scope can be
        checked against an analytic vacuum trajectory as well as against
        the compiled drag law."""
        self._trace = trace_fn
        return self

    # ---------------- zeroing ----------------
    def zero(self, tolerance_m: float = 0.0005, iterations: int = 60) -> float:
        """Solve the launch angle that puts the bullet on the line of
        sight at the zero range.

        A bisection rather than a formula, because the trajectory it is
        solving against has drag in it and there is no closed form. The
        bracket is generous at the top: a heavy subsonic round at a long
        zero genuinely needs a lot of elevation."""
        if self._trace is None:
            raise RuntimeError("scope has no trajectory bound")
        low, high = -0.02, 0.10
        for _ in range(iterations):
            mid = (low + high) / 2.0
            drop, _, _, _, _ = self._trace(mid, 0.0, self.zero_range_m, (0.0, 0.0, 0.0))
            if abs(drop) <= tolerance_m:
                break
            # drop is positive BELOW the line of sight: too low means
            # more elevation
            if drop > 0.0:
                low = mid
            else:
                high = mid
        self.zero_elevation_rad = mid
        return mid

    # ---------------- the firing solution ----------------
    def solution(self, range_m: float, wind=(0.0, 0.0, 0.0)) -> FiringSolution:
        """What to dial, or hold, for a target at this range."""
        if self._trace is None:
            raise RuntimeError("scope has no trajectory bound")
        drop, drift, speed, energy, tof = self._trace(
            self.zero_elevation_rad, 0.0, range_m, wind)
        # angle = miss / range, in milliradians
        elevation_mrad = drop / max(range_m, 1e-9) * 1000.0
        windage_mrad = -drift / max(range_m, 1e-9) * 1000.0
        return FiringSolution(
            range_m=range_m, drop_m=drop, drift_m=drift,
            elevation_mrad=elevation_mrad, windage_mrad=windage_mrad,
            elevation_clicks=int(round(elevation_mrad / max(self.click_mrad, 1e-9))),
            windage_clicks=int(round(windage_mrad / max(self.click_mrad, 1e-9))),
            arrival_speed_m_s=speed, arrival_energy_j=energy, time_of_flight_s=tof)

    def hold(self, solution: FiringSolution) -> str:
        """The same correction expressed as a reticle hold, for when you
        would rather not touch the turrets."""
        marks = solution.elevation_mrad / max(self.reticle.spacing_mrad, 1e-9)
        wind_marks = solution.windage_mrad / max(self.reticle.spacing_mrad, 1e-9)
        unit = "mil" if self.reticle.kind == "mil-dot" else "MOA"
        note = ""
        if self.reticle.focal_plane == "second" and abs(
                self.magnification - self.reticle.true_at_magnification) > 1e-6:
            note = (f"  (second focal plane: marks are only true at "
                    f"{self.reticle.true_at_magnification:.0f}x, not {self.magnification:.0f}x)")
        return (f"hold {marks:.1f} {unit} high, {abs(wind_marks):.1f} "
                f"{'left' if wind_marks < 0 else 'right'}{note}")

    def reticle_marks(self, range_m: float) -> list[str]:
        """What each mark covers downrange -- the ranging use of a
        reticle, and the reason a mil-dot can measure a target as well
        as hold on one."""
        out = []
        for i in range(1, self.reticle.marks + 1):
            out.append(f"    mark {i}: {self.reticle.subtension_m(range_m, self.magnification) * i * 100:6.1f} cm "
                       f"at {range_m:.0f} m")
        return out

    def describe(self) -> list[str]:
        return [
            f"  {self.magnification:.0f}x scope, {self.reticle.kind} reticle "
            f"({self.reticle.focal_plane} focal plane), {self.reticle.spacing_mrad:.2f} mrad marks",
            f"    sight {self.sight_height_m * 1000:.0f} mm over bore, zeroed at "
            f"{self.zero_range_m:.0f} m, {self.click_mrad:.2f} mrad per click",
            f"    launch angle at zero: {self.zero_elevation_rad * 1000:.2f} mrad "
            f"({self.zero_elevation_rad * 1000 / MOA_IN_MRAD:.2f} MOA above the line of sight)",
        ]


# =====================================================================
#  BINDING IT TO THE COMPILED TRAJECTORY LAW
# =====================================================================
def compiled_trace(calibre: str, *, dt: float = 5e-4, drag_coefficient: float = 0.295,
                   air_density: float = 1.225, sight_height_m: float = 0.045):
    """A trace function backed by the native trajectory kernel.

    Returns drop and drift RELATIVE TO THE LINE OF SIGHT, which is what
    a scope cares about -- the bore is below the optic, so the bullet
    starts `sight_height_m` low and has to climb to meet it."""
    from calibres import get_calibre
    from symbolic_parts import compile_trajectory_ssa
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)

    c = get_calibre(calibre)
    compiled = compile_trajectory_ssa()
    fn = compiled.function
    artifact = compile_artifact(emit_ssa_function_to_llvm(
        compiled.module, fn.name, entry_name="trajectory_step"))
    ids = {n: v.id for n, v in zip(fn.metadata["argument_names"], fn.args)}
    outs = dict(fn.metadata["named_outputs"])

    def trace(elevation_rad: float, azimuth_rad: float, range_m: float, wind):
        # the line of sight is the +z axis through the optic; the bore
        # starts below it and points up by the launch angle
        p = np.array([0.0, -sight_height_m, 0.0])
        v = np.array([math.sin(azimuth_rad) * c.muzzle_m_s,
                      math.sin(elevation_rad) * c.muzzle_m_s,
                      math.cos(elevation_rad) * math.cos(azimuth_rad) * c.muzzle_m_s])
        vals = {"dt": dt, "mass_kg": c.mass_kg, "diameter_m": c.diameter_m,
                "drag_coefficient": drag_coefficient, "air_density_kg_m3": air_density,
                "gravity_m_s2": 9.80665, "wind_x_m_s": wind[0],
                "wind_y_m_s": wind[1], "wind_z_m_s": wind[2]}
        t = 0.0
        speed = float(np.linalg.norm(v))
        energy = 0.5 * c.mass_kg * speed * speed
        for _ in range(400_000):
            vals.update({"position_x_m": p[0], "position_y_m": p[1], "position_z_m": p[2],
                         "velocity_x_m_s": v[0], "velocity_y_m_s": v[1], "velocity_z_m_s": v[2]})
            ex = prepare_artifact_execution(
                artifact, {ids[n]: np.array(x, dtype=np.float64) for n, x in vals.items()})
            ex.run()
            g = lambda n: float(np.asarray(ex.buffers[outs[n]]).reshape(-1)[0])
            p = np.array([g("position_next_x_m"), g("position_next_y_m"), g("position_next_z_m")])
            v = np.array([g("velocity_next_x_m_s"), g("velocity_next_y_m_s"),
                          g("velocity_next_z_m_s")])
            speed, energy = g("speed_m_s"), g("kinetic_energy_j")
            t += dt
            if p[2] >= range_m:
                break
        # drop is measured DOWN from the line of sight, so a bullet
        # below it has a positive drop
        return (-float(p[1]), float(p[0]), speed, energy, t)

    return trace


def scoped(calibre: str, *, zero_range_m: float = 100.0, magnification: float = 10.0,
           reticle: Reticle | None = None, click_mrad: float = 0.1) -> Scope:
    """A scope, bound to a cartridge and zeroed, ready to give
    solutions."""
    s = Scope(magnification=magnification, zero_range_m=zero_range_m,
              click_mrad=click_mrad, reticle=reticle or Reticle())
    s.bind(compiled_trace(calibre, sight_height_m=s.sight_height_m))
    s.zero()
    return s
