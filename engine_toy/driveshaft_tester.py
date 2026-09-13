"""A drive shaft, loaded until something gives.

Pick a material, a tube, a length and a joint. The rig winds torque into
it until it fails, and reports WHICH of several competing failure modes
got there first -- because the interesting thing about a shaft is that
the answer changes with the parameters, and changes in ways that are not
obvious from any single formula.

    python driveshaft_tester.py
    python driveshaft_tester.py --material 6061t6 --joint braze --length 1.4

WHY A DRIVE SHAFT IS THE RIGHT FIRST RIG. It is a tube, which is all
this project builds. It is loaded mainly in TORSION, which exercises the
shear component of the member law that has never had a non-zero input.
And it fails four different ways depending on how it is made, so the
test actually discriminates between choices instead of ranking them:

  TORSIONAL YIELD. The metal gives. Shear stress is T*r/J and it yields
  at 0.577 of tensile yield (von Mises). This is the failure everyone
  expects and frequently not the one that happens.

  TORSIONAL BUCKLING. A thin tube does not have to yield to fail -- it
  can fold into a spiral wave at a fraction of its yield torque, and for
  a long thin shaft this governs. Making the wall thinner to save weight
  moves you toward this and away from yield, which is why a lightweight
  shaft is not simply a lighter version of a heavy one.

  JOINT SHEAR-OUT. The yoke has to be attached. A brazed lap joint
  carries on its overlap area at the braze's own strength -- a third of
  the tube's -- so a short overlap fails long before the shaft does. A
  weld carries at the parent's strength but only across its throat.

  WHIRLING. Nothing breaks, but past its first bending critical speed
  the shaft cannot be run at all. This is a LENGTH limit, independent of
  torque, and it is why long shafts are made in two pieces with a centre
  bearing rather than one long tube.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from milspec import MATERIAL_BY_KEY, TubeSection
from joinability import can_join


@dataclass
class ShaftJoint:
    """How the yoke is attached to the tube."""
    process: str = "fusion-weld"
    #: brazed and bonded joints are concentric laps: this is the overlap
    overlap_m: float = 0.050
    #: a weld is a fillet around the tube: this is its leg
    fillet_leg_m: float = 0.005
    #: a bolted or riveted yoke: how many fasteners and how big
    fastener_count: int = 6
    fastener_diameter_m: float = 0.008

    def capacity_nm(self, tube: TubeSection) -> tuple:
        """Torque the attachment carries, and what limits it."""
        mat = tube.mat
        cap = can_join(mat.key, self.process)
        if not cap.possible:
            return 0.0, f"{self.process} cannot join {mat.key}"
        r = tube.outer_diameter_m / 2.0
        circumference = math.pi * tube.outer_diameter_m
        shear_of_parent = mat.yield_pa * 0.577
        if self.process in ("braze", "adhesive"):
            # A CONCENTRIC LAP. The bond works in shear over the overlap
            # area, at the bond's own strength -- which is the whole
            # story of a brazed shaft and why overlap length is the
            # design variable rather than wall thickness.
            area = circumference * self.overlap_m
            strength = shear_of_parent * cap.joint_efficiency
            return area * strength * r, f"{self.process} lap shear over {self.overlap_m * 1000:.0f} mm"
        if self.process == "fusion-weld":
            throat = self.fillet_leg_m * math.sqrt(0.5)
            area = circumference * throat
            return (area * shear_of_parent * cap.joint_efficiency * r,
                    f"weld throat {throat * 1000:.1f} mm")
        # bolted or riveted: fasteners in double shear on a bolt circle
        a = math.pi * self.fastener_diameter_m ** 2 / 4.0
        force = self.fastener_count * 2.0 * a * shear_of_parent * cap.joint_efficiency
        return force * r, f"{self.fastener_count} fasteners in double shear"


@dataclass
class DriveShaftTest:
    material: str = "4130n"
    outer_diameter_m: float = 0.0762      # 3 inch, a common shaft
    wall_m: float = 0.0018                # 16 gauge
    length_m: float = 1.400
    joint: ShaftJoint = None
    service_speed_rpm: float = 4500.0

    def __post_init__(self) -> None:
        if self.joint is None:
            self.joint = ShaftJoint()

    @property
    def tube(self) -> TubeSection:
        return TubeSection(self.outer_diameter_m, self.wall_m, self.material,
                           designation=f"{self.outer_diameter_m * 1000:.0f} x "
                                       f"{self.wall_m * 1000:.2f} {self.material}")

    # ------------------------------------------------------------------
    def torsional_yield_nm(self) -> float:
        """T = tau * J / r, with tau at the von Mises shear yield."""
        t = self.tube
        polar = 2.0 * t.second_moment_m4        # J = 2I for a circular tube
        r = t.outer_diameter_m / 2.0
        return t.mat.yield_pa * 0.577 * polar / r

    def torsional_buckling_nm(self) -> float:
        """A thin tube folds into a spiral before it yields.

        Donnell's result for a long thin-walled cylinder. It depends on
        the wall-to-radius ratio to the three-halves power, so halving
        the wall costs nearly two thirds of the buckling torque while
        costing only half the yield torque -- which is exactly why thin
        shafts stop failing by yielding and start failing by folding."""
        t = self.tube
        r = t.outer_diameter_m / 2.0 - t.wall_m / 2.0    # mid-wall radius
        nu = 0.3
        tau_cr = (0.272 * t.mat.youngs_pa * (t.wall_m / r) ** 1.5
                  / (1.0 - nu * nu) ** 0.75)
        polar = 2.0 * t.second_moment_m4
        return tau_cr * polar / (t.outer_diameter_m / 2.0)

    def joint_nm(self) -> tuple:
        return self.joint.capacity_nm(self.tube)

    def critical_speed_rpm(self) -> float:
        """First bending critical speed of a simply supported shaft.

        Nothing breaks here -- the shaft simply cannot be run past it,
        which is a limit on LENGTH rather than on torque, and the reason
        long shafts are made in two pieces."""
        t = self.tube
        mass_per_m = t.mass_per_m_kg
        if mass_per_m <= 0 or self.length_m <= 0:
            return float("inf")
        omega = (math.pi / self.length_m) ** 2 * math.sqrt(
            t.mat.youngs_pa * t.second_moment_m4 / mass_per_m)
        return omega * 60.0 / (2.0 * math.pi)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Wind it up until something gives."""
        yield_nm = self.torsional_yield_nm()
        buckle_nm = self.torsional_buckling_nm()
        joint_nm, joint_why = self.joint_nm()
        modes = {
            "torsional yield": yield_nm,
            "torsional buckling": buckle_nm,
            f"joint: {joint_why}": joint_nm,
        }
        mode = min(modes, key=modes.get)
        critical = self.critical_speed_rpm()
        return {
            "tube": self.tube,
            "modes": modes,
            "failed_by": mode,
            "failure_torque_nm": modes[mode],
            "critical_speed_rpm": critical,
            "overspeed": self.service_speed_rpm > critical,
            "mass_kg": self.tube.mass_per_m_kg * self.length_m,
        }

    def report(self) -> list:
        r = self.run()
        t = r["tube"]
        out = [f"  {t.designation}, {self.length_m * 1000:.0f} mm, "
               f"{self.joint.process} yoke", ]
        out.extend(t.describe()[1:])
        out.append(f"    mass {r['mass_kg']:.2f} kg")
        for name, value in sorted(r["modes"].items(), key=lambda kv: kv[1]):
            marker = "  <-- FAILS HERE" if name == r["failed_by"] else ""
            out.append(f"      {name:44s} {value:8.0f} N.m{marker}")
        out.append(f"    first critical speed {r['critical_speed_rpm']:7.0f} rpm"
                   + (f"  -- SERVICE SPEED {self.service_speed_rpm:.0f} IS ABOVE IT"
                      if r["overspeed"] else ""))
        return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--material", default="4130n", choices=sorted(MATERIAL_BY_KEY))
    ap.add_argument("--outer-mm", type=float, default=76.2)
    ap.add_argument("--wall-mm", type=float, default=1.8)
    ap.add_argument("--length", type=float, default=1.4)
    ap.add_argument("--joint", default="fusion-weld",
                    choices=("fusion-weld", "braze", "adhesive", "bolted", "hot-rivet"))
    ap.add_argument("--overlap-mm", type=float, default=50.0)
    ap.add_argument("--rpm", type=float, default=4500.0)
    a = ap.parse_args(argv)
    test = DriveShaftTest(material=a.material, outer_diameter_m=a.outer_mm / 1000.0,
                          wall_m=a.wall_mm / 1000.0, length_m=a.length,
                          joint=ShaftJoint(process=a.joint, overlap_m=a.overlap_mm / 1000.0),
                          service_speed_rpm=a.rpm)
    print("\n".join(test.report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
