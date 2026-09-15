"""Real couplings between an engine and a load, and the difference
between the ones that slip and the ones that do not.

The dyno rig here used one generic friction clutch for everything,
which meant every engine was tested through a coupling that slips. A
real test cell does not do that: it uses the coupling the engine is
actually built to drive through, and on a big industrial engine that
is very often one that locks.

Four kinds, and the distinctions matter:

  "dry-friction"      an ordinary dry plate, clamped by a spring and
                      released hydraulically or by cable. The hydraulic
                      part is only the ACTUATION -- the clutch itself is
                      still friction and still slips under enough
                      torque. This is what a workshop usually means by
                      "a hydraulic clutch", and it does not solve the
                      slipping problem.

  "wet-multi-plate"   friction plates running in oil, hydraulically
                      APPLIED. Its torque capacity is a real product of
                      the apply pressure, the piston area, the number of
                      friction faces, the mean radius and the friction
                      coefficient -- so it is modulated by pressure, and
                      once the capacity exceeds the torque being asked
                      of it, it stops slipping entirely and drives
                      solidly. Tractors, PTOs, motorcycles and every
                      automatic gearbox use these.

  "fluid-coupling"    a Foettinger coupling / fluid flywheel: an
                      impeller and a turbine in oil, no friction faces
                      at all. Torque goes with the SQUARE of the speed
                      difference (T = lambda rho D^5 n^2), so it always
                      slips a little -- 2 to 5 % at rated -- and can
                      never lock on its own. Wonderfully smooth, and a
                      standard soft-start on big industrial engines and
                      conveyor drives, but it can never hold a dyno
                      brake without slip.

  "torque-converter"  a fluid coupling with a stator, so it MULTIPLIES
                      torque at stall (a real 2:1-ish at zero output
                      speed, falling to 1:1 at the coupling point).

The thing that actually removes the slip is the LOCK-UP CLUTCH: a
small plate that mechanically bridges a fluid coupling or converter
once the two sides are turning at nearly the same speed. Fitted to a
fluid coupling or a converter it gives you both -- a soft start and a
solid drive. That is the honest answer to "can the dyno hold brake
torque without slipping": yes, through a wet plate or a locked
converter, not through a dry plate and not through a bare fluid
coupling.

Apply pressure comes from the engine's OWN hydraulic reservoir where
it has one (`plant.hydraulics`), which is the point: the rig is built
from parts matched to the engine and fed from its own circuits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

# a real wet-plate friction material in oil
WET_PLATE_MU = 0.12
DRY_PLATE_MU = 0.35
# a fluid coupling's own characteristic: T = lambda * rho * D^5 * n^2,
# with lambda the dimensionless torque factor for the working point
FLUID_LAMBDA = 3.6e-3
OIL_DENSITY_KG_M3 = 870.0
LOCKUP_SLIP_RAD_S = 6.0        # lock-up can close when the two sides are this close
LOCKUP_RELEASE_RAD_S = 25.0    # ...and drops out again if they pull this far apart


@dataclass
class Coupling:
    """One coupling between a driving and a driven shaft.

    `step` returns the torque transmitted, positive from drive to load,
    and reports whether it is slipping."""
    kind: str = "dry-friction"
    # geometry / rating
    plate_count: int = 1                  # friction FACES (a single dry plate has two)
    mean_radius_m: float = 0.12
    piston_area_m2: float = 0.006
    max_apply_pressure_pa: float = 2_000_000.0
    diameter_m: float = 0.30              # fluid coupling / converter
    stall_torque_ratio: float = 2.0       # converter only
    lockup_fitted: bool = False
    # rating fallback for a dry plate, which is normally quoted as a torque
    rated_torque_nm: float = 500.0
    stiffness_nm_per_rad_s: float = 4000.0
    # centrifugal only: the speed the shoes start to grip at, and the
    # speed `rated_torque_nm` is quoted at
    engage_rad_s: float = 0.0
    rated_rad_s: float = 0.0
    # live
    apply_pressure_pa: float = 0.0
    drive_omega_rad_s: float = 0.0        # centrifugal: what throws the shoes
    engagement: float = 1.0               # 0..1 the operator's pedal / apply command
    locked: bool = False
    slipping: bool = False
    slip_rad_s: float = 0.0
    transmitted_nm: float = 0.0
    capacity_nm: float = 0.0
    heat_w: float = 0.0

    # ------------------------------------------------------------------
    @property
    def is_hydraulically_applied(self) -> bool:
        return self.kind == "wet-multi-plate"

    def torque_capacity_nm(self) -> float:
        """What this coupling can hold before it slips."""
        if self.kind == "wet-multi-plate":
            # clamp force from the apply piston, through every friction
            # face, at the mean radius
            clamp_n = self.apply_pressure_pa * self.piston_area_m2
            return clamp_n * WET_PLATE_MU * self.mean_radius_m * max(1, self.plate_count)
        if self.kind == "dry-friction":
            return self.rated_torque_nm * max(0.0, min(1.0, self.engagement))
        if self.kind == "centrifugal":
            # THE SHOES ARE THROWN OUTWARD BY ROTATION, and a spring holds
            # them in until there is enough of it. Net grip force goes as
            # `m*r*w**2 - F_spring`, so capacity goes as `w**2 -
            # w_engage**2` and is exactly ZERO below the engagement speed.
            #
            # That is the whole point of the thing and the reason a
            # trimmer cannot be stalled by its load: below engagement the
            # clutch transmits nothing at all, so the engine idles on
            # while the head sits still. There is no operator pedal --
            # `engagement` is not consulted, because there is nothing to
            # press.
            omega = abs(self.drive_omega_rad_s)
            engage = max(self.engage_rad_s, 0.0)
            rated = max(self.rated_rad_s, engage + 1.0)
            if omega <= engage:
                return 0.0
            span = rated * rated - engage * engage
            return self.rated_torque_nm * (omega * omega - engage * engage) / span
        return 0.0      # a fluid coupling has no static capacity at all

    # How much relative speed this coupling takes to go from stuck to
    # fully slipping -- the width of the transition the solver's tanh
    # stands in for. This is a real property of the hardware, not a
    # numerical knob: a WET clutch transmits through an oil film and
    # genuinely modulates over a measurable slip (which is exactly how
    # a powershift gearbox shifts under load), while a DRY plate grabs
    # far more sharply. Too narrow and the element becomes bang-bang:
    # it saturates within a couple of rpm and then chatters around zero
    # relative speed, which is the classic stick-slip problem and shows
    # up as a permanently saturated clutch alternating sign.
    TRANSITION_SLIP_RAD_S = {"wet-multi-plate": 2.0, "dry-friction": 0.8,
                             "centrifugal": 0.8}

    @property
    def transition_slip_rad_s(self) -> float:
        return self.TRANSITION_SLIP_RAD_S.get(self.kind, 1.0)

    def set_apply(self, apply_pressure_pa: float) -> float:
        """Take this tick's apply pressure and return the capacity it
        buys. Used by the friction kinds, where the SOLVER's own
        spring/damper is what actually computes the transmitted torque
        and this capacity is the ceiling it saturates at -- one
        capacity, owned here, rather than two caps in series."""
        self.apply_pressure_pa = min(apply_pressure_pa, self.max_apply_pressure_pa) * max(
            0.0, min(1.0, self.engagement))
        self.capacity_nm = self.torque_capacity_nm()
        return self.capacity_nm

    def observe(self, transmitted_nm: float, slip_rad_s: float) -> None:
        """Record what the solver actually passed through this coupling,
        so the coupling reports its own state without ever re-capping a
        torque that has already been limited by its capacity."""
        self.transmitted_nm = transmitted_nm
        self.slip_rad_s = slip_rad_s
        cap = max(self.capacity_nm, 1e-9)
        # a friction clutch is slipping when it is riding its ceiling
        self.slipping = (abs(transmitted_nm) >= cap * 0.98
                         and abs(slip_rad_s) > self.transition_slip_rad_s)
        self.locked = not self.slipping
        self.heat_w = abs(transmitted_nm * slip_rad_s) if self.slipping else 0.0

    def step(self, dt: float, omega_drive: float, omega_load: float,
             demanded_nm: float, apply_pressure_pa: float = 0.0) -> float:
        self.apply_pressure_pa = min(apply_pressure_pa, self.max_apply_pressure_pa) * max(
            0.0, min(1.0, self.engagement))
        slip = omega_drive - omega_load
        self.slip_rad_s = slip
        # the centrifugal capacity is a function of the DRIVING speed, so
        # it has to be recorded before the capacity is asked for
        self.drive_omega_rad_s = abs(omega_drive)

        if self.kind in ("fluid-coupling", "torque-converter"):
            # a lock-up plate bridges it once the two sides are close
            if self.lockup_fitted:
                if not self.locked and abs(slip) <= LOCKUP_SLIP_RAD_S and self.engagement > 0.5:
                    self.locked = True
                elif self.locked and abs(slip) >= LOCKUP_RELEASE_RAD_S:
                    self.locked = False
            else:
                self.locked = False
            if self.locked:
                # mechanically bridged: it drives solidly, no slip, no
                # churning loss -- this is what lets a dyno hold torque
                self.slipping = False
                self.capacity_nm = abs(demanded_nm) * 2.0
                self.transmitted_nm = demanded_nm
                self.heat_w = 0.0
                return self.transmitted_nm
            # hydrodynamic: torque with the SQUARE of the speed difference
            n = abs(slip) / (2.0 * math.pi)          # rev/s of slip
            t = FLUID_LAMBDA * OIL_DENSITY_KG_M3 * self.diameter_m ** 5 * n * n
            t = math.copysign(t, slip)
            if self.kind == "torque-converter" and abs(omega_drive) > 1e-6:
                # stall multiplication falling to 1:1 at the coupling point
                ratio = max(0.0, min(1.0, omega_load / max(abs(omega_drive), 1e-6)))
                t *= 1.0 + (self.stall_torque_ratio - 1.0) * (1.0 - ratio)
            self.slipping = True
            self.capacity_nm = abs(t)
            self.transmitted_nm = t
            self.heat_w = abs(t * slip)              # churning loss is real
            return self.transmitted_nm

        # --- friction kinds ---
        cap = self.torque_capacity_nm()
        self.capacity_nm = cap
        if abs(demanded_nm) <= cap:
            # it holds: locked up solid, nothing slipping, no heat
            self.slipping = False
            self.locked = True
            self.transmitted_nm = demanded_nm
            self.heat_w = 0.0
        else:
            # past capacity it slips, and every watt of that is heat
            self.slipping = True
            self.locked = False
            self.transmitted_nm = math.copysign(cap, demanded_nm if demanded_nm else slip)
            self.heat_w = abs(self.transmitted_nm * slip)
        return self.transmitted_nm

    def describe(self) -> str:
        state = ("LOCKED" if self.locked else ("slipping" if self.slipping else "holding"))
        bits = [f"{self.kind} {state}"]
        if self.capacity_nm:
            bits.append(f"capacity {self.capacity_nm:.0f} Nm")
        if self.is_hydraulically_applied:
            bits.append(f"apply {self.apply_pressure_pa / 1e5:.0f} bar")
        if self.slipping and abs(self.slip_rad_s) > 0.1:
            bits.append(f"slip {self.slip_rad_s * 60 / (2 * math.pi):.0f} rpm, {self.heat_w / 1000:.1f} kW of heat")
        return ", ".join(bits)


@dataclass
class CouplingSpec:
    """What an engine recommends be used to couple it to a load -- and
    therefore what the dyno rig should build. Declared per engine."""
    kind: str = "dry-friction"
    lockup_fitted: bool = False
    plate_count: int = 1
    mean_radius_m: float = 0.12
    piston_area_m2: float = 0.006
    apply_pressure_pa: float = 1_800_000.0
    diameter_m: float = 0.30
    stall_torque_ratio: float = 2.0
    # where the apply pressure comes from: the engine's own hydraulic
    # reservoir when it has one, else a dedicated pump on the rig
    apply_source: str = "engine-hydraulics"     # | "rig-pump" | "mechanical-spring"
                                                # | "none" for a centrifugal
    # centrifugal only: the speeds that decide when it grips and what it
    # holds. Zero elsewhere, and unread by every other kind.
    engage_rad_s: float = 0.0
    rated_rad_s: float = 0.0
    #: a centrifugal clutch is rated by its own shoes, not by the rig's
    #: request, so the spec may override what `build` is handed
    rated_torque_nm: float = 0.0

    def build(self, rated_torque_nm: float) -> Coupling:
        rating = self.rated_torque_nm or rated_torque_nm
        return Coupling(kind=self.kind, plate_count=self.plate_count, mean_radius_m=self.mean_radius_m,
                        piston_area_m2=self.piston_area_m2, max_apply_pressure_pa=self.apply_pressure_pa,
                        diameter_m=self.diameter_m, stall_torque_ratio=self.stall_torque_ratio,
                        lockup_fitted=self.lockup_fitted, rated_torque_nm=rating,
                        engage_rad_s=self.engage_rad_s, rated_rad_s=self.rated_rad_s)


def recommended_for(engine) -> CouplingSpec:
    """What this engine should be coupled through, from what it IS.

    The reasoning is the ordinary engineering one, not a table of
    special cases: a big slow industrial or marine engine wants a soft
    start and then a solid drive, so a fluid coupling or wet plate with
    lock-up; a road engine uses the dry plate it was built with; a
    turbine or a free-piston machine has no crank to clamp and needs a
    fluid coupling; an electric drive needs no coupling at all.
    """
    declared = getattr(engine, "coupling", None)
    if declared is not None:
        return declared
    arch = engine.architecture
    peak = float(getattr(engine, "peak_torque_nm", 500.0))
    if engine.kind == "turbine":
        # a turbine cannot be clamped to a stationary load: it needs
        # something that will let it come up to speed
        return CouplingSpec(kind="torque-converter", lockup_fitted=True, diameter_m=0.34,
                            apply_source="rig-pump")
    # A SMALL TWO-STROKE HAS A CENTRIFUGAL CLUTCH, and until now that was
    # a claim in a comment: `engines.py` says the trimmer has "a real
    # centrifugal clutch (engages above idle to spin the cutting head)"
    # while nothing in this file, the sim, or the port had ever heard of
    # one. It got a plain dry plate, which can be stalled by its load --
    # measured, the trimmer died 31 times in 90 seconds and sat across
    # the track as a wall for everyone behind it.
    #
    # Derived from what the engine DECLARES rather than from its name:
    # two-stroke, and small enough that a clutch is the only thing between
    # the crank and the tool.
    architecture = getattr(engine, "architecture", None)
    if (architecture is not None and getattr(architecture, "two_stroke", False)
            and float(getattr(engine, "displacement_l", 1.0)) <= 0.10):
        idle = float(getattr(engine, "idle_rpm", 2800) or 2800)
        peak = float(getattr(engine, "torque_peak_rpm", idle * 2.0) or idle * 2.0)
        rpm_to_rad = math.pi / 30.0
        return CouplingSpec(
            kind="centrifugal", apply_source="none",
            rated_torque_nm=float(getattr(engine, "clutch_torque_nm", 3.5) or 3.5),
            engage_rad_s=idle * 1.45 * rpm_to_rad,
            rated_rad_s=peak * rpm_to_rad)
    if engine.kind in ("electric", "servo"):
        return CouplingSpec(kind="dry-friction", apply_source="mechanical-spring")
    if peak >= 2000.0 or getattr(engine, "compression_ignition", False) and peak >= 1200.0:
        # a heavy diesel: a wet multi-plate, applied off its own
        # hydraulics, big enough to hold its own peak torque solidly
        radius = 0.16 if peak < 5000.0 else 0.24
        # size the piston so full apply pressure holds peak torque with
        # a real margin, rather than picking a number
        faces = 8 if peak < 5000.0 else 12
        pilot_pa = 3_500_000.0      # the machine's own regulated pilot supply
        area = peak * 1.6 / (pilot_pa * WET_PLATE_MU * radius * faces)
        return CouplingSpec(kind="wet-multi-plate", plate_count=faces, mean_radius_m=radius,
                            piston_area_m2=area, apply_pressure_pa=pilot_pa,
                            apply_source="engine-hydraulics")
    if arch.cylinders == 0:
        return CouplingSpec(kind="fluid-coupling", lockup_fitted=False, diameter_m=0.28,
                            apply_source="mechanical-spring")
    return CouplingSpec(kind="dry-friction", apply_source="mechanical-spring")
