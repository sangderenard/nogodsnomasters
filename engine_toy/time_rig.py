"""A TEST MACHINE FOR THE TIME FIELD: simple rotors, simple forces, one table.

    python time_rig.py

The engine sim is far too big to see the time field working in. This is
the smallest machine that can still be wrong in all the interesting ways:
a row of flywheels, each driven by a constant torque, each coupled to a
common load shaft through a joint of a declared kind, each sitting at its
own local time velocity. Everything that matters is printed every frame,
side by side, so the question "is this behaving truthfully" can be
answered by looking rather than by reasoning.

WHAT THE TABLE IS FOR

Each row is one lane. Read across and you can check, by eye:

  tau        its clock rate. 1.000 is the reference.
  dlogt/dt   whether the field is RAMPING. Zero means steady, and a
             steady gradient must cost nothing.
  world_s    world time it actually advanced this frame. A dilated lane
             advances less -- that is the price, and it should be
             visible without being explained.
  local_s    time it advanced on its OWN clock. Its physics is exact
             against this.
  omega      its own speed, in its own frame.
  shear      torque the joint had to supply. MUST be 0.000 while the
             field is steady, non-zero only while ramping, and inf for a
             joint with no rate freedom.
  dE         work in minus kinetic energy gained, on the lane's own
             clock. This is the conservation check and it must stay at
             zero whatever tau is: local physics cannot know its own
             rate. If this drifts, the gauge argument is wrong.

The last column is the one to watch. Everything else can look plausible
while being subtly scaled wrong; `dE` is the number that catches it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from time_field import TimeField, TimeFieldConfig, TimeForceStore, ADAPTORS, adaptor_for


@dataclass
class Rotor:
    """One flywheel under a constant torque. The simplest machine that
    still has a state worth getting wrong."""

    name: str
    inertia_kg_m2: float
    drive_torque_nm: float
    joint_kind: str
    omega_rad_s: float = 0.0
    #: accumulated on the rotor's OWN clock
    work_in_j: float = 0.0
    local_s: float = 0.0
    world_s: float = 0.0
    sheared: bool = False

    @property
    def kinetic_j(self) -> float:
        return 0.5 * self.inertia_kg_m2 * self.omega_rad_s ** 2

    @property
    def energy_error_j(self) -> float:
        """Work done minus kinetic energy gained, on the local clock.
        Must be zero regardless of time velocity."""
        return self.work_in_j - self.kinetic_j

    def advance(self, local_dt: float, resisting_nm: float = 0.0) -> None:
        """Integrate on the rotor's own clock. Nothing in here knows what
        a time velocity is, which is the point: local physics is written
        against local time and is exact."""
        if local_dt <= 0.0:
            return
        net = self.drive_torque_nm - resisting_nm
        # midpoint work so the energy ledger closes exactly for a
        # constant torque rather than to first order
        omega0 = self.omega_rad_s
        self.omega_rad_s = omega0 + (net / self.inertia_kg_m2) * local_dt
        mean_omega = 0.5 * (omega0 + self.omega_rad_s)
        self.work_in_j += net * mean_omega * local_dt
        self.local_s += local_dt


@dataclass
class TimeRig:
    rotors: list[Rotor]
    field_: TimeField
    reference_dt_s: float = 1.0 / 60.0
    substeps: int = 8
    frame: int = 0

    @classmethod
    def build(cls, *, store: str = TimeForceStore.ADAPTOR_INERTIA) -> "TimeRig":
        rotors = [
            #            name            I      torque  joint
            Rotor("free-diff",          0.40,   12.0,  "differential"),
            Rotor("slow-diff",          0.40,   12.0,  "differential"),
            Rotor("converter",          0.40,   12.0,  "torque-converter"),
            Rotor("rigid",              0.40,   12.0,  "rigid-keyed-hub"),
            Rotor("timing-locked",      0.40,   12.0,  "camshaft-timing-drive"),
        ]
        names = [r.name for r in rotors] + ["load"]
        tf = TimeField.flat(names, TimeFieldConfig(store=store))
        return cls(rotors=rotors, field_=tf)

    # -- the frame -----------------------------------------------------
    def step(self) -> None:
        """One reference frame. Every lane takes the SAME number of
        substeps; each lane's own step is its window divided by that
        count, and its window is its time velocity times the reference
        frame. Lockstep for the batch, stability for the lane."""
        self.frame += 1
        for rotor in self.rotors:
            tau = self.field_.velocity(rotor.name)
            window = self.reference_dt_s * tau      # world time it may cover
            local_step = window / self.substeps
            reaction = self.field_.shear_reaction_nm(
                rotor.name, "load", rotor.joint_kind,
                omega_ref_rad_s=max(abs(rotor.omega_rad_s), 1.0))
            if math.isinf(reaction):
                rotor.sheared = True
                reaction = 0.0                      # it has failed; it carries nothing
            for _ in range(self.substeps):
                rotor.advance(local_step, resisting_nm=0.0)
            rotor.world_s += window
            rotor._last_shear = reaction            # telemetry only

    def ramp(self, name: str, target_tau: float, max_ramp_per_s: float) -> None:
        self.field_.set_target(name, math.log(target_tau),
                               dt=self.reference_dt_s,
                               max_ramp_per_s=max_ramp_per_s)

    # -- the whole point -----------------------------------------------
    def table(self) -> str:
        head = (f"frame {self.frame:3d}   reference dt {self.reference_dt_s*1000:.2f} ms"
                f"   K={self.substeps}   store={self.field_.config.store}")
        cols = (f"{'lane':<15}{'joint':<24}{'tau':>7}{'dlogt/dt':>10}"
                f"{'world_s':>9}{'local_s':>9}{'omega':>9}{'shear':>9}{'dE (J)':>11}  status")
        rows = [head, cols, "-" * len(cols)]
        for r in self.rotors:
            i = self.field_._index[r.name]
            shear = getattr(r, "_last_shear", 0.0)
            status = "SHEARED" if r.sheared else ("ramping" if abs(self.field_.dlog_tau_dt[i]) > 1e-12 else "steady")
            rows.append(
                f"{r.name:<15}{r.joint_kind:<24}"
                f"{self.field_.velocity(r.name):>7.3f}"
                f"{self.field_.dlog_tau_dt[i]:>10.3f}"
                f"{r.world_s:>9.4f}{r.local_s:>9.4f}{r.omega_rad_s:>9.2f}"
                f"{shear:>9.3f}{r.energy_error_j:>11.2e}  {status}"
            )
        return "\n".join(rows)


def main() -> None:
    import dataclasses

    # give the differential a real spider inertia so the ramp limit comes
    # from hardware rather than from a constant
    ADAPTORS["differential"] = dataclasses.replace(
        ADAPTORS["differential"], store_inertia_kg_m2=0.02)
    ADAPTORS["torque-converter"] = dataclasses.replace(
        ADAPTORS["torque-converter"], store_inertia_kg_m2=0.05)

    rig = TimeRig.build()
    print("PHASE 1 -- flat field. Every lane at the reference, nothing dilated.")
    for _ in range(3):
        rig.step()
    print(rig.table())

    print("\nPHASE 2 -- ramp three lanes down to half rate. The field is MOVING,")
    print("           so a joint with rate freedom must supply torque, and a")
    print("           joint without one shears.")
    for _ in range(6):
        for name in ("slow-diff", "converter", "rigid", "timing-locked"):
            rig.ramp(name, 0.5, max_ramp_per_s=8.0)
        rig.step()
    print(rig.table())

    print("\nPHASE 3 -- field settled at the new levels. The gradient is STEADY,")
    print("           so it must now cost exactly nothing again.")
    rig.field_.settle()
    for _ in range(6):
        rig.step()
    print(rig.table())

    worst = max(abs(r.energy_error_j) for r in rig.rotors)
    print(f"\nworst energy error across every lane and every rate: {worst:.3e} J")
    print("free-diff advanced %.4f s of world time; slow-diff advanced %.4f s"
          % (rig.rotors[0].world_s, rig.rotors[1].world_s))
    print("-> the dilated lane did less work in the same wall clock. That is the price.")


if __name__ == "__main__":
    main()


# =====================================================================
# THE COUPLED RIG: two lanes at different rates actually driving each
# other, so the reaction is APPLIED and the ledger has to close ACROSS
# the gradient rather than only within each lane.
# =====================================================================

@dataclass
class Differential:
    """An open differential, as the thing that joins two time zones.

    THE RESULT THIS RIG EXISTS TO CHECK: the differential's constraint is
    GEOMETRIC -- theta_c = (theta_a + theta_b)/2, an identity about
    angles -- and work is torque times angle. Both are scalars that carry
    no clock. So for an IDEAL open differential, time drops out of the
    joint completely and it needs no time awareness whatsoever. That is
    why it "just works".

    Time comes back the moment the joint has a RATE-dependent term of its
    own: carrier inertia, or a limited-slip bias that reads the rate
    difference. Then each side's speed must be resampled into the
    differential's own frame,
        omega_side_in_joint_frame = omega_side * (tau_side / tau_joint)
    because the two sides' speeds are quoted on different clocks and
    cannot be compared, let alone averaged, without it. `resample_rates`
    is the switch, so the rig can show the ledger breaking when it is
    off.
    """

    name: str
    carrier_torque_nm: float
    carrier_inertia_kg_m2: float = 0.0
    lsd_bias_nm_per_rad_s: float = 0.0
    resample_rates: bool = True
    #: ledger, all in torque*angle, which is frame-free
    carrier_work_j: float = 0.0
    delivered_work_j: float = 0.0
    #: an LSD is a friction device: the bias torque does work against the
    #: rate difference and that work becomes heat. Without this term the
    #: ledger looks broken when it is merely incomplete.
    dissipated_j: float = 0.0

    def side_rate_in_frame(self, field_: TimeField, side: str, omega: float) -> float:
        if not self.resample_rates:
            return omega                     # the bug, on purpose
        return omega * field_.ratio(side, self.name)

    def split(self, field_: TimeField, a: Rotor, b: Rotor) -> tuple[float, float]:
        """Torque to each output. Equal for an open diff; a bias term
        reads the rate difference, and reading it correctly is where the
        resample matters."""
        half = 0.5 * self.carrier_torque_nm
        if self.lsd_bias_nm_per_rad_s == 0.0:
            return half, half
        wa = self.side_rate_in_frame(field_, a.name, a.omega_rad_s)
        wb = self.side_rate_in_frame(field_, b.name, b.omega_rad_s)
        bias = self.lsd_bias_nm_per_rad_s * (wa - wb)
        return half - bias, half + bias


@dataclass
class CoupledRig:
    """Carrier -> differential -> two rotors, each in its own time zone."""

    diff: Differential
    a: Rotor
    b: Rotor
    field_: TimeField
    reference_dt_s: float = 1.0 / 60.0
    substeps: int = 16
    frame: int = 0

    @classmethod
    def build(cls, *, tau_a: float, tau_b: float, resample: bool = True,
              lsd: float = 0.0) -> "CoupledRig":
        a = Rotor("out_a", 0.30, 0.0, "differential")
        b = Rotor("out_b", 0.50, 0.0, "differential")
        diff = Differential("diff", carrier_torque_nm=20.0,
                            lsd_bias_nm_per_rad_s=lsd, resample_rates=resample)
        tf = TimeField.flat(["out_a", "out_b", "diff"])
        tf.log_tau[tf._index["out_a"]] = math.log(tau_a)
        tf.log_tau[tf._index["out_b"]] = math.log(tau_b)
        return cls(diff=diff, a=a, b=b, field_=tf)

    def step(self) -> None:
        self.frame += 1
        for rotor in (self.a, self.b):
            rotor._window = self.reference_dt_s * self.field_.velocity(rotor.name)
        for _ in range(self.substeps):
            t_a, t_b = self.diff.split(self.field_, self.a, self.b)
            for rotor, torque in ((self.a, t_a), (self.b, t_b)):
                step_s = rotor._window / self.substeps
                before = rotor.omega_rad_s
                rotor.drive_torque_nm = torque
                rotor.advance(step_s)
                # work is torque * ANGLE -- a scalar with no clock in it
                d_theta = 0.5 * (before + rotor.omega_rad_s) * step_s
                self.diff.delivered_work_j += torque * d_theta
                # the carrier turns the mean angle, and takes 2T
                self.diff.carrier_work_j += self.diff.carrier_torque_nm * 0.5 * d_theta
                rotor._d_theta = d_theta
            # the bias torque works against the angle DIFFERENCE: heat
            if self.diff.lsd_bias_nm_per_rad_s != 0.0:
                bias = 0.5 * self.diff.carrier_torque_nm - t_a
                self.diff.dissipated_j += bias * (
                    getattr(self.a, "_d_theta", 0.0) - getattr(self.b, "_d_theta", 0.0))
        for rotor in (self.a, self.b):
            rotor.world_s += rotor._window

    @property
    def ledger_error_j(self) -> float:
        """Carrier work in, minus work delivered to both outputs. Must be
        zero across ANY time gradient."""
        return (self.diff.carrier_work_j
                - self.diff.delivered_work_j
                - self.diff.dissipated_j)

    def line(self, label: str) -> str:
        return (f"{label:<26}"
                f"tau {self.field_.velocity('out_a'):.2f}/{self.field_.velocity('out_b'):.2f}"
                f"  w_a {self.a.omega_rad_s:7.3f}  w_b {self.b.omega_rad_s:7.3f}"
                f"  in {self.diff.carrier_work_j:9.4f} J"
                f"  out {self.diff.delivered_work_j:9.4f} J"
                f"  heat {self.diff.dissipated_j:8.4f} J"
                f"  err {self.ledger_error_j:10.3e} J")


def coupled_main() -> None:
    print("\n" + "=" * 118)
    print("COUPLED: carrier -> open differential -> two rotors, each on its own clock.")
    print("The ledger is carrier work in minus work delivered out. It must close")
    print("whatever the time gradient is, because the diff's constraint is on ANGLES.")
    print("=" * 118)
    for label, ta, tb in (("equal rates", 1.0, 1.0),
                          ("2:1 gradient", 1.0, 0.5),
                          ("10:1 gradient", 1.0, 0.1),
                          ("both dilated", 0.3, 0.7)):
        rig = CoupledRig.build(tau_a=ta, tau_b=tb)
        for _ in range(30):
            rig.step()
        print(rig.line(label))

    print("\nNow a LIMITED-SLIP bias, which reads the rate difference -- so the joint")
    print("genuinely needs each side resampled into its own frame:")
    ok = CoupledRig.build(tau_a=1.0, tau_b=0.25, lsd=0.4, resample=True)
    bad = CoupledRig.build(tau_a=1.0, tau_b=0.25, lsd=0.4, resample=False)
    for _ in range(30):
        ok.step(); bad.step()
    print(ok.line("LSD, resampled"))
    print(bad.line("LSD, NOT resampled"))
    print()
    print("Both ledgers now close -- and THAT is the warning. Conservation does")
    print("not catch the resampling error: the bookkeeping is self-consistent for")
    print("whatever bias you computed, right or wrong. The two rigs reach")
    print("different speeds from identical inputs (%.3f/%.3f vs %.3f/%.3f)"
          % (ok.a.omega_rad_s, ok.b.omega_rad_s, bad.a.omega_rad_s, bad.b.omega_rad_s))
    print("because the unresampled one compared two speeds quoted on different")
    print("clocks. An energy gate would accept both. Cross-rate correctness needs")
    print("its own check.")


if __name__ == "__main__":
    main()
    coupled_main()
