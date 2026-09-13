"""A multi-pinion slew ring: several drives on one gear, chosen to suit
the conditions and whatever equipment is alive.

The arrangement asked for is real and it is what large mounts actually
use. A RING GEAR sits between two plates with the rolling elements
between them -- that sandwich IS the bearing, and the gear is cut into
one of its races, so the thing that carries the load and the thing that
transmits the drive are the same ring. Several PINIONS engage that ring
around its circumference. Wind turbine yaw drives, excavator houses,
radar and gun mounts are all built this way, and for four reasons that
matter separately:

  LOAD SHARING. N pinions each carry about 1/N of the torque, so the
  drive units shrink faster than the count grows. Three small drives
  beat one enormous one on cost, on packaging and on what you have to
  carry as a spare.

  ANTI-BACKLASH, which is the one people underrate. A single pinion in
  a ring gear has lash, and lash on a gun mount is pointing error you
  cannot control away -- the barrel simply sits somewhere within the
  slop. Preload two pinions AGAINST each other and the lash disappears,
  at the cost of a constant circulating torque that does no useful work
  and must be paid for continuously.

  REDUNDANCY. Lose a drive and the mount still turns, just slower. A
  single drive is a single point of failure on the one motion the mount
  exists to perform.

  MIXED TECHNOLOGY, which is what makes "run it for the conditions"
  possible. The pinions do not have to be the same kind of drive. An
  electric pinion is quiet, precise and efficient but limited in torque;
  a hydraulic one is brutal and fast but drags its pump along; a
  pneumatic one is inefficient but silent at the mount and safe in
  vapour. Engaging different subsets gives genuinely different mounts
  out of one ring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RingGear:
    """The ring, and the sandwich it lives in."""
    pitch_diameter_m: float = 1.10
    module_m: float = 0.008               # tooth size
    face_width_m: float = 0.060
    plate_gap_m: float = 0.075            # the space between the two plates
    backlash_m: float = 0.00025           # circumferential lash at the pitch line

    @property
    def teeth(self) -> int:
        return int(round(self.pitch_diameter_m / self.module_m))

    @property
    def radius_m(self) -> float:
        return self.pitch_diameter_m / 2.0

    @property
    def backlash_mrad(self) -> float:
        """Lash expressed as pointing error, which is the form that
        matters: a quarter of a millimetre at the pitch line of a metre
        ring is half a milliradian of barrel wander."""
        return self.backlash_m / max(self.radius_m, 1e-9) * 1000.0

    def tooth_capacity_nm(self, pinion_teeth: int) -> float:
        """Lewis bending capacity of the mesh, as the ring sees it.

        Real enough to size with: allowable bending stress on a tooth of
        this module and face width, times the pitch radius."""
        allowable_pa = 200e6
        lewis_y = 0.32                     # a 20 degree tooth of ordinary proportion
        tangential_n = allowable_pa * self.face_width_m * self.module_m * lewis_y
        return tangential_n * self.radius_m

    def describe(self) -> list[str]:
        return [f"  ring gear {self.pitch_diameter_m * 1000:.0f} mm pitch, {self.teeth} teeth, "
                f"module {self.module_m * 1000:.0f} mm, face {self.face_width_m * 1000:.0f} mm",
                f"    two plates {self.plate_gap_m * 1000:.0f} mm apart with the rolling elements "
                f"between them: the bearing and the gear are one ring",
                f"    lash {self.backlash_m * 1000:.2f} mm at the pitch line = "
                f"{self.backlash_mrad:.2f} mrad of pointing wander, unless preloaded out"]


@dataclass
class Pinion:
    """One drive on the ring."""
    identity: str
    technology: str                        # "electric" | "hydraulic" | "pneumatic"
    teeth: int = 18
    drive: object = None                   # ServoMotor / FluidMotor / whatever turns it
    stall_torque_nm: float = 120.0         # at the pinion shaft
    rated_rate_rad_s: float = 3.2
    efficiency: float = 0.80               # wall to pinion shaft
    engaged: bool = True
    healthy: bool = True
    noise_at_mount_dba: float = 52.0

    def ring_torque_nm(self, ring: RingGear) -> float:
        """What this pinion can put into the ring, limited by whichever
        gives first -- the drive or the tooth."""
        ratio = ring.teeth / max(self.teeth, 1)
        from_drive = self.stall_torque_nm * ratio * 0.97       # mesh loss
        return min(from_drive, ring.tooth_capacity_nm(self.teeth))

    def ring_rate_rad_s(self, ring: RingGear) -> float:
        return self.rated_rate_rad_s * max(self.teeth, 1) / max(ring.teeth, 1)


@dataclass
class SlewDrive:
    """The ring with all its pinions, and the logic that picks which to
    use."""
    ring: RingGear = field(default_factory=RingGear)
    pinions: list = field(default_factory=list)
    preload_pair: tuple = ()               # two identities held against each other
    preload_torque_nm: float = 0.0
    rotating_inertia_kg_m2: float = 180.0
    friction_torque_nm: float = 140.0      # the bearing's own drag, loaded

    # ---------------- what the current set can do ----------------
    @property
    def active(self) -> list:
        return [p for p in self.pinions if p.engaged and p.healthy]

    def torque_nm(self) -> float:
        """Load sharing is never perfect -- the ring flexes and the
        pinions do not share equally -- so a real multi-drive is derated
        as the count grows rather than scaling linearly."""
        act = self.active
        if not act:
            return 0.0
        share = 1.0 - 0.06 * (len(act) - 1)         # ~6 % penalty per extra drive
        total = sum(p.ring_torque_nm(self.ring) for p in act) * max(share, 0.6)
        return max(0.0, total - self.friction_torque_nm - self.preload_torque_nm)

    def rate_rad_s(self) -> float:
        """They are geared to one ring, so the slowest sets the pace."""
        act = self.active
        return min((p.ring_rate_rad_s(self.ring) for p in act), default=0.0)

    def acceleration_rad_s2(self) -> float:
        return self.torque_nm() / max(self.rotating_inertia_kg_m2, 1e-9)

    @property
    def effective_backlash_mrad(self) -> float:
        """Preloading two pinions against each other removes the lash
        entirely. It is the only way to remove it, and it is never
        free."""
        return 0.0 if self.preload_torque_nm > 0.0 else self.ring.backlash_mrad

    def noise_at_mount_dba(self) -> float:
        """Sound adds on power, not on level: two 60 dBA drives make
        63 dBA, not 120."""
        act = self.active
        if not act:
            return 0.0
        power = sum(10 ** (p.noise_at_mount_dba / 10.0) for p in act)
        return 10.0 * math.log10(power)

    def input_power_w(self, torque_nm: float, rate_rad_s: float) -> float:
        act = self.active
        if not act:
            return 0.0
        useful = abs(torque_nm * rate_rad_s)
        mean_efficiency = sum(p.efficiency for p in act) / len(act)
        circulating = abs(self.preload_torque_nm * rate_rad_s)
        return (useful + circulating) / max(mean_efficiency, 1e-6)

    # ---------------- choosing a set for the conditions ----------------
    def select(self, mode: str) -> list[str]:
        """Engage the pinions that suit the situation.

        This is the point of having more than one kind on the ring. The
        modes are not presets so much as answers to different questions:
        what is quiet, what is fast, what still works, what is safe near
        fuel."""
        wanted = {
            # everything that is alive: maximum torque and rate
            "urgent": lambda p: True,
            # quiet: no hydraulics, because the pump is the noise
            "silent": lambda p: p.technology in ("electric", "pneumatic"),
            # precise: electric only, and preloaded against lash
            "precision": lambda p: p.technology == "electric",
            # safe in vapour: nothing that can arc
            "hazardous-atmosphere": lambda p: p.technology in ("hydraulic", "pneumatic"),
            # economical: whatever wastes least
            "economy": lambda p: p.efficiency >= 0.7,
        }.get(mode, lambda p: True)
        for p in self.pinions:
            p.engaged = bool(wanted(p))
        # precision mode preloads a pair against each other to kill lash
        if mode == "precision" and len(self.active) >= 2:
            pair = tuple(p.identity for p in self.active[:2])
            self.preload_pair = pair
            self.preload_torque_nm = 0.12 * sum(
                p.ring_torque_nm(self.ring) for p in self.active[:2])
        else:
            self.preload_pair = ()
            self.preload_torque_nm = 0.0
        return [p.identity for p in self.active]

    def fail(self, identity: str) -> None:
        for p in self.pinions:
            if p.identity == identity:
                p.healthy = False

    # ---------------- reporting ----------------
    def describe(self) -> list[str]:
        out = list(self.ring.describe())
        for p in self.pinions:
            state = "engaged" if (p.engaged and p.healthy) else (
                "FAILED" if not p.healthy else "idle")
            out.append(f"    {p.identity:12s} {p.technology:10s} {p.teeth:2d}T  "
                       f"{p.ring_torque_nm(self.ring) / 1000:6.1f} kNm  "
                       f"{math.degrees(p.ring_rate_rad_s(self.ring)):5.1f} deg/s  {state}")
        rate = self.rate_rad_s()
        out.append(f"  -> {self.torque_nm() / 1000:6.1f} kNm at "
                   f"{math.degrees(rate):5.1f} deg/s, "
                   f"{math.degrees(self.acceleration_rad_s2()):5.1f} deg/s^2, "
                   f"lash {self.effective_backlash_mrad:.2f} mrad, "
                   f"{self.noise_at_mount_dba():4.1f} dBA, "
                   f"{self.input_power_w(self.torque_nm(), rate) / 1000:5.1f} kW in")
        return out


def standard_mixed_ring() -> SlewDrive:
    """A ring carrying three kinds of drive, so it can be run several
    ways from one piece of hardware."""
    ring = RingGear()
    return SlewDrive(ring=ring, pinions=[
        Pinion("elec-a", "electric", teeth=16, stall_torque_nm=95.0, rated_rate_rad_s=2.6,
               efficiency=0.82, noise_at_mount_dba=49.0),
        Pinion("elec-b", "electric", teeth=16, stall_torque_nm=95.0, rated_rate_rad_s=2.6,
               efficiency=0.82, noise_at_mount_dba=49.0),
        Pinion("hyd-a", "hydraulic", teeth=20, stall_torque_nm=260.0, rated_rate_rad_s=5.4,
               efficiency=0.58, noise_at_mount_dba=67.0),
        Pinion("pneu-a", "pneumatic", teeth=16, stall_torque_nm=70.0, rated_rate_rad_s=3.0,
               efficiency=0.14, noise_at_mount_dba=41.0),
    ])


# =====================================================================
#  A CAPTIVE PINION: ONE RING UNDER IT, OPTIONALLY ONE OVER IT
# =====================================================================
@dataclass
class CaptivePinionRing:
    """A pinion held IN its mesh instead of merely pressed against it --
    and, optionally, a second ring laid over the first with a gap
    between them so the pinion runs captive in the middle.

    Worth having as a parametric option because the second ring changes
    three different things at once, and which of them you were after
    decides how you set it up.

    THE SEPARATING FORCE IS THE PROBLEM. A gear mesh does not only push
    tangentially. At a 20 degree pressure angle every newton of useful
    tangential force comes with 0.36 N trying to shove the pinion
    radially out of mesh, and that force is entirely the carrier
    bearing's problem. It is why a single-ring drive's carrier is so
    much heavier than the torque alone suggests, and why a drive that is
    merely bolted near the ring walks out of mesh under shock.

    ONE RING ON TOP CANCELS IT. Mesh the same pinion into a second ring
    of the same pitch radius on the other side of the gap and the two
    separating forces point opposite ways. What reaches the carrier is
    the imbalance between the two meshes, not the whole of either --
    typically a seventh of it. That is the difference between a carrier
    sized for the separating load and a carrier sized for nothing much.

    IT ALSO STRADDLES THE SHAFT. With only a lower ring the pinion
    overhangs its bearing and the shaft is a cantilever: the bending
    moment is the resultant times the overhang. With plates above and
    below, the shaft is supported on both sides of the mesh and the
    moment is the resultant times a quarter of the gap. Same part, same
    load, a fraction of the bending -- the identical reason a barrel
    trunnion runs on bearings either side of the tube rather than
    cantilevered off one cheek.

    AND IF THE UPPER RING IS HELD TO THE FRAME INSTEAD, IT IS A
    DIFFERENT MACHINE. Declare `upper_ring_reference="frame"` and the
    pinion stops being a drive and becomes a PLANET between a fixed ring
    and a moving one. Reduction is then N_l / (N_l - N_u): one tooth of
    difference between the two rings is a hundred-and-something to one,
    in a package no taller than the gap. That is a real drive (Wolfrom,
    and every "differential planetary" slew unit), it is how you get
    enormous reduction flat, and its price is that the two rings must
    share a centre distance while carrying different tooth counts --
    paid for in profile shift, and in efficiency.

    Nothing here is chosen by inspecting a name. The arrangement is
    DECLARED -- an upper ring or no upper ring, referenced to the same
    body or to the frame -- and every number below follows from that.
    """
    identity: str = "captive-pinion"
    lower_ring: RingGear = field(default_factory=RingGear)
    upper_ring: "RingGear | None" = None
    upper_ring_reference: str = "same-body"   # "same-body" | "frame"
    pinion_teeth: int = 18
    pressure_angle_deg: float = 20.0
    plate_gap_m: float = 0.075                # the space the pinion runs in
    carrier_overhang_m: float = 0.085         # cantilever arm when unstraddled
    shaft_diameter_m: float = 0.045
    mesh_share_error: float = 0.15            # how unequal two real meshes are

    # ---------------------------------------------------------------
    @property
    def straddled(self) -> bool:
        """Two plates means bearings either side of the mesh."""
        return self.upper_ring is not None

    @property
    def is_planetary(self) -> bool:
        return self.upper_ring is not None and self.upper_ring_reference == "frame"

    @property
    def meshes(self) -> int:
        """How many meshes carry the drive. A frame-referenced upper
        ring is a reaction member, not a second drive path."""
        return 2 if (self.upper_ring is not None
                     and self.upper_ring_reference == "same-body") else 1

    # ---------------------------------------------------------------
    def separating_force_n(self, tangential_n: float) -> float:
        """Per mesh: the radial component trying to open the mesh."""
        return abs(tangential_n) * math.tan(math.radians(self.pressure_angle_deg))

    def carrier_radial_load_n(self, tangential_n: float) -> float:
        """What the carrier bearing actually has to hold.

        The tangential reaction IS the drive torque and never goes away.
        The separating force does, if there is something on the other
        side pushing back."""
        per_mesh = abs(tangential_n) / max(self.meshes, 1)
        sep = self.separating_force_n(per_mesh)
        if self.upper_ring is None:
            radial = sep
        else:
            # two meshes, opposed: only their imbalance escapes
            radial = sep * self.mesh_share_error
        return math.hypot(abs(tangential_n), radial)

    def shaft_bending_moment_nm(self, tangential_n: float) -> float:
        resultant = self.carrier_radial_load_n(tangential_n)
        arm = (self.plate_gap_m / 4.0) if self.straddled else self.carrier_overhang_m
        return resultant * arm

    def shaft_bending_stress_pa(self, tangential_n: float) -> float:
        d = max(self.shaft_diameter_m, 1e-6)
        section_modulus = math.pi * d ** 3 / 32.0
        return self.shaft_bending_moment_nm(tangential_n) / section_modulus

    def tooth_load_per_mesh_n(self, tangential_n: float) -> float:
        return abs(tangential_n) / max(self.meshes, 1)

    # ---------------------------------------------------------------
    def reduction(self) -> float:
        """Input revolutions per output revolution.

        A plain mesh unless the upper ring is grounded, in which case it
        is the two-ring differential and the DIFFERENCE in tooth count
        is the whole story."""
        if not self.is_planetary:
            return self.lower_ring.teeth / max(self.pinion_teeth, 1)
        n_l, n_u = self.lower_ring.teeth, self.upper_ring.teeth
        if n_l == n_u:
            return math.inf        # a locked drive: nothing moves at all
        return abs(n_l / (n_l - n_u))

    def required_profile_shift_m(self) -> float:
        """Two rings of different tooth count on ONE centre distance do
        not fit without help. The mismatch is half a module per tooth of
        difference, and profile shift is what absorbs it."""
        if self.upper_ring is None:
            return 0.0
        return abs(self.lower_ring.radius_m - self.upper_ring.radius_m)

    def efficiency(self) -> float:
        """A plain mesh is excellent. A high-reduction differential
        planetary is not: it recirculates power between two meshes that
        nearly cancel, and the loss scales with how nearly they do."""
        if not self.is_planetary:
            return 0.97
        i = self.reduction()
        if not math.isfinite(i):
            return 0.0
        return max(0.25, min(0.90, 0.97 - 0.0032 * i))

    # ---------------------------------------------------------------
    def describe(self, tangential_n: float = 12_000.0) -> list:
        bare = CaptivePinionRing(identity=self.identity, lower_ring=self.lower_ring,
                                 pinion_teeth=self.pinion_teeth,
                                 pressure_angle_deg=self.pressure_angle_deg,
                                 carrier_overhang_m=self.carrier_overhang_m,
                                 shaft_diameter_m=self.shaft_diameter_m)
        out = [f"  {self.identity}: pinion of {self.pinion_teeth} teeth, "
               f"{'captive between two rings' if self.straddled else 'on a single ring'}"]
        if self.straddled:
            out.append(f"    upper ring referenced to the {self.upper_ring_reference}, "
                       f"{self.plate_gap_m * 1000:.0f} mm gap, {self.meshes} driving mesh(es)")
        out.append(f"    at {tangential_n / 1000:.1f} kN tangential: carrier sees "
                   f"{self.carrier_radial_load_n(tangential_n) / 1000:.1f} kN "
                   f"(single ring: {bare.carrier_radial_load_n(tangential_n) / 1000:.1f} kN)")
        out.append(f"    shaft bending {self.shaft_bending_moment_nm(tangential_n):.0f} N.m "
                   f"= {self.shaft_bending_stress_pa(tangential_n) / 1e6:.0f} MPa "
                   f"({'straddled' if self.straddled else 'cantilevered'}; single ring: "
                   f"{bare.shaft_bending_stress_pa(tangential_n) / 1e6:.0f} MPa)")
        out.append(f"    tooth load {self.tooth_load_per_mesh_n(tangential_n) / 1000:.1f} kN "
                   f"per mesh across {self.meshes}")
        i = self.reduction()
        out.append(f"    reduction {i:.1f}:1 at {self.efficiency() * 100:.0f}% "
                   f"{'(differential planetary)' if self.is_planetary else '(plain ring mesh)'}")
        if self.is_planetary:
            out.append(f"    {self.lower_ring.teeth} vs {self.upper_ring.teeth} teeth on one "
                       f"centre distance: {self.required_profile_shift_m() * 1000:.1f} mm of "
                       f"profile shift buys the whole ratio")
        return out
