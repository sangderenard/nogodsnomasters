"""Mechanical torque-coupling ports, grounded directly in the real
game's code (turing/src/compiler/abstract_ui_vehicles.py:2313-2356 and
mechanical_ports.py:116-159), and one deliberate, honestly-labeled
extension past it.

The real file actually has TWO torque paths across the engine/input-
shaft coupling, not one:

  clutch_torque              -- the friction disc: clutch_maximum_torque
                                 * tanh(clutch_stiffness * relative_speed
                                        / clutch_maximum_torque). Pure
                                 relative-speed-driven saturating slip,
                                 no backlash -- a friction disc doesn't
                                 have gear lash, it just slips.
  direct_drive_bypass_torque -- a direct-drive DOG-clutch bypass path
                                 (also a real edge:
                                 "synchronized-positive-dog-clutch-bypass",
                                 with a real tooth_health_coordinate and
                                 an interlock requiring low relative
                                 speed AND unloaded teeth before it's
                                 even allowed to engage). Real dog teeth
                                 have physical clearance before they
                                 catch -- that IS backlash, and neither
                                 source spells out its formula.

`ClutchPort.backlash_rad` (default 0.0) reproduces the friction-disc
formula exactly when off, and is the honest, motivated extension for
the dog-bypass path (or any other rotational-play connection: splined
slip yokes, U-joint articulation slop) when turned on -- a genuine
proposal for what the real system is currently missing, not a
deviation dressed up as a match.

mechanical_ports.py's `generic_rotational_torque_port` is a port
*schema* (identity/owner/axis/rated torque & speed/state variable
names/a "law" string) used to describe a boundary for validators and
compiled-graph binding -- it doesn't carry the physics itself. The
physics lives in abstract_ui_vehicles.py, which is what ClutchPort
mirrors.

`GearedCoupling` below is this toy's own model for the mechanisms
INSIDE the engine that genuinely are belt/gear/chain driven in real
life -- a supercharger drive belt, a camshaft timing chain or gear set
-- which the real game's vehicle graph never had a reason to cover: it
treats the whole engine as one opaque "powertrain.engine" node with
torque channels in and out, not decomposed into camshaft/supercharger-
drive internals. So there's no real-game system to defer to for those;
this is genuinely this toy's own physics for a real mechanism, same as
ClutchPort's backlash extension above. What the real game DOES cover,
and what GearedCoupling must NOT be used for, is the accessory
alternator/compressor/motor takeoff off the crank -- that one is a
real edge (drive="no-belt", a direct rigid shaft), see
mechanical_graph.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class ClutchPort:
    """`stiffness_nm_per_rad_s` is the source's `clutch_stiffness`
    naming carried over even though it's a gain against relative
    angular *speed*, not position -- that's the real code's own
    vocabulary. `backlash_rad=0` (default) reproduces the real
    friction-disc formula exactly; set it nonzero to model a
    connection that genuinely has rotational play before it engages
    (a dog-clutch bypass, a splined slip yoke)."""
    stiffness_nm_per_rad_s: float = 4000.0
    max_torque_nm: float = 500.0
    engagement: float = 1.0   # 0..1, e.g. a slipping/partially-engaged clutch
    backlash_rad: float = 0.0

    relative_angle_rad: float = field(default=0.0, init=False)
    last_torque_nm: float = field(default=0.0, init=False)
    engaged: bool = field(default=True, init=False)

    def step(self, dt: float, omega_drive: float, omega_load: float) -> float:
        relative_speed = omega_drive - omega_load

        if self.backlash_rad > 0.0:
            self.relative_angle_rad += relative_speed * dt
            gap = self.backlash_rad
            self.engaged = self.relative_angle_rad > gap or self.relative_angle_rad < -gap
        else:
            self.engaged = True

        if not self.engaged:
            torque = 0.0
        else:
            torque = self.engagement * self.max_torque_nm * math.tanh(
                self.stiffness_nm_per_rad_s * relative_speed / max(self.max_torque_nm, 1e-6))
        self.last_torque_nm = torque
        return torque

    @property
    def locked(self) -> bool:
        """Near enough zero relative speed that the coupling is
        essentially rigid rather than actively slipping."""
        return abs(self.last_torque_nm) < 0.02 * self.max_torque_nm

    @property
    def slack_frac(self) -> float:
        """0 at dead center of the backlash gap, 1 at either edge.
        Meaningless (stays 0) when backlash_rad == 0."""
        if self.backlash_rad <= 0.0:
            return 0.0
        return min(1.0, abs(self.relative_angle_rad) / self.backlash_rad)


@dataclass
class GearedCoupling:
    """A toothed/pulley connection: a real backlash gap (no contact,
    zero torque) followed by a spring/damper on the ENGAGED portion of
    relative angle once contact is made. For the mechanisms INSIDE an
    engine that genuinely mesh or ride a belt -- a supercharger drive
    belt, a camshaft timing chain/gear set -- not for the accessory
    alternator takeoff (that one's a real direct shaft, no belt, see
    mechanical_graph.py)."""
    stiffness_nm_per_rad: float = 150.0
    damping_nm_per_rad_s: float = 3.0
    max_torque_nm: float = 15.0
    backlash_rad: float = 0.01

    relative_angle_rad: float = field(default=0.0, init=False)
    last_torque_nm: float = field(default=0.0, init=False)
    engaged: bool = field(default=False, init=False)

    def step(self, dt: float, omega_drive: float, omega_load: float) -> float:
        relative_speed = omega_drive - omega_load
        self.relative_angle_rad += relative_speed * dt

        gap = self.backlash_rad
        if self.relative_angle_rad > gap:
            engaged_angle = self.relative_angle_rad - gap
            self.engaged = True
        elif self.relative_angle_rad < -gap:
            engaged_angle = self.relative_angle_rad + gap
            self.engaged = True
        else:
            engaged_angle = 0.0
            self.engaged = False

        if self.engaged:
            raw = self.stiffness_nm_per_rad * engaged_angle + self.damping_nm_per_rad_s * relative_speed
            torque = self.max_torque_nm * math.tanh(raw / max(self.max_torque_nm, 1e-6))
        else:
            torque = 0.0
        self.last_torque_nm = torque
        return torque

    @property
    def slack_frac(self) -> float:
        return min(1.0, abs(self.relative_angle_rad) / max(self.backlash_rad, 1e-9))

    def reset(self) -> None:
        self.relative_angle_rad = 0.0
        self.last_torque_nm = 0.0
        self.engaged = False


@dataclass
class GearedCoupling:
    """A toothed/pulley connection: a real backlash gap (no contact,
    zero torque) followed by a spring/damper on the ENGAGED portion of
    relative angle once contact is made. This is not lifted from the
    real game's file -- it's this toy's own model for the one kind of
    mechanical connection (gears, belts) that genuinely has lash,
    written the same way real engineering treats it: dead zone, then
    stiffness+damping."""
    stiffness_nm_per_rad: float = 150.0
    damping_nm_per_rad_s: float = 3.0
    max_torque_nm: float = 15.0
    backlash_rad: float = 0.01

    relative_angle_rad: float = field(default=0.0, init=False)
    last_torque_nm: float = field(default=0.0, init=False)
    engaged: bool = field(default=False, init=False)

    def step(self, dt: float, omega_drive: float, omega_load: float) -> float:
        relative_speed = omega_drive - omega_load
        self.relative_angle_rad += relative_speed * dt

        gap = self.backlash_rad
        if self.relative_angle_rad > gap:
            engaged_angle = self.relative_angle_rad - gap
            self.engaged = True
        elif self.relative_angle_rad < -gap:
            engaged_angle = self.relative_angle_rad + gap
            self.engaged = True
        else:
            engaged_angle = 0.0
            self.engaged = False

        if self.engaged:
            raw = self.stiffness_nm_per_rad * engaged_angle + self.damping_nm_per_rad_s * relative_speed
            torque = self.max_torque_nm * math.tanh(raw / max(self.max_torque_nm, 1e-6))
        else:
            torque = 0.0
        self.last_torque_nm = torque
        return torque

    @property
    def slack_frac(self) -> float:
        return min(1.0, abs(self.relative_angle_rad) / max(self.backlash_rad, 1e-9))

    def reset(self) -> None:
        self.relative_angle_rad = 0.0
        self.last_torque_nm = 0.0
        self.engaged = False


@dataclass
class RackPinionClutch:
    """A real rack-and-pinion, WITH force -- the one thing turing's own
    steering rack (abstract_ui_vehicles.py's `pinion_radius_m` /
    state_loop_deployment.py's `updateVehicleSteeringWrench`) doesn't
    carry: that system tracks rack TRAVEL from a commanded column angle
    (a kinematic position solve) but never turns a rack force into a
    pinion torque or feeds a reaction back -- there's no real F=T/r
    coupling in it at all. This is that coupling, genuinely bidirectional
    (rack force <-> pinion torque, rack velocity <-> pinion omega,
    exact real gear geometry: T = F * r, omega = v / r), plus a second,
    separable stage most rack-and-pinions DON'T have but a few real
    mechanisms genuinely do: a one-way (overrunning/sprag) clutch
    between the pinion and whatever it drives -- the real mechanism a
    bicycle freewheel, a ratchet-and-pawl hand winch, or (what this was
    built for) an Otto-Langen atmospheric engine's power take-off all
    use: the pinion drives the output shaft only while it's trying to
    turn it FASTER in the allowed direction than the shaft is already
    going; otherwise it overruns freely, transmitting zero torque and
    (Newton's third law) exerting zero reaction force back on the rack.

    Modeled as a real, very stiff, one-directional ClutchPort (the same
    honest simplification ClutchPort's own docstring already uses for
    the dog-clutch bypass: idealizing a mechanically-locking coupling as
    a stiff saturating one, rather than inventing a separate rigid-
    constraint solver) -- `power_sign` picks which sign of
    (pinion_omega - shaft_omega) is the driving direction; the other
    sign always reads zero torque, exactly the real overrun."""
    pinion_radius_m: float
    power_sign: int = 1          # +1: pinion drives shaft only while spinning it forward faster; -1: reversed
    max_torque_nm: float = 1e6   # a real sprag/ratchet's engaged stiffness is effectively rigid up to its rating
    stiffness_nm_per_rad_s: float = 1e6

    _clutch: ClutchPort = field(init=False)

    def __post_init__(self) -> None:
        self._clutch = ClutchPort(stiffness_nm_per_rad_s=self.stiffness_nm_per_rad_s,
                                  max_torque_nm=self.max_torque_nm)

    def pinion_omega(self, rack_velocity_m_s: float) -> float:
        return rack_velocity_m_s / self.pinion_radius_m

    def step(self, dt: float, rack_velocity_m_s: float, shaft_omega: float) -> tuple[float, float]:
        """Returns (rack_reaction_force_n, shaft_torque_nm). Steps the
        internal one-way clutch on the (pinion, shaft) pair, gated to
        `power_sign`, then converts the transmitted torque back to a
        linear reaction force on the rack via the same real radius --
        so a driven flywheel genuinely loads the piston/rack during the
        engaged stroke, and genuinely doesn't during overrun."""
        omega_pinion = self.pinion_omega(rack_velocity_m_s)
        relative = self.power_sign * (omega_pinion - shaft_omega)
        if relative <= 0.0:
            # overrunning: the pinion isn't trying to drive the shaft
            # forward at all in the allowed sense -- no engagement,
            # exactly the real freewheel (a stiff-but-zero-relative-
            # speed ClutchPort call would otherwise still report a
            # transient torque as relative crosses zero; a one-way
            # clutch has no torque capacity in this direction at all)
            self._clutch.engaged = False
            self._clutch.last_torque_nm = 0.0
            return 0.0, 0.0
        shaft_torque_nm = self.power_sign * self._clutch.step(
            dt, self.power_sign * omega_pinion, self.power_sign * shaft_omega)
        rack_reaction_force_n = -shaft_torque_nm / self.pinion_radius_m
        return rack_reaction_force_n, shaft_torque_nm

    @property
    def engaged(self) -> bool:
        return self._clutch.engaged
