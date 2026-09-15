"""AIR MOVERS: a propeller, a jet and a cooling fan are one device.

Every rotor that pushes air does the same three things at once -- it
absorbs shaft torque, it throws a wake behind itself, and it pushes back.
Those are not three models. They are three readings of one actuator disk,
and `drivetrain_graph.rotor_aero_load_torque_nm` already computes the
middle one internally and throws it away:

    flow          Q = flow_coeff * omega
    exit velocity v = Q / disk_area          <- computed, then discarded
    shaft power   P = 0.5 * rho * Q * v**2
    torque        P / omega

This module keeps the wake and adds the reaction, so the same declared
numbers that already load a fan's shaft now also say how fast the air
behind it is moving and how hard it pushes. The torque is that same law
GENERALISED to a freestream -- at zero airspeed it returns exactly what
`rotor_aero_load_torque_nm` returns, to the last bit, so nothing that
uses it today changes.

WHY THAT MATTERS BEYOND TIDINESS

  * A cooling fan's whole purpose is its wake, and until now the wake
    was the one quantity not available.
  * A propeller is the same object at a different size, so thrust needs
    no separate engine kind.
  * The air behind a mover is what a radiator, an oil cooler, an
    intercooler or a bare hot casting actually sees, so passive cooling
    stops being a constant and starts being a consequence.

FREESTREAM IS THE OTHER HALF

A mover on a moving vehicle is not the same mover. Air already arriving
at speed changes the mass flow through the disk, the thrust it makes and
the power it costs. So every reading here takes a freestream speed, and
the local air a device sees is:

    ambient tile wind + vehicle ground speed + every wake it sits in

which is why a radiator behind a propeller is cooled by the propeller at
a standstill and by the airspeed in the cruise, with no special case for
either -- it is the same sum.

THE THIN AIR SIM THIS SETS UP

`AirField` is deliberately the smallest thing that can carry a
disturbance: a local mean plus a decaying turbulent component, fed by
whatever moves. Movers deposit their wake, moving bodies deposit theirs.
What it is FOR is not visual -- it is to make the air genuinely uncertain
where something has stirred it, so ballistics and targeting have to cope
with a crosswind that is real, local and unrepeatable rather than a
tuning constant. A shot through a propeller wash should be a worse shot.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

from drivetrain_graph import (
    AIR_DENSITY_KG_M3,
    rotor_aero_load_torque_nm,
)


@dataclass(frozen=True)
class AirMoverReading:
    """What one rotor is doing to the air, and what that costs it."""

    omega_rad_s: float
    freestream_m_s: float
    flow_m3_s: float
    #: velocity the rotor ADDS. The wake is freestream + this.
    induced_m_s: float
    wake_m_s: float
    thrust_n: float
    torque_nm: float

    @property
    def shaft_power_w(self) -> float:
        return self.torque_nm * self.omega_rad_s

    @property
    def useful_power_w(self) -> float:
        """Thrust times the speed it is delivered at. Zero at a
        standstill however hard the rotor works, which is the honest
        reason a static propeller is a heater and not a mover."""
        return self.thrust_n * self.freestream_m_s


def read_air_mover(*, omega_rad_s: float, flow_coeff_m3_s_per_rad_s: float,
                   disk_area_m2: float, freestream_m_s: float = 0.0,
                   density_kg_m3: float = AIR_DENSITY_KG_M3) -> AirMoverReading:
    """Everything one air mover is doing, from what it already declares.

    `flow_coeff_m3_s_per_rad_s` is the fan node's own
    ``fan_airflow_m3_s_per_rad_s``; `disk_area_m2` its swept area.

    Thrust is momentum added to the stream: mass flow times the velocity
    increment. At rest that is rho*A*v_induced**2, and it stays exact as
    the freestream rises because the mass flow rises with it.
    """
    if omega_rad_s <= 0.0 or flow_coeff_m3_s_per_rad_s <= 0.0 or disk_area_m2 <= 0.0:
        return AirMoverReading(max(omega_rad_s, 0.0), freestream_m_s,
                               0.0, 0.0, max(freestream_m_s, 0.0), 0.0, 0.0)
    induced = flow_coeff_m3_s_per_rad_s * omega_rad_s / disk_area_m2
    wake = float(freestream_m_s) + induced
    # mass flow through the disk includes what the freestream brings
    flow = disk_area_m2 * wake
    thrust = density_kg_m3 * flow * induced
    # THE TORQUE MUST RISE WITH THE FREESTREAM, or the rotor invents
    # energy. Calling rotor_aero_load_torque_nm directly gives the
    # STATIC answer at every airspeed: thrust grows with the mass flow
    # the freestream brings, shaft power does not, and a propeller at
    # 60 m/s reads 724 kW of useful output from a 14.6 kW shaft.
    #
    # The actuator disk does its work at the disk, which sees the
    # freestream plus half the increment it is adding:
    #     P = thrust * (v0 + v_induced / 2)
    # At v0 = 0 that is 0.5 * rho * A * v_i**3, which is EXACTLY what
    # rotor_aero_load_torque_nm returns -- so this generalises that
    # function rather than disagreeing with it, and every existing
    # caller (all of which are static) is unaffected.
    disk_velocity = float(freestream_m_s) + 0.5 * induced
    torque = thrust * disk_velocity / omega_rad_s
    return AirMoverReading(omega_rad_s, float(freestream_m_s), flow,
                           induced, wake, thrust, torque)


def disk_area_m2(diameter_m: float, hub_diameter_m: float = 0.0) -> float:
    """Swept area of a rotor, less whatever the hub blocks."""
    outer = math.pi * 0.25 * max(diameter_m, 0.0) ** 2
    inner = math.pi * 0.25 * max(hub_diameter_m, 0.0) ** 2
    return max(outer - inner, 0.0)


# ---------------------------------------------------------------------
# the air a device actually sits in
# ---------------------------------------------------------------------

@dataclass
class AirField:
    """Local air state: a mean the machine causes, and a turbulence it
    cannot predict.

    Kept deliberately thin. The mean is the sum of everything that moves
    air here; the turbulent part is what makes a disturbed patch of air
    genuinely uncertain rather than merely faster. Nothing in here is
    a flow solver and it is not trying to be one.
    """

    #: wind belonging to the ground itself. Not tracked anywhere yet, so
    #: it is an input rather than a lookup -- when tiles carry weather
    #: this is where it arrives.
    tile_wind_m_s: float = 0.0
    #: how fast the machine is moving through the air
    ground_speed_m_s: float = 0.0
    #: turbulent kinetic content, as an r.m.s. velocity. Decays.
    turbulence_m_s: float = 0.0
    #: how quickly a disturbance settles out, seconds
    settle_tau_s: float = 1.6
    #: wakes deposited this tick, name -> speed
    wakes: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    _rng: random.Random = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # seeded, because unrepeatable air must still be reproducible --
        # the same reason the engine sim is seeded per identity
        self._rng = random.Random(self.seed)

    @property
    def freestream_m_s(self) -> float:
        """What a stationary device on this machine feels before any
        mover behind or in front of it."""
        return abs(self.tile_wind_m_s) + abs(self.ground_speed_m_s)

    def deposit_wake(self, name: str, wake_m_s: float,
                     turbulent_frac: float = 0.12) -> None:
        """A mover puts its wake into the air here.

        Part of it is directed flow, which cools; part is turbulence,
        which does not cool much and does confound anything trying to
        fly through it. The split is a disclosed fraction, not a
        measurement.
        """
        speed = max(float(wake_m_s), 0.0)
        self.wakes[str(name)] = speed
        self.turbulence_m_s = math.hypot(self.turbulence_m_s,
                                         speed * float(turbulent_frac))

    def deposit_passage(self, speed_m_s: float, size_m: float) -> None:
        """A body moving through the air stirs it whether or not it is
        trying to. Scaled by how much air it displaces."""
        self.turbulence_m_s = math.hypot(
            self.turbulence_m_s,
            abs(float(speed_m_s)) * 0.08 * min(max(float(size_m), 0.0), 4.0))

    def speed_over(self, *behind: str) -> float:
        """The air speed a device sits in.

        Freestream plus the wakes it is actually behind -- so a radiator
        behind the fan is cooled by the fan at a standstill and by the
        airspeed in the cruise, out of one sum with no special case.
        Wakes combine by the larger rather than by addition: standing in
        two wakes does not give you their sum.
        """
        wake = max((self.wakes.get(name, 0.0) for name in behind), default=0.0)
        return self.freestream_m_s + wake

    def sample_crosswind(self) -> float:
        """One draw of the unpredictable part, for something in flight.

        This is the point of the turbulent term: a projectile crossing
        stirred air gets a real, local, unrepeatable push. Seeded, so a
        replay of the same shot through the same air lands in the same
        place.
        """
        if self.turbulence_m_s <= 0.0:
            return 0.0
        return self._rng.gauss(0.0, self.turbulence_m_s)

    def step(self, dt: float) -> None:
        """Let the disturbance settle and clear this tick's wakes."""
        if dt <= 0.0:
            return
        if self.settle_tau_s > 0.0:
            self.turbulence_m_s *= math.exp(-float(dt) / self.settle_tau_s)
        else:
            self.turbulence_m_s = 0.0
        self.wakes.clear()


def cooling_airflow_m3_s(field_: AirField, face_area_m2: float,
                         *behind: str) -> float:
    """Air actually passing through a cooling face.

    What a radiator, an oil cooler or a bare hot casting gets. It is the
    same quantity whether the air arrives from a fan, from a propeller
    ahead of it, or from the machine simply moving -- which is the whole
    reason to have one function for it.
    """
    return max(face_area_m2, 0.0) * field_.speed_over(*behind)


if __name__ == "__main__":
    # a cooling fan and a propeller, read through the same function
    FAN = dict(flow_coeff_m3_s_per_rad_s=0.0075, disk_area_m2=disk_area_m2(0.42))
    PROP = dict(flow_coeff_m3_s_per_rad_s=0.95, disk_area_m2=disk_area_m2(3.6, 0.35))

    print("the same reading, at two sizes and two airspeeds")
    print(f"{'device':<12}{'rpm':>7}{'v_air':>8}{'wake':>8}{'thrust':>10}{'torque':>10}{'shaft kW':>10}")
    print("-" * 65)
    for label, spec, rpm in (("cooling fan", FAN, 2600.0), ("propeller", PROP, 1350.0)):
        for v0 in (0.0, 60.0):
            r = read_air_mover(omega_rad_s=rpm * 0.10471975511965977,
                               freestream_m_s=v0, **spec)
            print(f"{label:<12}{rpm:>7.0f}{v0:>8.1f}{r.wake_m_s:>8.1f}"
                  f"{r.thrust_n:>10.0f}{r.torque_nm:>10.1f}{r.shaft_power_w/1000:>10.1f}")

    print()
    air = AirField(ground_speed_m_s=0.0, seed=7)
    prop = read_air_mover(omega_rad_s=1350 * 0.10471975511965977, **PROP)
    air.deposit_wake("propeller", prop.wake_m_s)
    radiator = cooling_airflow_m3_s(air, 0.30, "propeller")
    print(f"standstill: prop wake {prop.wake_m_s:.1f} m/s -> radiator sees "
          f"{radiator:.2f} m3/s through 0.30 m2")

    air_cruise = AirField(ground_speed_m_s=70.0, seed=7)
    air_cruise.deposit_wake("propeller", prop.wake_m_s)
    print(f"cruise 70 m/s: radiator sees "
          f"{cooling_airflow_m3_s(air_cruise, 0.30, 'propeller'):.2f} m3/s "
          f"-- same sum, no special case")

    print()
    air.deposit_passage(speed_m_s=70.0, size_m=2.5)
    draws = [air.sample_crosswind() for _ in range(6)]
    print("crosswind draws through the disturbed air (m/s):",
          ", ".join(f"{d:+.2f}" for d in draws))
    for _ in range(4):
        air.step(1.0)
    print(f"after 4 s of settling, turbulence {air.turbulence_m_s:.3f} m/s")
