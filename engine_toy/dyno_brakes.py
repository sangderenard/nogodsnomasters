"""The two absorbers a real test cell actually uses, as real objects.

dyno_sizing already RATES an absorber -- how much torque and power it
must swallow, how heavy the drum is. What it does not carry is the thing
that decides whether a given rig can hold a given point at all: an
absorber's torque-versus-speed CHARACTERISTIC. Eddy-current and water
brakes have opposite ones, and the difference is the whole reason a cell
often has both.

Neither is a curve fit. Both fall out of one line of physics each.

EDDY CURRENT -- a conducting rotor dragged through a magnetic field.

    The field induces an EMF in the rotor proportional to speed and
    field strength, e ~ omega.B. That EMF drives current round a loop
    with real resistance AND real inductance, so |i| = omega.B /
    sqrt(R^2 + (omega.L)^2). Retarding torque goes as the in-phase part
    of B.i:

        T  ~  omega . B^2 . R / (R^2 + omega^2 L^2)
           =  (B^2/R) . omega / (1 + (omega/omega_c)^2),
                                          omega_c = R/L

    which peaks at exactly omega = omega_c. Writing T_max for that peak
    gives the standard form this class uses:

        T(omega) = 2 . T_max . (omega/omega_c) / (1 + (omega/omega_c)^2)

    Two real consequences, both of which matter and neither of which is
    a limitation anyone chose:

      - IT MAKES NO TORQUE AT ZERO SPEED. T -> 0 as omega -> 0, because
        a stationary rotor induces nothing. An eddy brake physically
        cannot hold a stalled engine, and cannot start a pull from rest.
      - IT FADES ABOVE ITS CRITICAL SPEED, falling off as 1/omega, since
        the rotor's own induced field increasingly opposes penetration.
        A brake sized for peak torque at 2000 rpm has HALF of it left
        near 5000.

    Control is the field current, and torque goes with the SQUARE of it
    because torque goes with B^2. Half current is a quarter of torque,
    not half -- which is why an eddy cell's control feels non-linear.

WATER BRAKE -- a rotor churning water against a stator.

    Momentum exchange with a fluid, so it obeys the same affinity law
    every pump and fan obeys:

        T = K . rho . D^5 . omega^2        and so    P = T.omega ~ omega^3

    Control is the FILL: how much of the working chamber holds water.
    An empty chamber makes almost nothing; a full one makes everything
    the casing can take.

    Its consequences are the mirror image of the eddy brake's:

      - ALSO NOTHING AT ZERO SPEED, and worse than the eddy brake just
        above it, since torque falls as omega^2 rather than omega.
      - BUT IT NEVER FADES. Torque climbs without limit until something
        mechanical gives, which is why the biggest absorbers in the
        world are water brakes and why they are what you put in front
        of a large marine diesel.

BOTH TURN EVERY WATT INTO HEAT. An absorber is a device for destroying
mechanical power, so absorbed power IS thermal power, exactly, with no
efficiency to argue about. What differs is where it goes: the eddy
brake's heat appears in the rotor iron and has to be dragged out through
its surface, while the water brake's heat leaves dissolved in the water
it already uses as its working fluid. That asymmetry is why an eddy
brake is the one that gets thermally limited first, and it is computed
here rather than asserted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Water's real density and specific heat, for the outlet-temperature
#: calculation. Not tuning constants -- the fluid's own properties.
WATER_DENSITY_KG_M3 = 998.0
WATER_SPECIFIC_HEAT_J_PER_KGK = 4182.0

RPM_TO_RAD_S = 2.0 * math.pi / 60.0


@dataclass
class AbsorberPoint:
    """What the brake is doing at one operating point."""
    torque_nm: float
    power_w: float
    control: float          # 0..1, whatever this brake's control variable is
    saturated: bool         # control is at 1.0 and it still is not enough
    heat_w: float           # identical to power_w -- an absorber has no other output
    note: str = ""


class Absorber:
    """Shared contract: give it a speed and a control setting, it tells
    you the torque it makes. `hold` inverts that -- what control setting
    is needed for a wanted torque, and whether it is even reachable."""

    identity: str = "absorber"

    def torque_nm(self, rpm: float, control: float) -> float:
        raise NotImplementedError

    def capacity_nm(self, rpm: float) -> float:
        """Most torque available at this speed, control wide open."""
        return self.torque_nm(rpm, 1.0)

    def at(self, rpm: float, control: float) -> AbsorberPoint:
        t = self.torque_nm(rpm, control)
        p = t * max(0.0, rpm) * RPM_TO_RAD_S
        return AbsorberPoint(torque_nm=t, power_w=p, control=control,
                             saturated=control >= 1.0, heat_w=p)

    def hold(self, rpm: float, wanted_nm: float) -> AbsorberPoint:
        """The control setting that produces `wanted_nm` at this speed.

        Returns a SATURATED point when the brake cannot reach it, rather
        than silently clamping -- a rig that quietly runs out of
        absorber reads as an engine that stopped making power, which is
        exactly the misreading dyno_sizing.consumes_range exists to
        prevent."""
        cap = self.capacity_nm(rpm)
        if wanted_nm <= 0.0:
            return self.at(rpm, 0.0)
        if cap <= 1e-9:
            p = self.at(rpm, 1.0)
            p.saturated = True
            p.note = (f"{self.identity} makes no torque at {rpm:.0f} rpm -- "
                      "neither absorber type can hold a point this slow")
            return p
        if wanted_nm >= cap:
            p = self.at(rpm, 1.0)
            p.saturated = True
            p.note = (f"{self.identity} saturated: {cap:.0f} Nm available at "
                      f"{rpm:.0f} rpm, {wanted_nm:.0f} Nm wanted")
            return p
        return self.at(rpm, self._control_for(rpm, wanted_nm, cap))

    def _control_for(self, rpm: float, wanted_nm: float, cap: float) -> float:
        raise NotImplementedError


@dataclass
class EddyCurrentBrake(Absorber):
    """A conducting rotor in a controlled magnetic field.

    `peak_torque_nm` is the torque at `critical_rpm` with the field at
    full excitation -- the top of the characteristic, not a rating that
    applies everywhere. Away from that speed the brake genuinely has
    less, and `capacity_nm` says how much."""
    identity: str = "eddy-current"
    peak_torque_nm: float = 500.0
    critical_rpm: float = 2000.0
    #: the rotor's own thermal path to ambient. An eddy brake's heat is
    #: made INSIDE the iron, so this is what limits a long hold.
    rotor_thermal_conductance_w_per_k: float = 300.0
    rotor_max_temp_k: float = 573.15     # ~300 C, past which the iron's magnetic properties go

    def torque_nm(self, rpm: float, control: float) -> float:
        """The derived characteristic, with excitation squared.

        control is the field current fraction, and torque goes with B^2,
        so it enters squared. This is the real non-linearity of an eddy
        cell's control, not a smoothing choice."""
        w = max(0.0, float(rpm)) / max(self.critical_rpm, 1e-9)
        c = max(0.0, min(1.0, float(control)))
        return self.peak_torque_nm * (c * c) * 2.0 * w / (1.0 + w * w)

    def _control_for(self, rpm: float, wanted_nm: float, cap: float) -> float:
        # torque is linear in c^2, so the inverse is an exact square root
        return math.sqrt(max(0.0, min(1.0, wanted_nm / cap)))

    def rotor_temp_rise_k(self, power_w: float) -> float:
        """Steady-state rotor temperature rise for a sustained hold.

        Every absorbed watt becomes heat in the iron, and leaves only
        through the rotor's own conductance to ambient, so the steady
        rise is simply Q/UA. This is the number that decides whether a
        point can be HELD or only swept through."""
        return max(0.0, power_w) / max(self.rotor_thermal_conductance_w_per_k, 1e-9)

    def thermally_limited(self, power_w: float, ambient_k: float = 293.15) -> bool:
        return ambient_k + self.rotor_temp_rise_k(power_w) > self.rotor_max_temp_k


@dataclass
class WaterBrake(Absorber):
    """A rotor churning water, obeying the pump affinity law.

    `torque_coefficient` bundles K.rho.D^5 into the one number the
    geometry actually fixes, because on a real brake it is a single
    measured constant for that casing and nobody separates it. It is
    solved from a rated point instead of being typed in -- see
    `for_engine`."""
    identity: str = "water-brake"
    torque_coefficient_nm_per_rad_s2: float = 0.0
    #: cooling water actually flowing through it. All absorbed power
    #: leaves in this stream, so the flow sets the outlet temperature
    #: and the outlet temperature is what limits the brake.
    water_flow_kg_s: float = 2.0
    water_inlet_k: float = 288.15
    water_boiling_k: float = 373.15

    def torque_nm(self, rpm: float, control: float) -> float:
        """T = K.rho.D^5.omega^2, scaled by fill fraction.

        Fill enters linearly: it is how much of the working chamber is
        doing the churning, and half a chamber exchanges momentum with
        half the water."""
        omega = max(0.0, float(rpm)) * RPM_TO_RAD_S
        fill = max(0.0, min(1.0, float(control)))
        return self.torque_coefficient_nm_per_rad_s2 * omega * omega * fill

    def _control_for(self, rpm: float, wanted_nm: float, cap: float) -> float:
        # linear in fill, so the inverse is exact
        return max(0.0, min(1.0, wanted_nm / cap))

    def water_outlet_k(self, power_w: float) -> float:
        """Where the heat goes: straight into the water it already uses.

        A steady-flow energy balance, Q = m.cp.dT, with no efficiency
        term because an absorber converts all of it."""
        rise = max(0.0, power_w) / max(self.water_flow_kg_s * WATER_SPECIFIC_HEAT_J_PER_KGK, 1e-9)
        return self.water_inlet_k + rise

    def boiling(self, power_w: float) -> bool:
        """A water brake that boils stops being a water brake: the
        chamber fills with steam, which exchanges almost no momentum,
        and the load collapses. Real failure mode, real limit."""
        return self.water_outlet_k(power_w) >= self.water_boiling_k


def for_engine(engine, kind: str = "eddy-current",
               gear_ratio: float = 1.0) -> Absorber:
    """An absorber actually sized to this engine, from the engine's own
    declared envelope rather than from a catalogue of rigs.

    gear_ratio IS NOT OPTIONAL IN PRACTICE, only in the signature. An
    absorber coupled through a gearbox sees the engine's torque
    MULTIPLIED by the overall ratio and its speed DIVIDED by it -- on
    the AMC in fourth that is 3.84, so a brake sized for 421 Nm at the
    crank is asked for 1600 Nm at the drum and saturates at every point
    in the sweep. Sizing it at the crank and then using it at the drum
    is the same class of error as comparing a drum torque reading
    against a catalogue crank figure.

    The sizing rule is the same for both and it is not arbitrary: the
    brake must hold the engine's peak torque at the speed the engine
    makes it, with margin, or the cell cannot measure its own subject.
    Where the two differ is what that requirement implies about the rest
    of the range, which is the entire point of having both."""
    g = max(float(gear_ratio), 1e-6)
    # torque up, speed down, exactly as the gearing does it
    peak_nm = float(getattr(engine, "peak_torque_nm", 0.0) or 0.0) * g
    tpeak_rpm = float(getattr(engine, "torque_peak_rpm", 0.0) or 0.0) / g
    redline = float(getattr(engine, "redline_rpm", 0.0) or 0.0) / g
    margin = 1.5   # a cell is built with headroom; a brake at its own limit cannot control
    if kind == "eddy-current":
        # Put the critical speed AT the engine's torque peak, which is
        # where the most torque is needed. That is the real design
        # choice an eddy cell makes, and it is also why such a cell then
        # reads soft at the top of the range -- a real, visible
        # consequence of a real decision rather than a modelled flaw.
        return EddyCurrentBrake(peak_torque_nm=peak_nm * margin,
                                critical_rpm=max(tpeak_rpm, 1.0),
                                rotor_thermal_conductance_w_per_k=max(
                                    300.0, peak_nm * 2.0))
    if kind == "water-brake":
        # Solve the coefficient from the requirement directly: it must
        # make peak_nm * margin at the torque peak. One equation, one
        # unknown, no fitting.
        omega = max(tpeak_rpm, 1.0) * RPM_TO_RAD_S
        k = peak_nm * margin / (omega * omega)
        # water flow sized to carry the engine's own peak power without
        # boiling, which is the real constraint on the plumbing
        peak_w = peak_nm * max(redline, tpeak_rpm) * RPM_TO_RAD_S
        flow = peak_w / (WATER_SPECIFIC_HEAT_J_PER_KGK * 60.0)   # 60 K rise budget
        return WaterBrake(torque_coefficient_nm_per_rad_s2=k,
                          water_flow_kg_s=max(0.5, flow))
    raise KeyError(f"unknown absorber {kind!r}; declared: eddy-current, water-brake")


def compare(engine, rpms=None) -> str:
    """Both brakes against one engine, which is the honest way to see
    why a cell keeps two."""
    import engine_sim as es
    eddy = for_engine(engine, "eddy-current")
    water = for_engine(engine, "water-brake")
    if rpms is None:
        hi = float(getattr(engine, "redline_rpm", 4000.0) or 4000.0)
        rpms = [hi * f for f in (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)]
    peak = float(getattr(engine, "peak_torque_nm", 1.0) or 1.0)
    lines = [f"absorber capacity vs {getattr(engine, 'identity', '?')}",
             f"  {'rpm':>6s} {'engine Nm':>10s} {'eddy cap':>10s} {'water cap':>10s}"
             f"   {'eddy':>6s} {'water':>6s}"]
    for r in rpms:
        need = peak * es.torque_fraction(engine, r)
        e_cap, w_cap = eddy.capacity_nm(r), water.capacity_nm(r)
        lines.append(f"  {r:6.0f} {need:10.1f} {e_cap:10.1f} {w_cap:10.1f}"
                     f"   {'ok' if e_cap >= need else 'SHORT':>6s}"
                     f" {'ok' if w_cap >= need else 'SHORT':>6s}")
    lines.append("")
    lines.append("  Neither holds the bottom of the range: both make zero torque at "
                 "zero speed, the")
    lines.append("  eddy brake falling off as omega and the water brake as omega^2, "
                 "so the water")
    lines.append("  brake gives up sooner. Above its critical speed the eddy brake "
                 "fades while the")
    lines.append("  water brake keeps climbing -- which is the real reason a cell "
                 "testing across a")
    lines.append("  wide range keeps one of each rather than a bigger one of either.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import engines
    print(compare(engines.get(sys.argv[1] if len(sys.argv) > 1 else "amc-258-jeep-i6")))
