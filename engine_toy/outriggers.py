"""The outrigger legs, on the real hydraulics.

WHAT THIS REPLACES. The first version of "deploy the outriggers" added
the stroke to every node's y coordinate and added the same number to the
jack's rest length. That is a moved mesh. Nothing was pumped, nothing
was loaded, nothing took any time, and the legs would have "extended"
identically with the engine off and the reservoir empty.

So the legs run on what the project already has and what every other
hydraulic thing in it runs on:

  `bench.HydraulicPowerUnit`  a real pump: displacement times speed is
                              the flow, the motor's power limits that
                              flow at pressure, the relief dumps the
                              rest as heat, and the reservoir is finite.
  `actuators.LinearActuator`  a real cylinder: bore and rod give the
                              two areas, force is pressure times area
                              minus seal friction, SPEED IS FLOW
                              DIVIDED BY AREA, and it stops at the end
                              of its stroke.

That gives the answers that were being asserted before: how long it
takes to stand up, whether the pump can hold the machine at all, what
happens when four legs are asked for at once (they share one pump, so
they go up at a quarter speed), and what it costs in heat.

FOUR LEGS ON ONE PUMP IS THE INTERESTING PART. The load is the
machine's weight, so each leg needs its quarter of it -- but the
pressure the pump has to make is set by the WORST leg, not the average,
and all four see that pressure. Raise them together on soft ground and
one leg sinks, its pressure drops, the others take its share, and the
machine goes over. That is why real outrigger sets are levelled leg by
leg, and it is why this reports per-leg pressure rather than a single
"deployed" flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from actuators import LinearActuator, BREAKAWAY_MULTIPLIER
from bench import HydraulicPowerUnit, ATM_PA
from hydraulic_losses import CircuitLosses

#: default leg names; a caller with a different leg set passes its own.
#: Assuming four was fine until the base grew a mid-edge leg on each
#: side, at which point "a quarter of the machine per leg" was wrong by
#: a factor of two in the pump sizing.
CORNERS = ("nn", "np", "pn", "pp")


@dataclass
class OutriggerLeg:
    """One leg: a cylinder, the share of the machine it holds up, and
    the ground it is standing on."""
    identity: str
    cylinder: LinearActuator
    share_kg: float
    pad_area_m2: float = 0.36 * 0.36
    ground_bearing_pa: float = 250_000.0     # firm soil; tarmac is ~10x
    #: live
    sunk_m: float = field(default=0.0, init=False)
    last_pressure: dict = field(default_factory=dict, init=False)
    on_ground: bool = field(default=False, init=False)

    @property
    def load_n(self) -> float:
        return self.share_kg * 9.81

    @property
    def pad_pressure_pa(self) -> float:
        return self.load_n / max(self.pad_area_m2, 1e-9)

    @property
    def sinks(self) -> bool:
        """A pad pushing harder than the ground can carry goes into it.
        The pad is sized by this and by nothing else."""
        return self.pad_pressure_pa > self.ground_bearing_pa

    @property
    def extension_m(self) -> float:
        return self.cylinder.position_m


@dataclass
class OutriggerSet:
    """Four legs on one power pack."""
    machine_mass_kg: float
    bore_m: float = 0.100
    rod_m: float = 0.060
    stroke_m: float = 1.300
    #: A TELESCOPIC CYLINDER IS ONE ACTUATOR WITH STAGES. The stages
    #: are in SERIES: total travel is the sum of them, and the
    #: `LinearActuator` already models what that does to force and
    #: speed -- a later stage has less area, so it is faster and
    #: weaker on the same flow and pressure. Reading one stage's
    #: stroke as the whole leg's travel halves the lift and is exactly
    #: the kind of quiet wrong answer this is meant to stop.
    stages: int = 1
    pad_area_m2: float = 0.36 * 0.36
    ground_bearing_pa: float = 250_000.0
    #: how long you are prepared to spend standing up, and what the
    #: circuit is expected to run at -- these two size the pack
    target_deploy_s: float = 75.0
    expected_pressure_pa: float = 17.0e6
    hpu: HydraulicPowerUnit = None
    #: THE PLUMBING, which was free until now. See hydraulic_losses.
    losses: CircuitLosses = None
    leg_names: tuple = CORNERS
    legs: dict = field(default_factory=dict)
    elapsed_s: float = field(default=0.0, init=False)
    electrical_w: float = field(default=0.0, init=False)
    electrical_wh: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.losses is None:
            self.losses = CircuitLosses()
        if self.hpu is None:
            # the pack that lives in the bay: small, because it only has
            # to stand the machine up occasionally and it shares the
            # reservoir with the traverse and elevation circuits
            # THE PACK IS SIZED BY HOW LONG YOU ARE WILLING TO STAND
            # THERE. Deploy time is stroke divided by rod speed, rod
            # speed is flow divided by area, and the flow is shared
            # between every leg at once -- so doubling the legs doubles
            # the time on the same pump. Eight legs on the old 16 cc
            # pack took six minutes, which is six minutes of standing
            # still in the open with the gun unable to fire.
            area = math.pi * (self.bore_m / 2.0) ** 2
            want_l_min = (area * self.stroke_m / max(self.target_deploy_s, 1.0)
                          * 60_000.0 * len(CORNERS if not self.leg_names
                                           else self.leg_names))
            cc = want_l_min * 1000.0 / (1450.0 * 0.93)
            # and the motor has to drive that flow at working pressure
            kw = want_l_min * self.expected_pressure_pa / 60_000.0 / 0.80 / 1000.0
            self.hpu = HydraulicPowerUnit(
                identity="base.hpu", motor_kw=round(max(7.5, kw), 1),
                pump_displacement_cc_rev=round(max(16.0, cc), 1),
                pump_rpm=1450.0,
                relief_pressure_pa=35.0e6, reservoir_l=120.0, oil_l=105.0,
                accumulator_l=4.0, accumulator_precharge_pa=12.0e6)
        for name in self.leg_names:
            self.legs[name] = OutriggerLeg(
                identity=f"outrigger.{name}",
                cylinder=LinearActuator(
                    identity=f"outrigger.jack.{name}",
                    bore_m=self.bore_m, rod_m=self.rod_m,
                    stroke_m=self.stroke_m,
                    stages=max(1, int(self.stages)),
                    # DECLARE IT TELESCOPIC. `force_at` steps the force
                    # DOWN as each narrower stage takes over, which is
                    # the real and frequently surprising behaviour of a
                    # telescopic ram -- and it never ran, because the
                    # kind was left at its default.
                    kind="telescopic" if self.stages > 1 else "double-acting",
                    mounting="flange-rigid",
                    rated_pressure_pa=35.0e6),
                share_kg=self.machine_mass_kg / max(len(self.leg_names), 1),
                pad_area_m2=self.pad_area_m2,
                ground_bearing_pa=self.ground_bearing_pa)

    # ------------------------------------------------------------------
    def step(self, dt: float, command: float = 1.0) -> dict:
        """One tick of raising (or lowering) the machine.

        The four legs are asked for at once and SHARE ONE PUMP, so the
        flow each gets is its quarter -- which is exactly why a machine
        this size takes as long as it does to stand up."""
        self.elapsed_s += dt
        # what the pump is asked for, and against what pressure
        demand = 0.0
        need_p = ATM_PA
        for leg in self.legs.values():
            a = leg.cylinder
            q = abs(command) * a.area_extend_m2 * 0.06 * 60_000.0
            demand += q
            # BREAKAWAY, NOT RUNNING FRICTION. A cylinder at rest has to
            # be broken loose before it will move, and that takes
            # strictly more pressure than keeping it moving does. The
            # old demand used running friction only, so the pump was
            # told to make a pressure that could not start the rod --
            # which read as "the pack cannot lift this machine" when the
            # truth was that the pressure command was too low.
            moving = abs(getattr(a, "velocity_m_s", 0.0)) > 1e-4
            r = self.losses.pump_pressure_for(
                load_n=leg.load_n,
                piston_area_m2=a.area_extend_m2,
                annulus_area_m2=a.area_retract_m2,
                flow_l_min=q,
                oil_temp_k=self.hpu.temp_k,
                seal_friction_frac=getattr(a, "seal_friction_frac", 0.05),
                breakaway=1.0 if moving else BREAKAWAY_MULTIPLIER,
                holding=True)
            leg.last_pressure = r
            need_p = max(need_p, r["at_pump_pa"])
        out = self.hpu.step(dt, demand, need_p)
        self.electrical_w = self.losses.electrical_w(
            pump_pressure_pa=out["pressure_pa"],
            flow_l_min=max(out["delivered_l_min"], demand * 0.0))
        share = out["delivered_l_min"] / max(len(self.legs), 1)
        state = {}
        for name, leg in self.legs.items():
            r = leg.cylinder.step(dt, out["pressure_pa"], share, command,
                                  leg.load_n)
            leg.on_ground = leg.cylinder.position_m > 1e-4
            state[name] = {
                "extension_m": leg.cylinder.position_m,
                "force_n": r.get("force_n", 0.0),
                "pressure_pa": out["pressure_pa"],
                "pad_pressure_pa": leg.pad_pressure_pa,
                "sinks": leg.sinks,
            }
        self.electrical_wh += self.electrical_w * dt / 3600.0
        return {"legs": state, "pump": out,
                "electrical_w": self.electrical_w,
                "electrical_wh": self.electrical_wh,
                "required_pressure_pa": need_p,
                "demand_l_min": demand,
                "elapsed_s": self.elapsed_s,
                "extension_m": min(l.cylinder.position_m
                                   for l in self.legs.values())}

    # ------------------------------------------------------------------
    def raise_fully(self, *, dt: float = 0.05, limit_s: float = 600.0) -> dict:
        """Run the pump until the legs stop moving, and report what it
        took -- which is the honest form of "deployed"."""
        last = -1.0
        log = None
        while self.elapsed_s < limit_s:
            log = self.step(dt, 1.0)
            now = log["extension_m"]
            if now >= self.stroke_m - 1e-6:
                break
            if abs(now - last) < 1e-9 and self.elapsed_s > 1.0:
                break                      # stalled: it cannot lift this
            last = now
        return {**log, "stalled": log["extension_m"] < self.stroke_m - 1e-4}

    def describe(self, log: dict) -> list[str]:
        legs = log["legs"]
        a = next(iter(self.legs.values())).cylinder
        out = [
            # COUNT THE LEGS, do not write "four". The set grew to eight
            # and this line went on saying four, and the line below it
            # went on dividing the machine's mass by four -- so a
            # correctly-loaded eight-leg machine reported twice the
            # load per leg that its pads were actually carrying.
            f"{len(self.legs)} x {a.bore_m * 1000:.0f}/{a.rod_m * 1000:.0f} x "
            f"{a.stroke_m * 1000:.0f} "
            f"{'%d-stage telescopic ' % self.stages if self.stages > 1 else ''}"
            f"mm jacks on one {self.hpu.motor_kw:.1f} kW pack",
            f"  machine {self.machine_mass_kg:.0f} kg -> "
            f"{self.machine_mass_kg / max(len(self.legs), 1) * 9.81 / 1000:.1f} "
            f"kN per leg",
            f"  pressure needed {log['required_pressure_pa'] / 1e5:.0f} bar, "
            f"pump made {log['pump']['pressure_pa'] / 1e5:.0f} bar",
            f"  flow asked {log['demand_l_min']:.1f} l/min, "
            f"delivered {log['pump']['delivered_l_min']:.1f} l/min",
            f"  raised {log['extension_m'] * 1000:.0f} mm in "
            f"{log['elapsed_s']:.1f} s"
            + ("  -- STALLED, the pack cannot lift this" if log.get("stalled")
               else ""),
        ]
        for name, st in legs.items():
            out.append(
                f"    {name}: {st['extension_m'] * 1000:6.0f} mm  "
                f"pad {st['pad_pressure_pa'] / 1000:.0f} kPa"
                + ("  SINKING" if st["sinks"] else ""))
        return out
