"""Firing to an EFFECT, not to a cartridge.

WHAT AN OPERATOR ACTUALLY WANTS TO SAY. Not "load charge four" -- that
is a number about the gun. What they want to say is about the target:

    put the reticle on it, "hit that with 400 kJ, 180 gram darts,
    hold that setting"

and everything between those words and the primer is arithmetic. This
module is that arithmetic, and it runs backwards through every stage
the gun already models:

    intended energy AT THE TARGET
      -> impact velocity           (energy and dart mass)
      -> muzzle velocity           (the drag the round will actually
                                    suffer over that range -- the
                                    scope's own solution, not a guess)
      -> charge                    (how many pucks, pressed to what
                                    web, from the interior ballistics
                                    kernel)
      -> and out of that same charge: the recoil the mount must take,
         and the gas the port will have to harvest

THE SECOND FEED TRACK IS THE WHOLE IDEA. One track carries factory
full-charge rounds and feeds at the cyclic rate, because sometimes the
answer is simply "as much as possible, now". The other assembles a
round per task: pucks stacked, case sealed, primed and rammed. Assembly
takes milliseconds, so the limit is not whether it can be done between
shots -- it is whether the assembler can stay AHEAD of the gun, which
is a buffer question and is answered here rather than assumed.

WHY SHOOTING TO AN ENERGY IS WORTH THE MACHINERY. A full charge at a
close soft target is waste in five directions at once: barrel life,
propellant, recoil into the mount, noise and flash that give the
position away, and overpenetration that does nothing. Charge is the
cheapest variable on the gun and the only one that can be changed
between one shot and the next.

AND THE PLANT CAN ORDER ROUNDS TOO. The hit-or-miss expander runs on
gas the gun throws away; under sustained fire it is the outpost's
largest power source. So the generator is allowed to place a request --
"I need this much harvestable gas" -- and the loader can satisfy it by
choosing a charge that makes it, as long as doing so does not break the
firing task it already has. Firing and generating stop being separate
activities, which they never really were.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from reloading import (CASINGS, PRIMERS, PUCKS, SHIMS, KITS, Load,
                       check_load, fire as fire_load)


# =====================================================================
#  RUNNING THE PROBLEM BACKWARDS
# =====================================================================
def impact_speed_for(energy_j: float, dart_mass_kg: float) -> float:
    """The speed a dart of this mass must still have to land that hard."""
    return math.sqrt(2.0 * max(energy_j, 0.0) / max(dart_mass_kg, 1e-9))


def muzzle_for_impact(energy_j: float, dart_mass_kg: float, range_m: float,
                      drag_k_per_m: float) -> dict:
    """What it has to leave at, to arrive like that.

    Uses the same exponential drag the calibre table and the scope use
    -- v(r) = v0 exp(-k r) -- so the number handed to the charge solver
    is the number the scope will later agree with. Two different drag
    models either side of this would be a gun that cannot hit what its
    own sight says it can.
    """
    v_impact = impact_speed_for(energy_j, dart_mass_kg)
    v_muzzle = v_impact * math.exp(drag_k_per_m * max(range_m, 0.0))
    return {
        "impact_speed_m_s": v_impact,
        "muzzle_speed_m_s": v_muzzle,
        "muzzle_energy_j": 0.5 * dart_mass_kg * v_muzzle ** 2,
        "energy_lost_to_drag_j": 0.5 * dart_mass_kg * (v_muzzle ** 2
                                                       - v_impact ** 2),
        "retained_fraction": (v_impact / v_muzzle) ** 2 if v_muzzle else 0.0,
    }


@dataclass
class TaskedRound:
    """One round, assembled for one job."""
    kit: str
    load: Load
    want_energy_j: float
    at_range_m: float
    predicted_muzzle_m_s: float
    predicted_impact_j: float
    peak_breech_pa: float
    recoil_impulse_n_s: float
    harvestable_gas_kg: float
    pucks: int
    web_m: float

    def describe(self) -> list[str]:
        return [
            f"{self.kit}: {self.want_energy_j / 1000:.0f} kJ at "
            f"{self.at_range_m:.0f} m",
            f"  {self.pucks} pucks, web {self.web_m * 1000:.2f} mm "
            f"= {self.load.charge_kg * 1000:.0f} g",
            f"  muzzle {self.predicted_muzzle_m_s:.0f} m/s, "
            f"arrives at {self.predicted_impact_j / 1000:.0f} kJ",
            f"  peak {self.peak_breech_pa / 1e6:.0f} MPa, recoil "
            f"{self.recoil_impulse_n_s:.0f} N.s",
        ]


def solve_charge(kit: str, dart_mass_kg: float, want_energy_j: float,
                 range_m: float, *, drag_k_per_m: float = 0.0004,
                 barrel_m: float = None, peak_limit_pa: float = 4.2e8,
                 webs=None, max_pucks: int = 120,
                 port_fraction: float = 0.0) -> TaskedRound:
    """Assemble the round that lands that hard at that range.

    Searches what a reloading shop can actually vary -- PUCK COUNT and
    WEB -- and rejects anything the chamber cannot burn or the breech
    cannot survive. It does not adjust a propellant property to reach
    the answer, because a propellant property is not a thing the shop
    can change between two shots.
    """
    k = KITS[kit]
    barrel = barrel_m if barrel_m is not None else k.inner_barrel_length_m
    want = muzzle_for_impact(want_energy_j, dart_mass_kg, range_m,
                             drag_k_per_m)
    target_v = want["muzzle_speed_m_s"]
    if webs is None:
        webs = [0.05e-3 * i for i in range(3, 25)]
    best = None
    for n in range(1, max_pucks + 1):
        gate = check_load(k.casing, k.puck, n, k.shim or None)
        if not gate["ok"]:
            break
        for web in webs:
            load = Load(k.casing, k.primer, k.puck, n, dart_mass_kg, web,
                        label=f"{kit} tasked")
            r = fire_load(load, barrel)
            if r["peak_breech_pa"] > peak_limit_pa:
                continue
            err = abs(r["muzzle_m_s"] - target_v)
            if best is None or err < best[0]:
                best = (err, load, r, n, web)
    if best is None:
        raise ValueError(
            f"no assembleable charge fires a {dart_mass_kg * 1000:.0f} g "
            f"dart hard enough to land {want_energy_j / 1000:.0f} kJ at "
            f"{range_m:.0f} m without exceeding "
            f"{peak_limit_pa / 1e6:.0f} MPa")
    _, load, r, n, web = best
    v0 = r["muzzle_m_s"]
    v_imp = v0 * math.exp(-drag_k_per_m * range_m)
    return TaskedRound(
        kit=kit, load=load, want_energy_j=want_energy_j, at_range_m=range_m,
        predicted_muzzle_m_s=v0,
        predicted_impact_j=0.5 * dart_mass_kg * v_imp ** 2,
        peak_breech_pa=r["peak_breech_pa"],
        recoil_impulse_n_s=dart_mass_kg * v0 + load.charge_kg * v0 * 1.5,
        harvestable_gas_kg=load.charge_kg * port_fraction,
        pucks=n, web_m=web)


# =====================================================================
#  THE TWO FEED TRACKS
# =====================================================================
@dataclass
class DualFeed:
    """Track A: factory rounds at the cyclic rate. Track B: assembled.

    The buffer is the point. A round takes milliseconds to build, which
    sounds instant until you are firing five a second -- then the
    question is not "can it build one in time" but "can it keep
    building them faster than the gun empties the buffer". That is what
    `sustainable_rate_hz` answers, and it is why the standing order is
    held as a SETTING: once the task is fixed the assembler can run
    ahead and stack a magazine of identical tasked rounds, which is
    what makes a full-auto burst at a custom charge possible at all.
    """
    identity: str = "turret.dual_feed"
    #: track A
    ready_rounds: int = 40
    cyclic_hz: float = 5.5
    #: track B
    assemble_s: float = 0.045           # stack, seal, prime, ram
    buffer_depth: int = 12
    buffered: int = 0
    standing_order: TaskedRound = None
    #: live
    assembling_s: float = 0.0
    fired_a: int = 0
    fired_b: int = 0
    starved: int = 0

    @property
    def sustainable_rate_hz(self) -> float:
        """How fast track B can feed indefinitely, buffer or no buffer."""
        return 1.0 / max(self.assemble_s, 1e-9)

    def burst_rounds(self, rate_hz: float) -> float:
        """How long a burst at this rate before the buffer is empty.

        Above the sustainable rate the buffer drains at the difference,
        so the burst length is the buffer divided by that difference --
        and below it, the buffer never empties at all."""
        deficit = rate_hz - self.sustainable_rate_hz
        if deficit <= 0:
            return float("inf")
        return self.buffered / deficit * rate_hz

    def set_task(self, round_: TaskedRound) -> None:
        """'Hold that setting.' The buffer of the previous task is no
        longer what was ordered, so it is returned to stock rather than
        fired at the wrong thing."""
        self.standing_order = round_
        self.buffered = 0
        self.assembling_s = 0.0

    def step(self, dt: float, want_shots: int = 0,
             track: str = "b") -> dict:
        """One tick: build what we can, fire what was asked for."""
        made = 0
        if self.standing_order is not None and self.buffered < self.buffer_depth:
            self.assembling_s += dt
            while (self.assembling_s >= self.assemble_s
                   and self.buffered < self.buffer_depth):
                self.assembling_s -= self.assemble_s
                self.buffered += 1
                made += 1
        fired = 0
        for _ in range(max(0, int(want_shots))):
            if track == "a":
                if self.ready_rounds > 0:
                    self.ready_rounds -= 1
                    self.fired_a += 1
                    fired += 1
                else:
                    self.starved += 1
            else:
                if self.buffered > 0:
                    self.buffered -= 1
                    self.fired_b += 1
                    fired += 1
                else:
                    self.starved += 1
        return {"made": made, "fired": fired, "buffered": self.buffered,
                "ready_rounds": self.ready_rounds, "starved": self.starved}

    def describe(self) -> list[str]:
        out = [
            f"{self.identity}:",
            f"  track A  {self.ready_rounds} factory rounds, "
            f"{self.cyclic_hz:.1f} /s cyclic",
            f"  track B  assembles in {self.assemble_s * 1000:.0f} ms "
            f"= {self.sustainable_rate_hz:.1f} /s sustained, "
            f"buffer {self.buffer_depth}",
        ]
        if self.standing_order is not None:
            out.append(f"  standing order: "
                       f"{self.standing_order.want_energy_j / 1000:.0f} kJ at "
                       f"{self.standing_order.at_range_m:.0f} m")
        return out


def charge_for_harvest(kit: str, dart_mass_kg: float, want_gas_kg: float, *,
                       port_fraction: float = 0.02, barrel_m: float = None,
                       peak_limit_pa: float = 4.2e8) -> TaskedRound:
    """A round chosen for what the PORT will get out of it.

    The generator asking the gun for fuel. It is only ever a secondary
    consideration -- a round is fired at something, not at the
    flywheel -- but when the firing task leaves a choice of charges
    that all satisfy it, this is how the plant expresses a preference.
    """
    k = KITS[kit]
    barrel = barrel_m if barrel_m is not None else k.inner_barrel_length_m
    need_charge = want_gas_kg / max(port_fraction, 1e-9)
    pucks = max(1, int(round(need_charge / PUCKS[k.puck].mass_kg)))
    gate = check_load(k.casing, k.puck, pucks, k.shim or None)
    if not gate["ok"]:
        raise ValueError(
            f"{want_gas_kg * 1000:.0f} g of harvestable gas needs "
            f"{need_charge * 1000:.0f} g of charge, which will not burn in "
            f"a {gate['chamber_l'] * 1000:.0f} cc chamber")
    load = Load(k.casing, k.primer, k.puck, pucks, dart_mass_kg, 0.00027,
                label=f"{kit} harvest")
    r = fire_load(load, barrel)
    return TaskedRound(
        kit=kit, load=load, want_energy_j=0.0, at_range_m=0.0,
        predicted_muzzle_m_s=r["muzzle_m_s"],
        predicted_impact_j=0.5 * dart_mass_kg * r["muzzle_m_s"] ** 2,
        peak_breech_pa=r["peak_breech_pa"],
        recoil_impulse_n_s=dart_mass_kg * r["muzzle_m_s"],
        harvestable_gas_kg=load.charge_kg * port_fraction,
        pucks=pucks, web_m=0.00027)


# =====================================================================
#  SIZING THE CHARGE WITHOUT SEARCHING FOR IT
# =====================================================================
def charge_for_muzzle(target_m_s: float, *, shot_kg: float = 0.130,
                      bore_mm: float = 20.0, barrel_m: float = 8.40,
                      chamber_l: float = 0.048, web_m: float = 2.7e-4,
                      impetus: float = 1.1e6, gamma: float = 1.25,
                      refinements: int = 2, fire_fn=None) -> dict:
    """What charge makes that muzzle velocity -- solved, not searched.

    WHY NOT A SEARCH. `size_charge` above walks every puck count
    against every web and runs the ballistics kernel at each point:
    hundreds of full integrations, about nine seconds. That is fine for
    designing a cartridge at leisure and useless for answering "hit it
    with thirty kilojoules" while the reticle is on something.

    AND IT DOES NOT NEED ONE. The energy equation is already an
    expression for what a charge is worth:

        available work = f w / (gamma - 1)
        shot energy    = eta * available

    so an opening guess falls straight out of the target energy:

        w = (gamma - 1) E_target / (f eta)

    eta -- the piezometric efficiency, the share of the available work
    that ends up in the shot rather than in heat, gas motion and
    recoil -- is not guessed either: it is READ from one run of the
    kernel, because the kernel already reports every one of those
    terms. One integration gives the efficiency, and the efficiency
    turns the target into a charge.

    Then at most a couple of Newton corrections on the real kernel,
    because eta drifts slowly with charge. Three integrations total
    against several hundred, and it converges on the answer rather
    than the nearest point of a grid.
    """
    import math
    if fire_fn is None:
        import interior_ballistics as ib
        step, _ = ib.native_step()
        area = math.pi * (bore_mm / 2000.0) ** 2

        def fire_fn(charge_kg):
            kw = dict(dt_s=1e-6, bore_area_m2=area,
                      chamber_volume_m3=chamber_l / 1000.0,
                      shot_mass_kg=shot_kg, charge_mass_kg=charge_kg,
                      impetus_j_per_kg=impetus, covolume_m3_per_kg=1.0e-3,
                      propellant_density_kg_m3=1620.0, gamma=gamma,
                      burn_rate_coeff=6.4e-10, grain_chi=1.0, grain_lambda=0.0,
                      grain_web_m=web_m, shot_start_pressure_pa=3.5e7,
                      barrel_length_m=barrel_m, twist_calibres_per_turn=30.0,
                      band_interference_m=50e-6, bore_diameter_m=bore_mm / 1000.0,
                      primer_gas_kg=3.5e-4, recoiling_mass_kg=85.0,
                      gas_port_travel_m=1e9, gas_port_fraction=0.0)
            st = {"travel_m": 0.0, "velocity_m_s": 0.0, "burnt_fraction": 0.0,
                  "recoil_velocity_m_s": 0.0, "harvested_gas_kg": 0.0,
                  "wall_heat_j": 0.0}
            peak = 0.0
            for _ in range(600_000):
                r = step(**kw, **st)
                peak = max(peak, r["breech_pressure_pa"])
                st = {"travel_m": r["travel_next_m"],
                      "velocity_m_s": r["velocity_next_m_s"],
                      "burnt_fraction": r["burnt_fraction_next"],
                      "recoil_velocity_m_s": r["recoil_velocity_next_m_s"],
                      "harvested_gas_kg": r["harvested_gas_next_kg"],
                      "wall_heat_j": r["wall_heat_next_j"]}
                if r["at_muzzle"] > 0.5:
                    break
            return st["velocity_m_s"], peak

    want_energy = 0.5 * shot_kg * target_m_s ** 2
    # one probe, to learn this gun's efficiency
    probe_charge = 0.040
    v0, p0 = fire_fn(probe_charge)
    eta = (0.5 * shot_kg * v0 ** 2) / (probe_charge * impetus / (gamma - 1))
    charge = (gamma - 1) * want_energy / (impetus * max(eta, 1e-6))
    runs = 1
    v, p = v0, p0
    for _ in range(refinements):
        v, p = fire_fn(charge)
        runs += 1
        if abs(v - target_m_s) < 2.0:
            break
        # velocity goes roughly as the square root of charge, so the
        # correction is a ratio of squares -- one step gets very close
        charge *= (target_m_s / max(v, 1.0)) ** 2
        charge = max(0.002, charge)
    return {"charge_kg": charge, "muzzle_m_s": v, "peak_breech_pa": p,
            "piezometric_efficiency": eta, "kernel_runs": runs,
            "target_m_s": target_m_s, "error_m_s": v - target_m_s}
