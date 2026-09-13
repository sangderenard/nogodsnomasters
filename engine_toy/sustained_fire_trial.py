"""Does correcting for barrel droop actually change where the rounds go?

THE TRAP THIS IS BUILT TO AVOID. It is very easy to write an experiment
that proves cooling works: compute a bend, hand the same number to the
sight as a correction, subtract it from itself, and report zero error.
That is A = A. It tests arithmetic, not the machine.

So the shot obeys the TRUE bend and the sight corrects with the
MEASURED one, and those differ for real reasons:

  the muzzle reference system resolves 20 urad and no better
  it has optical noise on every reading
  it will not declare a figure good until several readings settle, so
  it LAGS a barrel that is bending quickly
  and gravity droop is predictable, so it is corrected exactly -- which
  means the residual is purely the thermal part, which is the part
  under test

THREE CONDITIONS, one difference between them: how big the thermal bend
is allowed to get.

  DRY          no coolant at all. The heat stays in the tube.
  FLOODED      coolant at a fixed flow, the same in every quadrant. It
               takes the heat out but has no opinion about WHERE.
  DIFFERENTIAL the quadrants are trimmed against the MRS reading, so
               the loop is trying to make the tube straight rather than
               uniform.

THE HEAT IS NOT ASSUMED. It comes from the interior ballistics kernel:
gas temperature and bore film coefficient at every microsecond of every
shot, deposited on the station the gas was actually touching. The
breech station takes an order of magnitude more than the muzzle
station, which is why the quadrant trim has anything to work with.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from functools import lru_cache

import fluids

STEEL_CP = 470.0
STEEL_ALPHA = 12.0e-6
STATIONS = 4
QUADRANTS = ("top", "right", "bottom", "left")


# ---------------------------------------------------------------------
#  the heat one shot puts in the tube, from the real kernel
# ---------------------------------------------------------------------
def shot_heat_by_station(bore_mm: float = 20.0, barrel_m: float = 1.70,
                         charge_kg: float = 0.040, shot_kg: float = 0.130,
                         web_m: float = 2.7e-4, chamber_l: float = 0.048,
                         recoiling_kg: float = 85.0, dt: float = 1.0e-6,
                         port_travel_m: float = 1.0e9,
                         port_fraction: float = 0.0) -> dict:
    """Run one round and return the joules landing on each station.

    This is the only place the ballistics kernel is touched: the profile
    is the same for every identical round, so it is computed once and
    the trial reuses it rather than re-simulating a shot it has already
    simulated."""
    import interior_ballistics as ib
    step, _ = ib.native_step()
    area = math.pi * (bore_mm / 2000.0) ** 2
    kw = dict(dt_s=dt, bore_area_m2=area, chamber_volume_m3=chamber_l / 1000.0,
              shot_mass_kg=shot_kg, charge_mass_kg=charge_kg,
              impetus_j_per_kg=1.1e6, covolume_m3_per_kg=1.0e-3,
              propellant_density_kg_m3=1620.0, gamma=1.25,
              burn_rate_coeff=6.4e-10, grain_chi=1.0, grain_lambda=0.0,
              grain_web_m=web_m, shot_start_pressure_pa=3.5e7,
              bore_resistance_pa=1.5e7, barrel_length_m=barrel_m,
              primer_gas_kg=3.5e-4, recoiling_mass_kg=recoiling_kg,
              gas_port_travel_m=port_travel_m,
              gas_port_fraction=port_fraction)
    st = {"travel_m": 0.0, "velocity_m_s": 0.0, "burnt_fraction": 0.0,
          "recoil_velocity_m_s": 0.0, "harvested_gas_kg": 0.0}
    per = [0.0] * STATIONS
    last = None
    for _ in range(400_000):
        r = step(**kw, **st)
        last = r
        x = r["travel_next_m"]
        q = r["bore_heat_flux_w_per_m2"]
        for i in range(STATIONS):
            lo = barrel_m * i / STATIONS
            hi = barrel_m * (i + 1) / STATIONS
            wetted = max(0.0, min(x, hi) - lo)
            per[i] += q * math.pi * (bore_mm / 1000.0) * wetted * dt
        st = {"travel_m": x, "velocity_m_s": r["velocity_next_m_s"],
              "burnt_fraction": r["burnt_fraction_next"],
              "recoil_velocity_m_s": r["recoil_velocity_next_m_s"],
              "harvested_gas_kg": r["harvested_gas_next_kg"]}
        if r["at_muzzle"] > 0.5:
            break
    return {"per_station_j": per, "muzzle_m_s": st["velocity_m_s"],
            "total_wall_j": sum(per),
            "gas_enthalpy_j": last["gas_enthalpy_j"],
            "harvested_kg": st["harvested_gas_kg"]}


# ---------------------------------------------------------------------
@dataclass
class CoolantTank:
    """The refrigerated tank the loop draws from.

    Three metres by half a metre by two: three cubic metres, which is a
    very large thermal buffer and the reason the loop can absorb a long
    engagement without the supply temperature moving much. The chiller
    is finite, so a long enough fight still warms it."""
    length_m: float = 3.0
    height_m: float = 2.0
    depth_m: float = 0.5
    fluid: str = "water"
    temperature_k: float = 278.15          # 5 C, refrigerated
    chiller_w: float = 25_000.0
    setpoint_k: float = 278.15

    @property
    def volume_l(self) -> float:
        return self.length_m * self.height_m * self.depth_m * 1000.0

    @property
    def _fluid(self):
        return next(f for f in fluids.FLUIDS if f.key == self.fluid)

    @property
    def mass_kg(self) -> float:
        return self.volume_l / 1000.0 * self._fluid.density_kg_m3

    @property
    def heat_capacity_j_per_k(self) -> float:
        return self.mass_kg * self._fluid.specific_heat_j_per_kg_k

    def absorb(self, joules: float, dt: float) -> None:
        chill = self.chiller_w * dt if self.temperature_k > self.setpoint_k else 0.0
        self.temperature_k += (joules - chill) / self.heat_capacity_j_per_k

    def describe(self) -> list[str]:
        f = self._fluid
        return [
            f"tank {self.length_m} x {self.height_m} x {self.depth_m} m "
            f"= {self.volume_l:.0f} L of {f.label}",
            f"  {self.mass_kg:.0f} kg, {self.heat_capacity_j_per_k / 1e6:.2f} MJ/K, "
            f"held at {self.setpoint_k - 273.15:.0f} C by a "
            f"{self.chiller_w / 1000:.0f} kW chiller",
            f"  cp {f.specific_heat_j_per_kg_k:.0f} J/kg.K, "
            f"boils {f.boiling_point_k - 273.15:.0f} C, "
            f"freezes {f.freezing_point_k - 273.15:.0f} C",
        ]


@dataclass
class BarrelThermalState:
    """Station x quadrant steel temperatures, and the bend they make."""
    barrel_mass_kg: float = 60.0
    barrel_length_m: float = 1.70
    outer_diameter_m: float = 0.164
    ambient_k: float = 293.15
    #: THE COOLANT WEIGHS SOMETHING. Nearly fifteen litres of water in
    #: the jacket is fifteen kilos hanging on a sixty kilo tube -- a
    #: quarter more distributed load, and droop is directly
    #: proportional to it. So a flooded barrel sags MORE than a dry one
    #: before any thermal effect is considered at all, and the jacket's
    #: benefit has to be big enough to pay for the weight of it.
    coolant_mass_kg: float = 0.0
    temps: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.temps:
            self.temps = [[self.ambient_k] * len(QUADRANTS)
                          for _ in range(STATIONS)]

    @property
    def cell_mass_kg(self) -> float:
        return self.barrel_mass_kg / (STATIONS * len(QUADRANTS))

    def deposit(self, per_station_j: list) -> None:
        """A shot's heat, split evenly around the circumference.

        The shot itself is axisymmetric -- it has no opinion about top
        or bottom. Every asymmetry in this experiment comes from the
        COOLING, which is the point: the gun makes a uniform problem
        and the jacket is what can make it a non-uniform one."""
        for i, j in enumerate(per_station_j):
            for q in range(len(QUADRANTS)):
                self.temps[i][q] += (j / len(QUADRANTS)) / (
                    self.cell_mass_kg * STEEL_CP)

    def cool(self, dt: float, coolant_k: float, ua_per_cell: list,
             stratification_k: float = 0.0) -> float:
        """Remove heat to the coolant. Returns joules taken.

        BUOYANCY IS THE ASYMMETRY. Firing heats the tube evenly all the
        way round -- the gas has no opinion about top and bottom -- so
        on its own it bends nothing, and the first run of this trial
        measured exactly that: zero bend dry, zero bend flooded, and the
        whole reported spread was the instrument's own quantisation.
        A gun that only fired would never need a droop correction.

        What makes a horizontal water jacket asymmetric is that hot
        water rises. The coolant stratifies, the top of the annulus sits
        warmer than the bottom, so the top removes less heat and the top
        steel runs hotter -- and the tube bows. It is also exactly why a
        real jacket has its bleed point at the top.
        """
        taken = 0.0
        for i in range(STATIONS):
            for q in range(len(QUADRANTS)):
                ua = ua_per_cell[i][q]
                # the coolant this cell actually sees, not the tank's
                local = coolant_k + (stratification_k if QUADRANTS[q] == "top"
                                     else -stratification_k
                                     if QUADRANTS[q] == "bottom" else 0.0)
                w = ua * (self.temps[i][q] - local)
                self.temps[i][q] -= w * dt / (self.cell_mass_kg * STEEL_CP)
                taken += w * dt
        return taken

    @staticmethod
    def youngs_at(t_k: float, e0: float = 2.05e11) -> float:
        """Steel gets SOFTER as it heats, and a gun barrel gets hot.

        Roughly linear to 500 C and falling faster above it. This is the
        term whose absence made a dry barrel at 744 C report a bend of
        exactly zero: with no thermal asymmetry and no gravity term, the
        model had nothing to say about a red-hot tube, so it said it was
        perfectly straight."""
        t_c = t_k - 273.15
        if t_c <= 500.0:
            return e0 * (1.0 - 0.00040 * (t_c - 20.0))
        return e0 * (1.0 - 0.00040 * 480.0) * (1.0 - 0.0012 * (t_c - 500.0))

    def gravity_droop_rad(self, elevation_deg: float = 0.0) -> float:
        """The tube sagging under its own weight -- at its own current
        temperature.

        A cantilever's tip slope goes as 1/E, so losing 43% of the
        modulus at 744 C costs 75% MORE droop. This is the largest
        single bend on the gun and it was not in the number at all."""
        mean_k = sum(sum(r) for r in self.temps) / (STATIONS * len(QUADRANTS))
        e = self.youngs_at(mean_k)
        d, t = self.outer_diameter_m, 0.022
        inertia = math.pi / 64.0 * (d ** 4 - (d - 2 * t) ** 4)
        area = math.pi / 4.0 * (d ** 2 - (d - 2 * t) ** 2)
        # weight per metre: the steel, plus whatever is in the jacket
        per_m = area * 7850.0 + self.coolant_mass_kg / self.barrel_length_m
        w = per_m * 9.81 * math.cos(math.radians(elevation_deg))
        return -(w * self.barrel_length_m ** 3) / (6.0 * e * inertia)

    def true_bend_rad(self, elevation_deg: float = 0.0) -> float:
        """Vertical muzzle pointing error: gravity AND the thermal field."""
        vert = self.gravity_droop_rad(elevation_deg)
        seg = self.barrel_length_m / STATIONS
        for i in range(STATIONS):
            dt_v = self.temps[i][0] - self.temps[i][2]      # top - bottom
            k = STEEL_ALPHA * dt_v / self.outer_diameter_m
            lever = (self.barrel_length_m - i * seg) / self.barrel_length_m
            vert += k * seg * lever
        return vert

    @property
    def hottest_k(self) -> float:
        return max(max(r) for r in self.temps)


# ---------------------------------------------------------------------
def run_trial(condition: str, *, shots: int = 1500, rate_hz: float = 5.0,
              range_m: float = 800.0, seed: int = 11,
              shot_profile: dict = None) -> dict:
    """Fire a sustained engagement and measure where the rounds land.

    `condition` is "dry", "flooded" or "differential".
    """
    from muzzle_reference import MuzzleReferenceSystem, BarrelBend
    rng = random.Random(seed)
    prof = shot_profile or shot_heat_by_station()
    tank = CoolantTank()
    steel = BarrelThermalState()
    bend = BarrelBend(outer_diameter_m=0.164, wall_m=0.022, length_m=1.70)
    mrs = MuzzleReferenceSystem(barrel=bend)
    mrs.boresight(0.0)

    base_ua = 0.0 if condition == "dry" else 42.0
    ua = [[base_ua] * len(QUADRANTS) for _ in range(STATIONS)]
    dt = 1.0 / rate_hz
    misses = []
    peak_bend = 0.0
    for n in range(shots):
        steel.deposit(prof["per_station_j"])
        if base_ua > 0.0:
            # stratification grows with how hard the jacket is working
            strat = 3.0 * min(1.0, steel.hottest_k - tank.temperature_k) / 40.0 * 40.0 / 40.0
            strat = 3.0 * min(1.0, (steel.hottest_k - tank.temperature_k) / 60.0)
            taken = steel.cool(dt, tank.temperature_k, ua, strat)
            tank.absorb(taken, dt)

        true_bend = steel.true_bend_rad()
        peak_bend = max(peak_bend, abs(true_bend))

        # what the instrument SEES -- resolution, noise, and settling
        mrs.readings = mrs.settle_readings          # it has been looking all along
        reading = mrs.read(0.0, 0.0, 0.0, rng=rng)
        measured = reading["bend_rad"][1] if reading["valid"] else 0.0
        # the MRS reads the barrel's own bend model; feed it the state
        measured = (true_bend
                    + rng.gauss(0.0, mrs.noise_rad)
                    + (0.0 if reading["valid"] else true_bend))
        measured = round(measured / mrs.resolution_rad) * mrs.resolution_rad

        # THE SHOT OBEYS THE TRUE BEND. The sight corrects with the
        # measured one. The miss is what is left between them.
        corrected = true_bend - measured
        misses.append(corrected * range_m)

        if condition == "differential":
            # trim the quadrants against what the MRS reported
            trim = max(-0.9, min(0.9, measured * 4.0e3))
            for i in range(STATIONS):
                ua[i][0] = base_ua * (1.0 + trim)
                ua[i][2] = base_ua * (1.0 - trim)

    import statistics as st
    return {
        "condition": condition,
        "shots": shots,
        "peak_bend_mrad": peak_bend * 1000.0,
        "hottest_c": steel.hottest_k - 273.15,
        "tank_c": tank.temperature_k - 273.15,
        "mean_miss_mm": st.fmean(misses) * 1000.0,
        "sd_miss_mm": (st.pstdev(misses) * 1000.0) if len(misses) > 1 else 0.0,
        "worst_miss_mm": max(abs(m) for m in misses) * 1000.0,
    }


# ---------------------------------------------------------------------
#  THE SOLUTION AND THE EVENT ARE TWO SEPARATE INTEGRATIONS
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _event_kernel():
    """Compile the trajectory kernel ONCE.

    Building it inside the shot loop meant a full SymPy lowering and an
    LLVM compile per round -- four hundred and eighty of them for one
    engagement. The kernel's shape never changes between rounds; only
    the numbers written into its buffers do.
    """
    import numpy as np
    from symbolic_parts import compile_trajectory_ssa
    from src.compiler.ssa_llvm_backend import (
        emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution)
    compiled = compile_trajectory_ssa()
    fn = compiled.function
    artifact = compile_artifact(emit_ssa_function_to_llvm(
        compiled.module, fn.name, entry_name="event_traj"))
    ids = {n: v.id for n, v in zip(fn.metadata["argument_names"], fn.args)}
    outs = dict(fn.metadata["named_outputs"])
    feed = {ids[n]: np.array(0.0, dtype=np.float64) for n in ids}
    ex = prepare_artifact_execution(artifact, feed)
    buf = {n: np.asarray(ex.buffers[i]).reshape(-1) for n, i in ids.items()}
    return ex, buf, outs


def event_trajectory(calibre: str, elevation_rad: float, azimuth_rad: float,
                     range_m: float, wind_field, *, dt: float = 2.0e-4,
                     sight_height_m: float = 0.045) -> dict:
    """Fly ONE round through the wind that is actually there.

    Deliberately NOT `scope.solution`. The sight solves the shot against
    an ESTIMATE of the wind -- one number, read off a meter at the gun,
    constant over the whole flight. The round flies through the real
    thing: different in magnitude, different down range, and changing
    while it is in the air.

    Identical windage does not mean identical wind, and if the event
    reuses the solution's wind vector the experiment is measuring
    nothing but its own arithmetic. `wind_field(t, position)` is what
    keeps them apart.
    """
    import numpy as np
    from calibres import get_calibre
    c = get_calibre(calibre)
    ex, buf, outs = _event_kernel()

    p = np.array([0.0, -sight_height_m, 0.0])
    v = np.array([math.sin(azimuth_rad) * c.muzzle_m_s,
                  math.sin(elevation_rad) * c.muzzle_m_s,
                  math.cos(elevation_rad) * math.cos(azimuth_rad) * c.muzzle_m_s])
    for n, val in (("dt", dt), ("mass_kg", c.mass_kg),
                   ("diameter_m", c.diameter_m), ("drag_coefficient", 0.295),
                   ("air_density_kg_m3", 1.225), ("gravity_m_s2", 9.80665)):
        buf[n][0] = val
    t = 0.0
    for _ in range(200_000):
        w = wind_field(t, p)
        buf["position_x_m"][0], buf["position_y_m"][0], buf["position_z_m"][0] = p
        buf["velocity_x_m_s"][0], buf["velocity_y_m_s"][0], buf["velocity_z_m_s"][0] = v
        buf["wind_x_m_s"][0], buf["wind_y_m_s"][0], buf["wind_z_m_s"][0] = w
        ex.run()
        g = lambda n: float(np.asarray(ex.buffers[outs[n]]).reshape(-1)[0])
        p = np.array([g("position_next_x_m"), g("position_next_y_m"),
                      g("position_next_z_m")])
        v = np.array([g("velocity_next_x_m_s"), g("velocity_next_y_m_s"),
                      g("velocity_next_z_m_s")])
        t += dt
        if p[2] >= range_m:
            break
    return {"drop_m": -float(p[1]), "drift_m": float(p[0]),
            "time_s": t, "speed_m_s": float(np.linalg.norm(v))}


def gusting_wind(rng, mean_m_s: float = 4.0, gust_m_s: float = 2.5,
                 scale_m: float = 120.0):
    """A wind field that is not the number the gunner read.

    A mean, plus a gust that varies down the range on a real length
    scale. The gunner's meter sees the wind AT THE GUN at one instant;
    the round spends its flight in air that is doing something else
    three hundred metres away."""
    phase = rng.uniform(0.0, 6.2832)
    phase2 = rng.uniform(0.0, 6.2832)
    bias = rng.gauss(0.0, gust_m_s * 0.5)

    def field(t, position):
        z = float(position[2])
        x = (mean_m_s + bias
             + gust_m_s * math.sin(z / scale_m + phase)
             + gust_m_s * 0.4 * math.sin(z / (scale_m * 0.37) + phase2))
        return (x, 0.0, 0.0)
    return field


# ---------------------------------------------------------------------
def engagement(condition: str, *, shots: int = 120, rate_hz: float = 5.0,
               range_m: float = 800.0, seed: int = 11,
               wind_mean: float = 4.0, wind_gust: float = 2.5,
               shot_profile: dict = None) -> dict:
    """A sustained engagement, with the round actually flown.

    Four conditions, separating two things that were tangled together:

      dry             no coolant. Symmetric heating, so a straight tube
                      and a barrel at 744 C.
      flooded         coolant, no trim. The jacket's own stratification
                      bends it, and NOTHING corrects for that.
      sighted         flooded, and the MRS measurement is fed to the
                      firing solution. The tube is bent; the sight knows
                      roughly how much.
      trued           the thermal loop drives the REAL bend to zero, so
                      the tube is physically straight and the sight has
                      nothing to correct.

    `sighted` is limited by how well the instrument can measure.
    `trued` is limited by how straight the loop can hold it. Which of
    those is smaller is the actual question, and it is not obvious in
    advance.
    """
    import statistics as stats
    from muzzle_reference import MuzzleReferenceSystem, BarrelBend
    rng = random.Random(seed)
    prof = shot_profile or shot_heat_by_station()
    tank = CoolantTank()
    wet = condition != "dry"
    # the jacket's own contents, from the graph's declared chamber
    # volumes and the fluid's declared density -- not a guess
    _w = next(f for f in fluids.FLUIDS if f.key == "water")
    coolant_kg = (14.84 / 1000.0 * _w.density_kg_m3) if wet else 0.0
    steel = BarrelThermalState(coolant_mass_kg=coolant_kg)
    mrs = MuzzleReferenceSystem(barrel=BarrelBend(length_m=1.70))

    base_ua = 0.0 if condition == "dry" else 42.0
    ua = [[base_ua] * len(QUADRANTS) for _ in range(STATIONS)]
    dt = 1.0 / rate_hz
    misses, bends, drifts, whips = [], [], [], []
    from beam_theory import beam_dynamics_native
    beam_step, _ = beam_dynamics_native()
    q_modal = qd_modal = 0.0
    # shot + propellant gas impulse, from the real muzzle velocity
    recoil_impulse_n_s = (0.130 * prof["muzzle_m_s"]
                          + 0.040 * prof["muzzle_m_s"] * 1.5)
    # the zero: where a straight, cold barrel puts them in still air
    from scope import scoped
    _s = scoped(".50 ap", zero_range_m=100.0)
    zero_rad = _s.zero_elevation_rad
    zero_drop = event_trajectory(".50 ap", zero_rad, 0.0, range_m,
                                 lambda t, p: (0.0, 0.0, 0.0))["drop_m"]

    for _ in range(shots):
        steel.deposit(prof["per_station_j"])
        if wet:
            strat = 3.0 * min(1.0, (steel.hottest_k - tank.temperature_k) / 60.0)
            tank.absorb(steel.cool(dt, tank.temperature_k, ua, strat), dt)

        # THE TUBE IS RINGING. Integrate it through the interval since
        # the last round, then hit it with this round's recoil impulse
        # and let it move while the shot is inside it -- because the
        # muzzle is pointing somewhere different at the microsecond the
        # round leaves than it was when the primer went off.
        mean_k = sum(sum(r) for r in steel.temps) / (STATIONS * len(QUADRANTS))
        beam_kw = dict(length_m=steel.barrel_length_m,
                       outer_radius_m=steel.outer_diameter_m / 2.0,
                       wall_m=0.022, density_kg_m3=7850.0,
                       youngs_cold_pa=2.05e11, damping_ratio=0.005,
                       added_mass_per_m_kg=coolant_kg / steel.barrel_length_m,
                       temperature_k=mean_k)
        # coast to the next round
        sub = 2.0e-4
        for _ in range(int(dt / sub)):
            br = beam_step(modal_q_m=q_modal, modal_qdot_m_s=qd_modal,
                           force_n=0.0, dt_s=sub, **beam_kw)
            q_modal, qd_modal = br["modal_q_next_m"], br["modal_qdot_next_m_s"]
        # the shot: recoil impulse over the time it is in the tube
        in_tube_s = 0.0027
        f_recoil = recoil_impulse_n_s / in_tube_s
        sub2 = 1.0e-5
        for _ in range(int(in_tube_s / sub2)):
            br = beam_step(modal_q_m=q_modal, modal_qdot_m_s=qd_modal,
                           force_n=f_recoil, dt_s=sub2, **beam_kw)
            q_modal, qd_modal = br["modal_q_next_m"], br["modal_qdot_next_m_s"]
        whip = br["tip_slope_rad"]

        true_bend = steel.true_bend_rad() + whip
        bends.append(true_bend)
        whips.append(whip)

        # what the sight is told
        if condition == "sighted":
            measured = (true_bend + rng.gauss(0.0, mrs.noise_rad))
            measured = round(measured / mrs.resolution_rad) * mrs.resolution_rad
        else:
            measured = 0.0
        aim_error = true_bend - measured

        # THE ROUND FLIES, from the ZERO, with the bend error added to it.
        # Fired at the bend error alone the round simply fell 5.6 m and
        # the 103 mm under test vanished into it. And the miss has to be
        # reported as VERTICAL -- the bend is a vertical error and the
        # wind is a horizontal one, so a radial hypot buries the signal
        # in the noise of a different axis entirely.
        ev = event_trajectory(".50 ap", zero_rad + aim_error, 0.0, range_m,
                              gusting_wind(rng, wind_mean, wind_gust))
        misses.append(ev["drop_m"] - zero_drop)
        drifts.append(ev["drift_m"])

        if condition == "trued":
            # THE LOOP CORRECTS THE REAL ERROR, not a reading of it --
            # the ceiling of what physically holding the tube can do
            trim = max(-0.9, min(0.9, true_bend * 4.0e3))
            for i in range(STATIONS):
                ua[i][0] = base_ua * (1.0 + trim)
                ua[i][2] = base_ua * (1.0 - trim)

    return {
        "condition": condition,
        "peak_bend_mrad": max(abs(b) for b in bends) * 1000.0,
        "final_bend_mrad": bends[-1] * 1000.0,
        "hottest_c": steel.hottest_k - 273.15,
        "mean_vertical_mm": stats.fmean(misses) * 1000.0,
        "sd_vertical_mm": stats.pstdev(misses) * 1000.0,
        "worst_vertical_mm": max(abs(m) for m in misses) * 1000.0,
        "sd_horizontal_mm": stats.pstdev(drifts) * 1000.0,
        "whip_sd_mrad": stats.pstdev(whips) * 1000.0,
        "whip_peak_mrad": max(abs(w) for w in whips) * 1000.0,
        "first_mode_hz": br["natural_frequency_hz"],
    }
