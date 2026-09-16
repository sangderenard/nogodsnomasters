"""The test cell as one procedure: start it, spin it, load it, log it.

A dyno pull already exists (EngineCycleSim.start_wot_dyno_pull) and it
is the right instrument for exactly one number. It is an INERTIA pull:
the drum accelerates, torque is only tracked while the clutch is locked,
and on the AMC 258 that means a sampled window of roughly 3140-4300 rpm.
Its peak power figure is trustworthy. Its "peak torque" is the largest
torque in that window, and the engine's real peak at 1800 rpm is never
visited at all.

A profile is the other thing a cell does. It HOLDS each speed while the
absorber eats the difference, waits for the engine to settle, and then
records everything -- which is the only way to get a torque curve, an
energy balance, or a fuel map, because all three need the engine to be
in equilibrium when the reading is taken.

THREE PHASES, BECAUSE THEY MEASURE THREE DIFFERENT THINGS

  START      The real starting system, through engage_starter -- not
             start(), which teleports. This is the only phase that can
             fail outright, and what it costs (bus sag, cranking
             current, receiver air, seconds to catch) is real data about
             the engine and its battery, not ceremony.

  NEUTRAL    Gearbox in neutral: the ENGINE ALONE, with no transmission
             drag and nothing downstream. Two measurements fall out, and
             both are real dyno techniques:

               free acceleration -- with no load at all, net torque is
                   simply J.dw/dt against the crank's own inertia. This
                   is an inertia dyno using the engine's own rotating
                   mass as the drum.
               overrun deceleration -- throttle shut, and the only thing
                   slowing it is friction, so J.dw/dt IS the friction
                   torque. This is how FMEP is really measured, and it
                   is the one number the geared phase cannot separate.

             The cell's exhaust is what a bare engine on a stand runs:
             a short dump, not the vehicle's cat and muffler.

  GEARED     Through the transmission, absorber holding each point, with
             the engine's OWN declared exhaust fitted -- the full system
             it would have installed. That difference is the real one
             between an engine dyno and a chassis dyno, and it is why
             the same engine reads differently on each.

WHAT "FULLY INSTRUMENTED" MEANS HERE. The sim carries 149 state fields.
A reading takes everything relevant at every held point: speed and
torque at BOTH shafts (never conflated -- the drum is post-gearing),
derived BMEP and volumetric efficiency, fuel flow and the specific
consumption and thermal efficiency that follow from it, every
temperature in the machine, both pressures that matter, combustion
health, per-cylinder film and carbon and float speed, the absorber's own
control setting and thermal state, and the harm chain's verdict on the
lubrication regime and the running fit.

AND AN ENERGY BALANCE THAT IS ALLOWED TO NOT CLOSE. Fuel power in,
against brake power out plus exhaust enthalpy plus rejected heat. The
residual is reported rather than absorbed into a fudge, because a
balance that always closes is one that has been made to, and its
closure then says nothing. A residual that drifts with speed is a real
statement that something is unaccounted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

RPM_TO_RAD_S = 2.0 * math.pi / 60.0

#: How long to hold a point before believing it. A held speed has to
#: settle the absorber loop AND let the gas path catch up; the exhaust
#: circuit's own thermal mass is the slow one.
DEFAULT_SETTLE_S = 1.5
#: And how long to average over once settled, so a reading is a mean
#: rather than whatever one tick happened to say.
DEFAULT_RECORD_S = 0.5

#: Air at the intake, for the volumetric-efficiency denominator. Real
#: sea-level density; the same figure engine_sim's own flow ceiling uses.
AIR_DENSITY_KG_M3 = 1.2
EXHAUST_CP_J_PER_KGK = 1150.0    # burnt gas, real, higher than dry air's 1005
AMBIENT_K = 293.15


@dataclass
class Reading:
    """One held operating point, with everything the cell can see."""
    rpm: float = 0.0
    # --- both shafts, never conflated ---------------------------------
    crank_torque_nm: float = 0.0
    crank_power_kw: float = 0.0
    drum_rpm: float = 0.0
    drum_torque_nm: float = 0.0
    drum_absorbed_kw: float = 0.0
    # --- derived from geometry ----------------------------------------
    bmep_pa: float = 0.0
    #: DERIVED, not measured -- see the note in _reading. The sim does
    #: not instrument an actual inducted air mass, so this comes from the
    #: head's own port-flow model and is labelled as such rather than
    #: presented alongside measurements as though it were one.
    volumetric_efficiency_derived: float = 0.0
    #: MEASURED: the charge-density multiplier the sim actually applied
    #: (boost, charge temperature, altitude). This is what
    #: _intake_demand_kg_s is scaled by, and it is a real reading.
    charge_density_frac: float = 0.0
    valve_health_frac: float = 0.0        # cylinder_breathing_frac: valve condition, 1.0 when fresh
    ideal_air_kg_s: float = 0.0           # what displacement alone would draw
    mean_piston_speed_m_s: float = 0.0
    # --- fuel -----------------------------------------------------------
    fuel_kg_s: float = 0.0
    fuel_power_kw: float = 0.0
    bsfc_g_per_kwh: float = 0.0
    thermal_efficiency: float = 0.0
    mixture_phi: float = 0.0
    # --- air path -------------------------------------------------------
    air_kg_s: float = 0.0
    manifold_pressure_frac: float = 0.0
    boost_frac: float = 0.0
    intake_charge_temp_k: float = 0.0
    # --- exhaust --------------------------------------------------------
    exhaust_temp_k: float = 0.0
    exhaust_tailpipe_temp_k: float = 0.0
    exhaust_pressure_frac: float = 0.0
    # --- thermal --------------------------------------------------------
    coolant_temp_k: float = 0.0
    oil_temp_k: float = 0.0
    oil_pressure_pa: float = 0.0
    bay_air_temp_k: float = 0.0
    # --- combustion health ----------------------------------------------
    knock_flag: bool = False
    knock_intensity: float = 0.0
    misfire_flag: bool = False
    preignition_flag: bool = False
    ignition_timing_deg: float = 0.0
    valve_float_risk: float = 0.0
    # --- per cylinder ----------------------------------------------------
    cylinder_oil_film_mg: tuple = ()
    cylinder_block_temps_k: tuple = ()
    cylinder_carbon_frac: tuple = ()
    cylinder_float_rpm: tuple = ()
    # --- the absorber itself ---------------------------------------------
    absorber_kind: str = ""
    absorber_control: float = 0.0
    absorber_saturated: bool = False
    absorber_capacity_nm: float = 0.0
    absorber_note: str = ""
    # --- the harm chain's verdict ----------------------------------------
    lubrication_regime: str = ""
    specific_film: float = 0.0
    friction_heat_w: float = 0.0
    running_clearance_m: float = 0.0
    seized: bool = False
    rod_margin: float = 0.0
    # --- energy balance, allowed not to close -----------------------------
    exhaust_enthalpy_kw: float = 0.0
    rejected_heat_kw: float = 0.0
    balance_residual_kw: float = 0.0
    settled: bool = True

    @property
    def bhp(self) -> float:
        import units
        return units.kw_to_hp(self.crank_power_kw)

    @property
    def lb_ft(self) -> float:
        import units
        return units.lb_ft(self.crank_torque_nm)


@dataclass
class StartRecord:
    """What starting this engine actually took."""
    mode: str = ""
    caught: bool = False
    seconds_to_catch: float = 0.0
    peak_crank_rpm: float = 0.0
    peak_bus_current_a: float = 0.0
    min_bus_voltage_v: float = 0.0
    air_used_kg: float = 0.0
    ring_gear_damage: float = 0.0
    note: str = ""


@dataclass
class NeutralRecord:
    """The engine by itself: what accelerates it and what slows it."""
    free_rev_points: list = field(default_factory=list)   # (rpm, net_torque_nm)
    overrun_points: list = field(default_factory=list)    # (rpm, friction_torque_nm)
    crank_inertia_kg_m2: float = 0.0
    peak_free_accel_rad_s2: float = 0.0
    note: str = ""


@dataclass
class Profile:
    identity: str = ""
    start: StartRecord = field(default_factory=StartRecord)
    neutral: NeutralRecord = field(default_factory=NeutralRecord)
    sweep: list = field(default_factory=list)             # list[Reading]
    seconds: float = 0.0
    warnings: list = field(default_factory=list)

    def peak_torque(self):
        settled = [r for r in self.sweep if r.settled]
        return max(settled, key=lambda r: r.crank_torque_nm, default=None)

    def peak_power(self):
        settled = [r for r in self.sweep if r.settled]
        return max(settled, key=lambda r: r.crank_power_kw, default=None)


# ---------------------------------------------------------------------


def _state_tuple(st, name):
    v = getattr(st, name, None)
    return tuple(v) if isinstance(v, (list, tuple)) else ()


def _reading(sim, engine, absorber, acc: dict, n: int) -> Reading:
    """Turn an averaged accumulation into one fully-populated Reading."""
    st = sim.state
    r = Reading()
    inv = 1.0 / max(n, 1)
    r.rpm = acc["rpm"] * inv
    r.crank_torque_nm = acc["torque"] * inv
    r.crank_power_kw = r.crank_torque_nm * r.rpm * RPM_TO_RAD_S / 1000.0
    r.drum_rpm = acc["drum_rpm"] * inv
    r.drum_torque_nm = acc["drum_torque"] * inv
    r.drum_absorbed_kw = acc["absorbed"] * inv
    r.fuel_kg_s = acc["fuel"] * inv

    arch = engine.architecture
    disp_m3 = engine.displacement_l / 1000.0
    cycle_rad = 4.0 * math.pi if arch.cycle_degrees >= 720 else 2.0 * math.pi
    r.bmep_pa = r.crank_torque_nm * cycle_rad / max(disp_m3, 1e-9)
    r.mean_piston_speed_m_s = 2.0 * arch.stroke_m * r.rpm / 60.0
    # VOLUMETRIC EFFICIENCY IS NOT INSTRUMENTED, and saying so is the
    # only honest thing to print. Two wrong answers were tried first,
    # both of which looked completely plausible:
    #
    #   _intake_demand_kg_s / swept  -- but demand IS swept x
    #       charge_density_frac (engine_cycle_sim:3221), the IDEAL draw
    #       that never passes through port flow. The ratio returns
    #       charge_density_frac and nothing else: the same number twice,
    #       reported as a flat VE near 0.98 across the whole range.
    #   cylinder_breathing_frac -- but valve_state:27 defines that as a
    #       product over the intake valves' CONDITION. It is 1.00 on a
    #       fresh engine by construction, which is exactly what it then
    #       printed.
    #
    # The sim never records an actual inducted air mass, so a profile
    # cannot measure VE. What it CAN do is report the head's own derived
    # figure, clearly labelled as derived, next to the charge density it
    # really did apply.
    swept_kg_s = disp_m3 * (r.rpm / 60.0) / (2.0 if arch.cycle_degrees >= 720 else 1.0) * AIR_DENSITY_KG_M3
    r.air_kg_s = acc["air"] * inv
    r.ideal_air_kg_s = swept_kg_s
    r.charge_density_frac = (r.air_kg_s / swept_kg_s) if swept_kg_s > 0.0 else 0.0
    health = _state_tuple(st, "cylinder_breathing_frac")
    r.valve_health_frac = (sum(health) / len(health)) if health else 0.0
    try:
        import port_flow as _pf
        import cylinder_ports as _cp
        try:
            n_valves = _cp.valves_per_cylinder(engine)
        except Exception:
            n_valves = 2
        lift = getattr(getattr(engine, "lifter_spring", None), "valve_lift_mm", 9.5) / 1000.0
        # built exactly as derived_torque builds it, so the two agree by
        # construction instead of by coincidence
        ports = _pf.PortSet(side="intake", bore_m=arch.bore_m, stroke_m=arch.stroke_m,
                            valves_total=n_valves, lift_m=lift)
        # derived_torque.port_limited_ve is the project's single VE
        # definition. PortSet.volumetric_efficiency alone is the port
        # CHOKE fraction and reads 1.00 below the choke speed, which is
        # what this column printed before.
        import derived_torque as _dt
        r.volumetric_efficiency_derived = _dt.port_limited_ve(ports, n_valves, r.rpm)
    except Exception:
        r.volumetric_efficiency_derived = 0.0

    # fuel energy in, from the fuel actually being burned
    try:
        import working_fluids as wf
        spec = wf.WORKING_FLUIDS.get(getattr(engine, "preferred_fuel_profile", "") or "")
        lhv = float(getattr(spec, "energy_density_j_per_kg", 0.0) or 0.0) if spec else 0.0
    except Exception:
        lhv = 0.0
    if lhv <= 0.0:
        lhv = 44.0e6     # a petroleum fuel; only used when nothing is declared
    r.fuel_power_kw = r.fuel_kg_s * lhv / 1000.0
    if r.crank_power_kw > 0.0 and r.fuel_kg_s > 0.0:
        r.bsfc_g_per_kwh = r.fuel_kg_s * 3.6e6 / r.crank_power_kw
        r.thermal_efficiency = r.crank_power_kw / max(r.fuel_power_kw, 1e-9)

    r.mixture_phi = float(getattr(st, "mixture_phi", 0.0) or 0.0)
    r.manifold_pressure_frac = float(getattr(st, "manifold_pressure_frac", 0.0) or 0.0)
    r.boost_frac = float(getattr(st, "boost_frac", 0.0) or 0.0)
    r.intake_charge_temp_k = float(getattr(st, "intake_charge_temp_k", 0.0) or 0.0)
    r.exhaust_temp_k = float(getattr(st, "exhaust_temp_k", 0.0) or 0.0)
    r.exhaust_tailpipe_temp_k = float(getattr(st, "exhaust_tailpipe_temp_k", 0.0) or 0.0)
    r.exhaust_pressure_frac = float(getattr(st, "exhaust_pressure_frac", 0.0) or 0.0)
    r.coolant_temp_k = float(getattr(st, "coolant_temp_k", 0.0) or 0.0)
    r.oil_temp_k = float(getattr(st, "oil_temp_k", 0.0) or 0.0)
    r.oil_pressure_pa = float(getattr(st, "oil_pressure_pa", 0.0) or 0.0)
    r.bay_air_temp_k = float(getattr(st, "bay_air_temp_k", 0.0) or 0.0)
    r.knock_flag = bool(getattr(st, "knock_flag", False))
    r.knock_intensity = float(getattr(st, "knock_intensity", 0.0) or 0.0)
    r.misfire_flag = bool(getattr(st, "misfire_flag", False))
    r.preignition_flag = bool(getattr(st, "preignition_flag", False))
    r.ignition_timing_deg = float(getattr(st, "ignition_timing_deg", 0.0) or 0.0)
    r.valve_float_risk = float(getattr(st, "valve_float_risk", 0.0) or 0.0)
    r.cylinder_oil_film_mg = _state_tuple(st, "cylinder_oil_film_mg")
    r.cylinder_block_temps_k = _state_tuple(st, "cylinder_block_temps_k")
    r.cylinder_carbon_frac = _state_tuple(st, "cylinder_carbon_frac")
    r.cylinder_float_rpm = _state_tuple(st, "cylinder_float_rpm")

    # the absorber: what it was asked for and whether it could give it
    if absorber is not None:
        cap = absorber.capacity_nm(r.drum_rpm if r.drum_rpm > 0.0 else r.rpm)
        held = absorber.hold(r.drum_rpm if r.drum_rpm > 0.0 else r.rpm,
                             abs(r.drum_torque_nm))
        r.absorber_kind = absorber.identity
        r.absorber_capacity_nm = cap
        r.absorber_control = held.control
        r.absorber_saturated = held.saturated
        r.absorber_note = held.note

    # the harm chain, read at this point rather than asserted
    try:
        import engine_harm as eh
        film_mg = r.cylinder_oil_film_mg
        film_kg = (min(film_mg) * 1e-6) if film_mg else 4e-4
        # normal load on the skirt: a real thrust load is a fraction of
        # the gas load, and the gas load is bmep times piston area
        area = math.pi * arch.bore_m ** 2 / 4.0
        thrust_n = max(1.0, r.bmep_pa * area * 0.1)
        lub = eh.lubrication(film_kg, arch.bore_m, arch.stroke_m,
                             thrust_n, r.mean_piston_speed_m_s,
                             oil_temp_k=max(r.oil_temp_k, 280.0))
        r.lubrication_regime = lub.regime
        r.specific_film = lub.specific_film
        r.friction_heat_w = lub.friction_heat_w
        temps = r.cylinder_block_temps_k
        piston_k = (max(temps) if temps else r.coolant_temp_k + 60.0)
        fit = eh.running_fit(arch.bore_m, piston_k, r.coolant_temp_k + 20.0)
        r.running_clearance_m = fit.hot_clearance_m
        r.seized = fit.seized
        r.rod_margin = eh.over_rev(engine, r.rpm).rod_margin
    except Exception:
        pass

    # the balance, reported rather than forced
    exh_kg_s = r.air_kg_s + r.fuel_kg_s
    r.exhaust_enthalpy_kw = (exh_kg_s * EXHAUST_CP_J_PER_KGK
                             * max(0.0, r.exhaust_temp_k - AMBIENT_K) / 1000.0)
    r.rejected_heat_kw = float(getattr(sim, "_waste_heat_kw", 0.0) or 0.0)
    r.balance_residual_kw = (r.fuel_power_kw - r.crank_power_kw
                             - r.exhaust_enthalpy_kw - r.rejected_heat_kw)
    return r


def _run_start(sim, dt: float, limit_s: float) -> StartRecord:
    """Crank it for real, and record what that cost."""
    rec = StartRecord()
    try:
        rec.mode = sim.starter.mode
    except Exception:
        rec.mode = "?"
    sim.engage_starter()
    t = 0.0
    min_v = 1e9
    while t < limit_s:
        sim.step(dt)
        t += dt
        st = sim.state
        rec.peak_crank_rpm = max(rec.peak_crank_rpm, float(st.rpm))
        try:
            rd = sim.starter.reading
            rec.peak_bus_current_a = max(rec.peak_bus_current_a, float(rd.bus_current_a))
            rec.air_used_kg += float(rd.air_kg_s) * dt
            if rd.note:
                rec.note = rd.note
        except Exception:
            pass
        v = float(getattr(st, "bus_voltage_v", 0.0) or 0.0)
        if v > 0.0:
            min_v = min(min_v, v)
        if not st.stalled and st.rpm >= sim.engine.idle_rpm * 0.9:
            rec.caught = True
            rec.seconds_to_catch = t
            break
    rec.min_bus_voltage_v = 0.0 if min_v >= 1e9 else min_v
    try:
        rec.ring_gear_damage = max(s.ring_gear_damage for s in sim.starter.systems)
    except Exception:
        pass
    if not rec.caught:
        rec.note = (rec.note or "") + f" -- did not catch in {limit_s:.0f} s"
    return rec


def _run_neutral(sim, engine, dt: float) -> NeutralRecord:
    """The engine alone: free acceleration, then overrun."""
    rec = NeutralRecord()
    j = float(getattr(engine, "inertia_kg_m2", 0.0) or 0.0)
    rec.crank_inertia_kg_m2 = j
    if j <= 0.0:
        rec.note = "engine declares no rotating inertia; J.dw/dt cannot be read"
        return rec
    sim.gear_index = 0          # neutral: nothing downstream
    sim.brake_target_rpm = None
    sim.brake_load_nm = 0.0
    redline = float(getattr(engine, "redline_rpm", 4000.0) or 4000.0)

    # --- free acceleration: net torque is J.dw/dt ----------------------
    sim.throttle = 1.0
    prev = sim.state.rpm
    t = 0.0
    while sim.state.rpm < redline * 0.98 and t < 30.0:
        sim.step(dt)
        t += dt
        now = sim.state.rpm
        alpha = (now - prev) * RPM_TO_RAD_S / dt
        if alpha > 0.0 and now > 0.0:
            rec.free_rev_points.append((now, j * alpha))
            rec.peak_free_accel_rad_s2 = max(rec.peak_free_accel_rad_s2, alpha)
        prev = now

    # --- overrun: the only torque left is friction ---------------------
    sim.throttle = 0.0
    prev = sim.state.rpm
    t = 0.0
    while sim.state.rpm > engine.idle_rpm * 1.05 and t < 30.0:
        sim.step(dt)
        t += dt
        now = sim.state.rpm
        decel = (prev - now) * RPM_TO_RAD_S / dt
        if decel > 0.0 and now > 0.0:
            rec.overrun_points.append((now, j * decel))
        prev = now
    rec.note = ("free acceleration gives NET torque (J.dw/dt with nothing "
                "loading it); overrun gives FRICTION torque directly, which "
                "is the one quantity the geared sweep cannot separate")
    return rec


def profile(engine, points: int = 8, settle_s: float = DEFAULT_SETTLE_S,
            record_s: float = DEFAULT_RECORD_S, absorber_kind: str = "eddy-current",
            gear_index: int = 4, do_start: bool = True, do_neutral: bool = True,
            max_seconds: float = 400.0) -> Profile:
    """Run the whole cell procedure and return everything it saw.

    COSTS REAL SIMULATED TIME, and that is the price of measuring rather
    than asserting: every held point has to settle before it means
    anything. Reduce `points` before reducing `settle_s` -- a reading
    taken before the gas path has caught up is not a cheaper reading, it
    is a wrong one, and `settled` on each Reading says whether the
    absorber actually got there.
    """
    import dataclasses as _dc
    import engine_cycle_sim as ecs
    import dyno_brakes

    prof = Profile(identity=str(getattr(engine, "identity", "?")))
    dt = ecs.FIXED_PHYSICS_DT_S
    # the absorber is on the DRUM, behind the gearbox and final drive,
    # so it must be sized for what that shaft actually sees
    overall = 1.0
    try:
        tx = engine.transmission
        ratios = tuple(tx.gear_ratios or ())
        if ratios:
            i = max(0, min(len(ratios) - 1, int(gear_index) - 1))
            overall = ratios[i] * float(tx.final_drive_ratio or 1.0)
    except Exception:
        prof.warnings.append("no transmission ratios; absorber sized at the crank")
    absorber = dyno_brakes.for_engine(engine, absorber_kind, gear_ratio=overall)

    # --- the bare-engine phases run on a CELL exhaust -------------------
    # a stand engine has a short dump, not the vehicle's cat and muffler
    bare = engine
    try:
        ex = engine.exhaust_system
        bare = _dc.replace(engine, exhaust_system=_dc.replace(ex, header_type="open-header"))
    except Exception:
        prof.warnings.append("could not fit a cell exhaust; bare phases ran on the "
                             "engine's own declared system")

    elapsed = 0.0
    if do_start or do_neutral:
        sim = ecs.EngineCycleSim(engine=bare)
        if do_start:
            prof.start = _run_start(sim, dt, limit_s=20.0)
            elapsed += prof.start.seconds_to_catch or 20.0
            if not prof.start.caught:
                prof.warnings.append("start did not catch; neutral and geared "
                                     "phases run from a warm start instead")
                sim.start()
        else:
            sim.start()
        if do_neutral:
            prof.neutral = _run_neutral(sim, bare, dt)

    # --- the geared sweep, on the engine's OWN declared exhaust ---------
    sim = ecs.EngineCycleSim(engine=engine)
    sim.start()
    sim.gear_index = max(1, gear_index)
    sim.throttle = 1.0
    redline = float(getattr(engine, "redline_rpm", 4000.0) or 4000.0)
    lo = max(float(getattr(engine, "idle_rpm", 600.0) or 600.0) * 1.3, redline * 0.2)
    targets = [lo + (redline - lo) * i / max(1, points - 1) for i in range(points)]

    for target in targets:
        sim.brake_target_rpm = target
        sim.throttle = 1.0
        t = 0.0
        while t < settle_s and elapsed < max_seconds:
            sim.step(dt)
            t += dt
            elapsed += dt
        settled = abs(sim.state.rpm - target) <= max(25.0, target * 0.04)
        acc = {k: 0.0 for k in ("rpm", "torque", "drum_rpm", "drum_torque",
                                "absorbed", "fuel", "air")}
        n = 0
        t = 0.0
        while t < record_s and elapsed < max_seconds:
            sim.step(dt)
            t += dt
            elapsed += dt
            st = sim.state
            acc["rpm"] += st.rpm
            acc["torque"] += st.current_torque_nm
            acc["drum_rpm"] += st.dyno_rpm
            acc["drum_torque"] += st.dyno_torque_nm
            acc["absorbed"] += st.dyno_absorbed_kw
            # THE ACTUAL BURN, not the stoichiometric demand.
            # engine_cycle_sim:2003 is explicit: fuel_kg_s =
            # _fuel_demand_kg_s * min(mixture_phi, 2.0). Reading demand
            # alone reports a stoichiometric engine no matter how rich
            # it is actually running, and at WOT no petrol engine is.
            acc["fuel"] += (float(getattr(sim, "_fuel_demand_kg_s", 0.0) or 0.0)
                            * min(float(getattr(st, "mixture_phi", 1.0) or 1.0), 2.0))
            acc["air"] += float(getattr(sim, "_intake_demand_kg_s", 0.0) or 0.0)
            n += 1
        if n == 0:
            prof.warnings.append(f"ran out of time before {target:.0f} rpm")
            break
        r = _reading(sim, engine, absorber, acc, n)
        r.settled = settled
        if not settled:
            prof.warnings.append(
                f"{target:.0f} rpm: absorber held {sim.state.rpm:.0f} instead -- "
                "reading marked unsettled")
        prof.sweep.append(r)

    sim.brake_target_rpm = None
    prof.seconds = elapsed
    return prof


def report(prof: Profile) -> str:
    import units
    L = [f"dyno profile: {prof.identity}   ({prof.seconds:.1f} simulated seconds)"]
    s = prof.start
    L.append("")
    if not s.mode:
        L.append("  START  not run")
        s = None
    if s is not None:
        L.append(f"  START [{s.mode}]  {'caught' if s.caught else 'DID NOT CATCH'}"
                 + (f" in {s.seconds_to_catch:.2f} s" if s.caught else "")
                 + f", cranked to {s.peak_crank_rpm:.0f} rpm")
        if s.peak_bus_current_a:
            L.append(f"        peak {s.peak_bus_current_a:.0f} A off the bus"
                     + (f", sagging to {s.min_bus_voltage_v:.1f} V" if s.min_bus_voltage_v else ""))
        if s.air_used_kg:
            L.append(f"        {s.air_used_kg * 1000:.0f} g of starting air")
        if s.ring_gear_damage:
            L.append(f"        ring gear {s.ring_gear_damage * 100:.1f}% ground away")
        if s.note:
            L.append(f"        {s.note}")

    n = prof.neutral
    L.append("")
    if n.free_rev_points:
        top = max(n.free_rev_points, key=lambda p: p[1])
        L.append(f"  NEUTRAL (engine alone, J = {n.crank_inertia_kg_m2:.3f} kg.m2)")
        L.append(f"        free accel peak net torque {units.torque(top[1])} at {top[0]:.0f} rpm")
        if n.overrun_points:
            lo_f = min(n.overrun_points, key=lambda p: p[0])
            hi_f = max(n.overrun_points, key=lambda p: p[0])
            L.append(f"        friction from overrun: {units.torque(lo_f[1])} at "
                     f"{lo_f[0]:.0f} rpm to {units.torque(hi_f[1])} at {hi_f[0]:.0f} rpm")
    else:
        L.append("  NEUTRAL  not run")

    L.append("")
    L.append("  GEARED SWEEP (engine's own exhaust, absorber holding each point)")
    L.append(f"    {'rpm':>5s} {'torque':>18s} {'power':>18s} {'bmep':>7s} {'VE':>5s} "
             f"{'BSFC':>7s} {'eff':>5s} {'exh K':>6s} {'phi':>5s}  flags")
    for r in prof.sweep:
        flags = []
        if not r.settled:
            flags.append("UNSETTLED")
        if r.knock_flag:
            flags.append(f"knock {r.knock_intensity:.2f}")
        if r.misfire_flag:
            flags.append("misfire")
        if r.absorber_saturated:
            flags.append("absorber saturated")
        if r.seized:
            flags.append("SEIZED")
        if r.lubrication_regime and r.lubrication_regime != "hydrodynamic":
            flags.append(r.lubrication_regime)
        film = min(r.cylinder_oil_film_mg) if r.cylinder_oil_film_mg else 0.0
        L.append(f"    {r.rpm:5.0f} {units.torque(r.crank_torque_nm):>18s} "
                 f"{units.power(r.crank_power_kw):>18s} {r.bmep_pa / 1e5:6.2f}b "
                 f"{r.volumetric_efficiency_derived:5.2f} {r.bsfc_g_per_kwh:7.0f} "
                 f"{r.thermal_efficiency:5.2f} {r.exhaust_temp_k:6.0f} {r.mixture_phi:5.2f}  "
                 + ", ".join(flags))

    pt, pp = prof.peak_torque(), prof.peak_power()
    L.append("")
    if pt:
        L.append(f"  MEASURED peak torque {units.torque(pt.crank_torque_nm)} at {pt.rpm:.0f} rpm"
                 f"   (catalogue says {units.torque(getattr(prof, '_declared_nm', 0.0))} "
                 f"at {getattr(prof, '_declared_rpm', 0.0):.0f})"
                 if hasattr(prof, "_declared_nm") else
                 f"  MEASURED peak torque {units.torque(pt.crank_torque_nm)} at {pt.rpm:.0f} rpm")
    if pp:
        L.append(f"  MEASURED peak power  {units.power(pp.crank_power_kw)} at {pp.rpm:.0f} rpm")
    if prof.sweep:
        res = [abs(r.balance_residual_kw) for r in prof.sweep if r.fuel_power_kw > 0]
        if res:
            L.append(f"  energy balance residual: {min(res):.1f} to {max(res):.1f} kW "
                     "unaccounted (reported, not absorbed)")
    for w in prof.warnings:
        L.append(f"  ! {w}")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    import engines
    ident = sys.argv[1] if len(sys.argv) > 1 else "amc-258-jeep-i6"
    print(report(profile(engines.get(ident))))
