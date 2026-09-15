"""Everything shared between the terminal frontend (main.py, msvcrt
single-char polling) and the pygame frontend (main_pygame.py, real
continuous multi-key state) -- the sim/audio wiring, the dashboard
content, the custom-engine builder form, and the bake-to-disk command.
Neither frontend owns any of this; they just render/poll differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time as _time

import numpy as np

import engines
from engines import Accessories, ForcedInduction, LIFTER_SPRING_PRESETS
from engine_builder import build_custom_engine, PERFORMANCE_PART_DEFAULTS as PERFORMANCE_PRESETS
from engine_cycle_sim import EngineCycleSim
import engine_baker
import wear as wear_module
import cylinder_ports

LISTEN_MODES = ("stereo", "header-solo", "bay-solo")


@dataclass
class LogEntry:
    text: str
    created: float
    kind: str = "info"

    def age_s(self) -> float:
        return _time.perf_counter() - self.created


class EventLog:
    """Transient engine events (knock, misfire, backfire, rev limiter,
    valve float, wastegate flutter, compressor surge) that persist for
    a few seconds instead of vanishing the instant the underlying flag
    clears on the next live-redraw tick. `cooldown_s` throttles a
    single event *kind* to at most one new entry per window, since the
    underlying flags can read True on many consecutive ticks during a
    sustained condition -- without it a bad-fuel knock spree would spam
    one entry per tick instead of reading as one lingering event."""
    def __init__(self, ttl_s: float = 4.0, max_entries: int = 10, cooldown_s: float = 0.6) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.cooldown_s = cooldown_s
        self.entries: list[LogEntry] = []
        self._last_kind_time: dict[str, float] = {}

    def add(self, text: str, kind: str = "info", *, cooldown: bool = True) -> None:
        now = _time.perf_counter()
        if cooldown and now - self._last_kind_time.get(kind, -1e9) < self.cooldown_s:
            return
        if cooldown:
            self._last_kind_time[kind] = now
        self.entries.append(LogEntry(text, now, kind))
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def active(self) -> list[LogEntry]:
        now = _time.perf_counter()
        self.entries = [e for e in self.entries if now - e.created < self.ttl_s]
        return self.entries


def poll_engine_events(sim: EngineCycleSim, log: EventLog) -> None:
    st = sim.state
    if st.knock_flag:
        log.add(f"KNOCK (intensity {st.knock_intensity * 100:.0f}%)", "knock")
    if st.misfire_flag:
        log.add("MISFIRE", "misfire")
    if st.compression_brake_active and not getattr(sim, "_jake_logged", False):
        log.add(f"COMPRESSION-RELEASE BRAKE: {st.compression_brake_torque_nm:.0f} Nm retarding", "limiter")
    sim._jake_logged = st.compression_brake_active
    if st.preignition_flag:
        log.add(f"PRE-IGNITION (hot spot, {st.preignition_intensity * 100:.0f}%) -- piston heating", "knock")
    if st.backfire_flag:
        log.add(f"BACKFIRE ({st.backfire_kind})", "backfire")
    if st.valve_float_flag:
        log.add(f"VALVE FLOAT (risk {st.valve_float_risk * 100:.0f}%)", "float")
    if st.rev_limiter_active:
        log.add("REV LIMITER", "limiter")
    if st.wastegate_flutter:
        log.add("WASTEGATE FLUTTER", "wastegate")
    if st.surge_flag:
        log.add("COMPRESSOR SURGE", "surge")
    if st.dyno_pull_complete_flag:
        log.add(f"DYNO PULL DONE: {st.dyno_pull_peak_torque_nm:.0f} Nm @ {st.dyno_pull_peak_torque_rpm:.0f} rpm, "
                f"{st.dyno_pull_peak_power_kw:.1f} kW @ {st.dyno_pull_peak_power_rpm:.0f} rpm", "dyno_pull")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Listen:
    def __init__(self) -> None:
        self.mode_index = 0

    @property
    def mode(self) -> str:
        return LISTEN_MODES[self.mode_index]

    def cycle(self) -> None:
        self.mode_index = (self.mode_index + 1) % len(LISTEN_MODES)


def make_audio_callback(streamer, listen: Listen):
    """The realtime PortAudio callback: pulls already-rendered samples out
    of an AudioStreamer's ring buffer (audio_stream.py) and never runs
    synthesis itself -- the streamer's own background thread does that,
    continuously, from a snapshot of sim state the physics tick writes."""
    def callback(outdata, frames, time_info, status):
        header, bay = streamer.pull(frames)
        mode = listen.mode
        if mode == "header-solo":
            outdata[:, 0] = header
            outdata[:, 1] = header
        elif mode == "bay-solo":
            outdata[:, 0] = bay
            outdata[:, 1] = bay
        else:
            outdata[:, 0] = header
            outdata[:, 1] = bay
    return callback


def accessory_summary(engine) -> str:
    # real per-accessory introspection for every engine kind -- a
    # cylinders==0 engine isn't necessarily an electric motor with
    # nothing but a coolant pump: the turbine has a real belt-driven
    # alternator (accessories.alternator=True), fully wired through the
    # real 12/24 V circuit solve this session, and the previous
    # cylinders==0 shortcut silently dropped it from this line
    fitted = []
    if engine.accessories.water_pump:
        fitted.append("water pump")
    elif engine.architecture.cylinders:
        fitted.append("air-cooled")
    if engine.accessories.mechanical_fan:
        fitted.append("belt fan")
    if engine.accessories.coolant_pump:
        fitted.append("coolant pump")
    if engine.accessories.alternator:
        fitted.append("alternator")
    elif engine.architecture.cylinders:
        fitted.append("battery-only electrics")
    if not fitted:
        return "none (passively cooled)"
    return ", ".join(fitted)


def current_fuel(sim: EngineCycleSim) -> str:
    return sim.fuel_choice or sim.engine.preferred_fuel_profile


def select_engine_by_index(sim: EngineCycleSim, roster: list, one_based_index: int) -> None:
    """1-9 -> roster[0..8], 0 -> roster[9]. Silently ignored if out of range."""
    idx = 9 if one_based_index == 0 else one_based_index - 1
    if 0 <= idx < len(roster):
        sim.set_engine(roster[idx])


def _bar(frac: float, width: int = 10) -> str:
    n = int(max(0.0, min(1.0, frac)) * width)
    return "#" * n + "." * (width - n)


def tanks_and_controls_lines(sim: EngineCycleSim) -> list[str]:
    """Every real capacity on this engine as one dense block of hash
    bars, plus the state of every switch that governs them.

    The plant's own report is honest but long (sixteen lines), and on a
    780-pixel column that pushed the tanks off the bottom of the screen
    -- which is exactly where a reading you check at a glance must not
    be. This is the same information, gauged rather than narrated: one
    bar per store, several to a line, and the controls as a row of
    on/off names instead of a paragraph each."""
    st = sim.state
    eng = sim.engine
    gauges: list[tuple[str, float, str]] = []

    # fuel, from the real depletable circuit
    gauges.append(("fuel", st.fuel_fill_frac, f"{st.fuel_fill_frac * 100:3.0f}%"))
    # the sump, against its own real capacity
    cs = getattr(sim, "_crankcase_state", None)
    if cs is not None and cs.oil_capacity_kg > 0.0:
        gauges.append(("oil", cs.oil_kg / cs.oil_capacity_kg, f"{st.sump_oil_l:4.1f}L"))
    # the cooling system's own real charge -- what is still in the
    # jacket, radiator and bottle after whatever has been lost out of a
    # hole (state.coolant_lost_l against the circuit's declared volume),
    # with the temperature beside it because on a cooling system those
    # two readings only mean anything together
    drivetrain = getattr(sim, "_drivetrain", None)
    if drivetrain is not None:
        cool = next((c for c in drivetrain.fluid_circuits
                     if c.kind_class == "thermal-liquid" and float(c.volume_l or 0.0) > 0.0
                     and any("coolant" in n or "radiator" in n for n in c.nodes)), None)
        if cool is not None:
            have_l = max(0.0, float(cool.volume_l) - float(getattr(st, "coolant_lost_l", 0.0)))
            gauges.append(("coolant", have_l / max(float(cool.volume_l), 1e-6), f"{have_l:4.1f}L"))

    # an automatic carries its fluid as a working fluid, so the gauge
    # that matters is how much of it is left and how hot it is
    at = getattr(sim, "automatic", None)
    if at is not None:
        gauges.append(("atf", at.fill_frac, f"{at.temp_k - 273.15:3.0f}C"))
        gauges.append(("atf-life", at.life_frac, f"{at.life_frac * 100:3.0f}%"))

    # every sealed case of gears, with what is actually left in it --
    # a differential that has lost its oil is a differential about to
    # weld itself shut, and that deserves a gauge, not a log line
    if drivetrain is not None:
        try:
            import gear_cases as _gc
            for case in _gc.discover(drivetrain.graph, eng):
                lost = sim.hole_emitters.lost_l.get(case.identity, 0.0)
                em = next((e for e in sim.hole_emitters.emitters
                           if e.identity == f"{case.identity}.contents"), None)
                have = (em.contained_l if em is not None
                        else max(0.0, case.capacity_l - lost))
                gauges.append((case.short_label, have / max(case.capacity_l, 1e-6), f"{have:4.1f}L"))
        except Exception:
            pass

    # EVERY PNEUMATIC VESSEL SEPARATELY. The wet tank, the reserve and
    # the two protected brake reservoirs are four real vessels with a
    # pressure-protection valve between them, and collapsing them into
    # one "air" bar hid the single most important thing the airpack has
    # to tell you: whether the brakes still have air after the service
    # side has lost it. air_vessels.py splits the circuit's one
    # authoritative stored mass across them; it does not invent any.
    vs = getattr(sim, "air_vessels", None)
    if vs is not None:
        for label, frac, txt in vs.gauges():
            gauges.append((label, frac, txt))
    # anything else depletable this build carries
    if drivetrain is not None:
        for c in drivetrain.fluid_circuits:
            if c.bottle_capacity_kg <= 0.0:
                continue
            ids = " ".join(sorted(c.nodes))
            if "fuel." in ids:
                continue
            if "pneumatic" in ids and vs is not None:
                continue          # already shown, vessel by vessel
            label = ("air" if "pneumatic" in ids else "N2O" if "nitrous" in ids
                     else "WMI" if "auxiliary_injection" in ids else "tank")
            gauges.append((label, c.fill_level_frac, f"{c.fill_level_frac * 100:3.0f}%"))
    p = getattr(sim, "plant", None)
    if p is not None:
        h = getattr(p, "hydraulics", None)
        if h is not None:
            # named for the fluid, not the subsystem: "hyd" next to a
            # dozen other three-letter labels reads as one more switch
            gauges.append(("hyd-oil", h.oil_l / max(h.tank_capacity_l, 1e-6), f"{h.oil_l:4.1f}L"))
            if h.nitrogen_fitted:
                gauges.append(("hyd-N2", h.nitrogen_fill_frac, f"{h.nitrogen_fill_frac * 100:3.0f}%"))
            gauges.append(("hyd-life", h.oil_life_frac, f"{h.oil_life_frac * 100:3.0f}%"))
        gauges.append(("refrig", p.loop.charge_frac, f"{p.loop.charge_frac * 100:3.0f}%"))
        if p.bank is not None:
            gauges.append(("acc-bat", p.bank.soc_frac, f"{p.bank.soc_frac * 100:3.0f}%"))
        d = getattr(p, "dryer", None)
        if d is not None and d.fitted:
            bed = d.online_bed
            gauges.append(("desicc", 1.0 - bed.loading_frac, f"{(1 - bed.loading_frac) * 100:3.0f}%"))
    gauges.append(("batt", st.battery_soc_frac if hasattr(st, "battery_soc_frac") else 1.0,
                   f"{(getattr(st, 'battery_soc_frac', 1.0)) * 100:3.0f}%"))

    out: list[str] = []
    # Anything bolted on that runs off this engine's fluids goes ABOVE
    # the tanks. It is the thing being operated, so it is the thing that
    # must survive a trim; the gauges under it are monitoring, and the
    # reference readings further down are neither.
    eq = getattr(sim, "equipment", None)
    if eq is not None:
        out.extend(eq.summary())
    out.append("  -- TANKS --")
    per_line = 3
    for i in range(0, len(gauges), per_line):
        chunk = gauges[i:i + per_line]
        out.append("  " + "  ".join(f"{name:>8s}[{_bar(frac)}]{txt}" for name, frac, txt in chunk))

    # every switch that governs them, on one or two lines
    if p is not None:
        c = p.controls
        switches = [("chiller", c.main_chiller), ("dryer", c.air_dryer_enable), ("reheat", c.reheater_enable),
                    ("ac-fan", c.aftercooler_fan), ("sep-drn", c.separator_auto_drain),
                    ("tank-drn", c.wet_tank_auto_drain), ("hyd-chill", c.hydraulic_chiller_enable),
                    ("res-iso", c.reserve_isolation)]
        out.append("  -- CONTROLS --")
        for i in range(0, len(switches), 4):
            out.append("  " + "  ".join(f"{n}:{'ON ' if v else 'off'}" for n, v in switches[i:i + 4]))
        h = getattr(p, "hydraulics", None)
        if h is not None:
            out.append(f"  blanket:{h.blanket_active_source:12s} pilot:{'LIVE' if h.pilot_available else 'DEAD'}  "
                       f"fan-drv:{h.fan_drive_rpm:4.0f}rpm  oil:{h.temp_k - 273.15:4.0f}C")
        out.append(f"  refrig: {'CLUTCH IN ' if p.loop.clutch_engaged else 'clutch out'} "
                   f"{p.loop.cooling_w / 1000:5.2f}kW" + (f"  [{p.loop.lockout}]" if p.loop.lockout else ""))
    if at is not None:
        out.extend(at.describe())
    return out


# How many lines of damage reporting the left column can actually show.
# The column is about fifty lines tall and the fixed readouts already
# use most of it, so an engine that is genuinely coming apart -- a dozen
# holes, four fires, a cascade log -- can push the gauges off the bottom
# of the screen. Losing the tank gauges at exactly the moment everything
# is emptying is the worst possible time to lose them, so the damage
# section is capped and says how much it is not showing.
LEAK_SECTION_MAX_LINES = 14


def _cap(lines: list[str], limit: int, what: str) -> list[str]:
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"  ... and {len(lines) - limit} more {what} (worst shown first)"]


def leak_lines(sim: EngineCycleSim) -> list[str]:
    """Every hole emitter with something passing, and each circuit's loss."""
    field = getattr(sim, "hole_emitters", None)
    if field is None or not field.emitters:
        return []
    lines = []
    live = [e for e in field.emitters if e.regime != "none" and e.kind != "splash"]
    splash = [e for e in field.emitters if e.kind == "splash" and e.regime != "none"]
    if splash:
        lines.append(f"  SPLASH: {len(splash)} dippers, {sum(e.mass_flow_kg_s for e in splash) * 1000:.0f} g/s oil at "
                     f"{splash[0].jet_speed_m_s:.1f} m/s, {splash[0].character}")
    live.sort(key=lambda e: -e.mass_flow_kg_s)      # worst first, so the cap keeps those
    for em in live[:6]:
        rate = (f"{em.mass_flow_kg_s * 1000:.1f} g/s" if em.fluid != "gas" else f"{em.mass_flow_kg_s * 1000:.2f} g/s gas")
        extra = (f" {em.drip_rate_hz:.1f} drops/s" if em.regime == "drip"
                 else " drawing in" if em.regime == "ingest" else f" jet {em.jet_speed_m_s:.1f} m/s")
        lines.append(f"  LEAK {em.part.split('.')[-1]}: {em.fluid or '-'} {em.regime} {rate}{extra} [{em.character}]")
    for cid, lost in field.lost_l.items():
        lines.append(f"  LOST {cid}: {lost:.2f} L")
    lines.extend(getattr(sim, "bursts", None).summary() if getattr(sim, "bursts", None) else [])
    if getattr(sim, "_node_conditions", None):
        import node_effects
        lines.extend(node_effects.condition_lines(sim._node_conditions, sim._effects, limit=6))
    if getattr(sim.state, "oil_pickup_air_frac", 0.0) > 0.01:
        lines.append(f"  OIL PICKUP UNCOVERED: drawing {sim.state.oil_pickup_air_frac * 100:.0f}% air, "
                     f"{sim.state.oil_pressure_pa / 1000:.0f} kPa at the gallery")
    c = getattr(sim, "coupling", None)
    if c is not None:
        lines.append("  COUPLING " + c.describe()
                     + (f"  (applied from {sim.coupling_spec.apply_source})"
                        if c.is_hydraulically_applied else ""))
    # the plant's long narration is available on demand (the dashboard
    # shows its capacities as gauges in tanks_and_controls_lines); only
    # the exceptions are worth a line here
    p_ = getattr(sim, "plant", None)
    if p_ is not None:
        for l_ in p_.summary():
            if "!" in l_ or "OFF:" in l_ or "FLAT" in l_:
                lines.append(l_)
    lines.extend(getattr(sim, "fires", None).summary() if getattr(sim, "fires", None) else [])
    lines.extend(getattr(sim, "fittings", None).summary()[:5] if getattr(sim, "fittings", None) else [])
    lines.extend(getattr(sim, "ordnance", None).summary()[:4] if getattr(sim, "ordnance", None) else [])
    for line in getattr(sim, "cascade_log", [])[-4:]:
        lines.append("  CASCADE " + line)
    for cid, fr in field.fouling().items():
        lines.append(f"  FOUL {cid}: " + ", ".join(f"{k} {v * 100:.2f}%" for k, v in list(fr.items())[:4]))
    idle_holes = sum(1 for e in field.emitters if e.kind != "splash" and e.regime == "none")
    if idle_holes:
        lines.append(f"  holes with nothing passing: {idle_holes}")
    return _cap(lines, LEAK_SECTION_MAX_LINES, "damage lines")

def tank_lines(sim: EngineCycleSim) -> list[str]:
    """Every real depletable reservoir this engine's own build actually
    carries, walked generically off the drivetrain graph's own fluid
    circuits (fuel already gets its own detailed line elsewhere in the
    dashboard; this covers whatever ELSE is fitted -- nitrous,
    auxiliary/WMI injection, the pneumatic starting-air or reserve-air
    receiver -- so a new accessory shows up here for free, no HUD code
    to update when one is added)."""
    drivetrain = getattr(sim, "_drivetrain", None)
    if drivetrain is None:
        return []
    out = []
    bar_w = 24
    for c in drivetrain.fluid_circuits:
        if c.bottle_capacity_kg <= 0.0:
            continue
        node_ids = sorted(c.nodes)
        if any(nid.startswith("fuel.") for nid in node_ids):
            continue   # fuel already gets its own detailed TANK/PUMP lines above
        if any("nitrous" in nid and "auxiliary" not in nid for nid in node_ids):
            label = "NITROUS"
        elif any("auxiliary_injection" in nid for nid in node_ids):
            label = "WMI TANK"
        elif any("pneumatic" in nid for nid in node_ids):
            label = "AIR RECEIVER" if c.working_pressure_pa > 2_000_000.0 else "AIR RESERVE"
        else:
            label = node_ids[0].rsplit(".", 1)[-1].upper()
        fill = clamp(c.fill_level_frac, 0.0, 1.0)
        bar = "#" * int(fill * bar_w) + "." * (bar_w - int(fill * bar_w))
        pressure_note = f"  {c.pressure_pa / 1000:6.0f} kPa" if c.pressure_pa > 150_000.0 else ""
        out.append(f"  {label:<13s}[{bar}] {fill * 100:5.1f}%  {fill * c.bottle_capacity_kg:5.2f} "
                   f"/ {c.bottle_capacity_kg:5.2f} kg{pressure_note}")
    return out


def dashboard_lines(sim: EngineCycleSim, roster: list, listen: Listen,
                     help_text: str | None, last_export: str, panel=None,
                     max_lines: int | None = None) -> list[str]:
    """Pure text content -- no cursor/rendering assumptions -- so both a
    terminal (ANSI cursor-up redraw) and a pygame window (blit each line)
    can present the exact same dashboard."""
    eng = sim.engine
    st = sim.state
    idx = roster.index(eng) + 1 if eng in roster else 0
    load_frac = min(1.0, sim.brake_load_nm / max(eng.peak_torque_nm, 1.0))
    # the real clutch/gearbox/dyno-absorber rig (self._brake_junction /
    # self._load_omega / self._current_gear_ratio() in engine_cycle_
    # sim.py) now couples the turbine and Otto-Langen engines to the
    # SAME test cell every piston engine sits on, not just crank
    # engines -- the dashboard's own gate for that section needs to
    # match, or none of that real state ever shows for these two kinds
    has_dyno_rig = bool(eng.architecture.cylinders) or eng.kind in ("turbine", "atmospheric", "expander")
    lines = []
    lines.append(f"[{idx}/{len(roster)}] {eng.label}  ({eng.kind}, {eng.architecture.layout}, "
                 f"{eng.architecture.cylinders or 'n/a'} cyl)")
    lines.append(f"  displacement {eng.displacement_l:.2f} L   peak torque {eng.peak_torque_nm:6.0f} Nm"
                 f"   inertia {eng.inertia_kg_m2:.3f} kg*m^2   mass {eng.mass_kg:.0f} kg")
    lines.append(f"  idle {eng.idle_rpm:.0f}  torque-peak {eng.torque_peak_rpm:.0f}  "
                 f"power-peak {eng.power_peak_rpm:.0f}  redline {eng.redline_rpm:.0f} rpm")
    # redline isn't a copied historical spec number to take on faith --
    # derived_redline_rpm() is what this engine's OWN geometry (piston
    # speed off its own stroke, valve float off its own spring) really
    # allows, computed independently of the declared number above. When
    # the declared redline sits past that real ceiling the engine is
    # genuinely being run into its own margin on purpose (a race build
    # on forged internals and a short service life), not a typo -- so
    # this is shown as a fact, not corrected.
    derived_limit = eng.derived_redline_rpm()
    if derived_limit is not None:
        # only worth a line when the redline is actually past what the
        # hardware can stand: "within its own real limit" is the normal
        # case and spends a line saying nothing is wrong
        if eng.redline_rpm > derived_limit:
            lines.append(f"  ! redline {eng.redline_rpm:.0f} is PAST its own piston-speed/valve-float "
                         f"ceiling of {derived_limit:.0f} rpm")
    if st.wear:
        worst = wear_module.worst_component(st.wear)
        if worst is not None:
            name, damage = worst
            lines.append(f"  wear: worst part {name} at {damage * 100:5.2f}% "
                         f"(of {len(st.wear)} tracked real components)")
    lines.append(f"  accessories: {accessory_summary(eng)}")
    if eng.architecture.cylinders:
        fuel = current_fuel(sim)
        octane = eng.fuel_compatibility.get(fuel, 1.0)
        lines.append(f"  fuel: {fuel}  (compatibility {octane * 100:4.0f}%, F to cycle, shift-F to convert)")
    net = getattr(sim.engine, "fuel_network", None)
    if net is not None:
        st = sim.state
        eng_ = sim.engine
        mode = ("expander" if eng_.kind == "expander"
                else "compression ignition" if eng_.compression_ignition else "spark")
        stages = " > ".join([net.source.kind] + [s_.kind for s_ in net.stages] + [net.admission.kind])
        lines.append(f"  conversion: {net.fluid} [{mode}]  CR {eng_.architecture.compression_ratio:.1f}"
                     f"  timing {eng_.ignition_timing_offset_deg:+.0f} deg")
        lines.append(f"    network: {stages}")
        warn = f"  ! {st.fuel_network_warnings[0]}" if st.fuel_network_warnings else ""
        lines.append(f"    supply {st.fuel_supply_pressure_pa / 1e5:5.2f} bar  availability {st.fuel_availability_frac * 100:3.0f}%"
                     f"  fill {st.fuel_fill_frac * 100:3.0f}%  fuel {st.fuel_temp_k - 273.15:4.0f} C{warn}")
    # per-cylinder valve factors matter once they diverge or carbon
    # builds; at build tolerance they are a line of noise
    if st.cylinder_valve_factor and (max(st.cylinder_carbon_frac[:8], default=0.0) > 0.01
                                     or max(abs(v - 1.0) for v in st.cylinder_valve_factor[:8]) > 0.05):
        lines.append("  valves: factor " + " ".join(f"{v:.3f}" for v in st.cylinder_valve_factor[:8])
                     + "  carbon% " + " ".join(f"{v * 100:.0f}" for v in st.cylinder_carbon_frac[:8]))
    if st.cylinder_oil_film_mg:
        lines.append(f"  crankcase: oil {st.sump_oil_l:.2f} L  burn {st.oil_consumption_ml_per_h:.1f} mL/h  "
                     f"blow-by {st.blowby_l_per_min:.2f} L/min  case {st.crankcase_pressure_kpa:.2f} kPa  dilution {st.oil_fuel_dilution_frac * 100:.1f}%")
    lines.append("-" * 64)
    status = "STALLED" if st.stalled else ("running" if sim.rpm > 1 else "stopped")
    bar_w = 40
    rpm_frac = clamp(sim.rpm / eng.redline_rpm, 0.0, 1.0)
    bar = "#" * int(rpm_frac * bar_w) + "." * (bar_w - int(rpm_frac * bar_w))
    # for a free-piston engine this genuinely ISN'T "engine speed" the
    # way it is for a crank engine -- it's the flywheel/output shaft's
    # own speed, only loosely coupled to the piston's real activity
    # through the one-way ratchet (see otto_langen.py's own module
    # docstring). Labeled accordingly so the dashboard doesn't imply a
    # relationship this mechanism doesn't have.
    rpm_label = "FLYWHEEL" if eng.kind == "atmospheric" else "RPM     "
    lines.append(f"  {rpm_label}[{bar}] {sim.rpm:7.0f}  ({status})")
    thr_bar = "#" * int(sim.throttle * bar_w) + "." * (bar_w - int(sim.throttle * bar_w))
    plate_note = f"  butterfly {st.throttle_plate_angle_deg:4.1f} deg" if eng.architecture.cylinders else ""
    lines.append(f"  THROTTLE[{thr_bar}] {sim.throttle * 100:5.1f}%{plate_note}")
    # the mode hint belongs ON the bar it governs, not under it: the
    # left column is only about fifty lines tall and every line spent
    # restating which key does what is a capacity gauge pushed off the
    # bottom of the screen.
    if sim.throttle_target_rpm is not None:
        lines[-1] += f"  AUTO {sim.throttle_target_rpm:.0f}rpm (R/Y/T)"
    else:
        lines[-1] += "  manual (W/S, T holds)"
    if has_dyno_rig:
        n_forward = len(eng.transmission.gear_ratios)
        gear_label = ("R" if sim.gear_index < 0 else "N" if sim.gear_index == 0 else str(sim.gear_index))
        clutch_bar = "#" * int(sim.clutch_frac * bar_w) + "." * (bar_w - int(sim.clutch_frac * bar_w))
        shift_note = f"  ({sim._quick_shift_state})" if sim._quick_shift_state else ""
        lines.append(f"  GEAR    {gear_label:>2s} / {n_forward}   CLUTCH [{clutch_bar}] {sim.clutch_frac * 100:5.1f}%{shift_note}"
                     f"   (. shift up, , shift down)")
    brk_bar = "#" * int(load_frac * bar_w) + "." * (bar_w - int(load_frac * bar_w))
    lines.append(f"  BRAKE   [{brk_bar}] {sim.brake_load_nm:6.0f} Nm  ({load_frac * 100:4.1f}% of peak)")
    if sim.brake_target_rpm is not None:
        lines[-1] += f"  TGT {sim.brake_target_rpm:.0f}rpm (E/D/U)"
    else:
        lines[-1] += "  manual (E/D, U holds)"
    lines.append(f"  torque RMS {st.torque_rms_nm:6.0f} Nm  (inst {sim.current_torque_nm:6.0f})"
                 f"   power RMS {st.power_rms_kw:6.1f} kW  (inst {sim.power_kw:6.1f})")
    if has_dyno_rig:
        # the real virtual dyno-absorber drum's OWN state -- a separate
        # rotating body from the crank, coupled through the clutch/gear
        # chain above, not just a readout of engine rpm/power. Absorbed
        # power is the real standard dyno figure (brake_load_nm * the
        # drum's own speed, not the crank's); kinetic energy is what's
        # actually stored in the spinning drum right now -- relevant to
        # how much the drum itself resists a quick rpm change, same real
        # reason a heavier flywheel/drum feels "harder to spin up."
        lines.append(f"  DYNO    {st.dyno_rpm:7.0f} rpm   absorbed {st.dyno_absorbed_kw:6.1f} kW"
                     f"   drum KE {st.dyno_kinetic_energy_j:7.1f} J   torque {st.dyno_torque_nm:6.0f} Nm")
        if st.dyno_pull_state is not None:
            lines.append(f"  DYNO PULL: {st.dyno_pull_state}...  peak so far "
                         f"{st.dyno_pull_peak_torque_nm:.0f} Nm @ {st.dyno_pull_peak_torque_rpm:.0f} rpm, "
                         f"{st.dyno_pull_peak_power_kw:.1f} kW @ {st.dyno_pull_peak_power_rpm:.0f} rpm")
        else:
            lines[-1] += "   (P: WOT pull)"
    if eng.kind == "atmospheric":
        # the piston's own REAL activity -- what's actually happening
        # this instant, as distinct from the flywheel's own speed above
        # (see the FLYWHEEL label's own comment: they're only loosely
        # coupled through the one-way ratchet, not the same quantity)
        cyl = sim._atmospheric_cyl
        gov = sim._atmo_governor
        if cyl is not None:
            lines.append(f"  PISTON  phase {cyl.phase:16s}  x {cyl.x_m:5.3f} m   v {cyl.v_m_s:+6.2f} m/s"
                         f"   P {cyl.working_pressure_pa / 1000.0:6.1f} kPa   cycles {cyl.cycle_count}")
        lines.append(f"  fire rate {st.real_fire_hz:5.2f} Hz  ({st.real_fire_hz * 60.0:5.1f} /min)"
                     f"{'  MISS (governed)' if st.misfire_flag else '  firing'}")
        if gov is not None:
            gov_bar_w = 24
            gov_frac = clamp((gov.r_m - gov.r_min_m) / max(gov.r_max_m - gov.r_min_m, 1e-6), 0.0, 1.0)
            gov_bar = "#" * int(gov_frac * gov_bar_w) + "." * (gov_bar_w - int(gov_frac * gov_bar_w))
            lines.append(f"  GOVERNOR[{gov_bar}] ball {gov.r_m * 1000:4.1f} mm"
                         f"  (trip {gov.trip_radius_m * 1000:4.1f} / release {gov.release_radius_m * 1000:4.1f} mm)"
                         f"  {'TRIPPED' if gov.tripped else 'clear'}"
                         f"  spring preload {gov.spring_preload_n:4.2f} N  (-/= to adjust)")
    if eng.kind == "turbine":
        # the real cockpit gauges this engine actually has -- N1/EGT are
        # the same real quantities gas_turbine.py's own SingleShaftGas
        # Turbine already computes every tick, not new state
        turb = sim._turbine
        n1_bar_w = 24
        n1_fill = int(clamp(st.turbo_spool_frac, 0.0, 1.1) * n1_bar_w)
        n1_bar = "#" * min(n1_bar_w, n1_fill) + "." * max(0, n1_bar_w - n1_fill)
        lines.append(f"  N1      [{n1_bar}] {st.turbo_spool_frac * 100:5.1f}%"
                     f"   EGT {st.exhaust_temp_k - 273.15:5.0f} C   PR {1.0 + st.boost_frac:4.2f}")
        if turb is not None:
            lines.append(f"  fuel flow {turb.fuel_flow_kg_s * 1000:5.2f} g/s   tank {st.fuel_fill_frac * 100:5.1f}%"
                         f"   gas-generator {turb.rpm:6.0f} rpm  (RPM line above is the OUTPUT shaft, post-gearbox)")
    # regen braking is a real, specific mechanism _step_electric alone
    # implements (engines.py's electric/servo-electric kinds) -- the
    # turbine and Otto-Langen also have cylinders==0 but neither
    # implements it, so labeling their (currently inert) brake toggle
    # "regen" would be dishonest
    brake_label = "regen braking" if eng.kind in ("electric", "servo-electric") else "engine brake"
    brake_note = "ON" if sim.engine_brake_enabled else "OFF (coasting -- B to toggle)"
    lines.append(f"  {brake_label}: {brake_note}")
    # MANIFOLD/EXHAUST are GENERAL quantities -- every engine kind here
    # populates EngineCycleState.manifold_pressure_frac/exhaust_temp_k
    # with its own real meaning (throttle-plate MAP for a crank engine,
    # N1-referenced pressure ratio for a turbine, the always-atmospheric
    # admission + real isentropic gas temperature for the free-piston
    # engine -- see _step_atmospheric's own comment on that derivation),
    # not just a piston-only concept. Shown for any engine that has a
    # real dyno rig at all; the deeper combustion-only breakdown below
    # (backpressure/tailpipe/intake chain/port flow %) stays gated,
    # since those specific real mechanisms genuinely don't exist on the
    # other two kinds.
    if has_dyno_rig:
        map_bar_w = 24
        map_fill = int(clamp(st.manifold_pressure_frac, 0.0, 1.0) * map_bar_w)
        map_bar = "#" * map_fill + "." * (map_bar_w - map_fill)
        lines.append(f"  MANIFOLD[{map_bar}] {st.manifold_pressure_frac * 100:5.1f}%   "
                     f"ignition timing {st.ignition_timing_deg:5.1f} deg BTDC")
        exhaust_fill = int(clamp((st.exhaust_temp_k - 293.15) / 900.0, 0, 1) * map_bar_w)
        exhaust_bar = "#" * exhaust_fill + "." * (map_bar_w - exhaust_fill)
        lines.append(f"  EXHAUST [{exhaust_bar}] {st.exhaust_temp_k - 273.15:5.0f} C")
    # STARTER is also general -- starter.py's StartingSystems runs the
    # same real cranking physics for every engine kind (self.starter.
    # step() is called once per tick in _step_once before the kind
    # dispatch), not just combustion engines
    starter = sim.starter
    if st.cranking:
        lines.append(f"  STARTER  {starter.mode}: {st.starter_note}")
    elif st.starter_note:
        lines.append(f"  STARTER  {starter.mode}  ({st.starter_note})")
    # FUEL/TANK/PUMP is general to any engine with a real onboard tank
    # -- the drivetrain graph now builds one for kind in (combustion,
    # turbine) (see drivetrain_graph.py's own comment on why kind==
    # "atmospheric" deliberately has none: a real coal-gas utility main,
    # not a depletable tank)
    if eng.kind in ("combustion", "turbine"):
        fd = eng.fuel_delivery
        fuel_bar_w = 24
        fuel_fill = int(clamp(st.fuel_fill_frac, 0.0, 1.0) * fuel_bar_w)
        fuel_bar = "#" * fuel_fill + "." * (fuel_bar_w - fuel_fill)
        starve_note = "  STARVED" if sim._fuel_starvation_frac < 0.95 else ""
        cooler_note = f"  cooler target {fd.cooler_target_temp_k - 273.15:.0f} C" if fd.cooler_target_temp_k else ""
        lines.append(f"  TANK    [{fuel_bar}] {st.fuel_fill_frac * 100:5.1f}%   "
                     f"{st.fuel_fill_frac * fd.tank_capacity_l:6.1f} / {fd.tank_capacity_l:.0f} L{starve_note}"
                     f"   fuel {st.fuel_temp_k - 273.15:4.0f} C{cooler_note}")
        pump_frac = clamp(sim._fuel_demand_kg_s / max(fd.pump_flow_capacity_kg_s, 1e-9), 0.0, 1.0)
        pump_fill = int(pump_frac * fuel_bar_w)
        pump_bar = "#" * pump_fill + "." * (fuel_bar_w - pump_fill)
        lines.append(f"  PUMP    [{pump_bar}] {fd.pump_kind}, {sim._fuel_demand_kg_s * 1000:5.2f} g/s demand "
                     f"of {fd.pump_flow_capacity_kg_s * 1000:5.2f} g/s rated"
                     f"  ({sim._fuel_starvation_frac * 100:3.0f}% delivered)")
    if eng.architecture.cylinders:
        map_bar_w = 24
        # real exhaust-gas circuit state (drivetrain_graph.py's own solved
        # combustion heat-share + choked-flow backpressure), and the
        # real intake charge temp the boost/charge-air-cooling feedback
        # actually uses -- both previously computed but never surfaced
        lines.append(f"  EXHAUST backpressure {st.exhaust_pressure_frac * 100:+5.1f}%   "
                     f"charge {st.intake_charge_temp_k - 273.15:4.0f} C")
        # the real per-segment convective cooling chain's own endpoint --
        # what actually reaches the tailpipe opening after real heat loss
        # through the primary/collector/cat/muffler on the way there
        lines.append(f"  tailpipe exit {st.exhaust_tailpipe_temp_k - 273.15:5.0f} C"
                     f"  (manifold - tailpipe drop: {st.exhaust_temp_k - st.exhaust_tailpipe_temp_k:4.0f} C)")
        # the real staged intake temperature chain: ambient air, after
        # the runner's own conductive heat-soak off the live block temp
        # (engine_cycle_sim.py's RUNNER_BLOCK_CONDUCTION_FRAC), then
        # after plenum compression/soak and any real fuel evaporative
        # charge cooling -- three genuinely different monitored points,
        # not one lumped "intake temp"
        lines.append(f"  INTAKE TEMP  ambient {293.15 - 273.15:4.0f} C -> runner {st.intake_runner_temp_k - 273.15:4.0f} C "
                     f"-> charge {st.intake_charge_temp_k - 273.15:4.0f} C"
                     f"   air {sim._intake_demand_kg_s * 1000:5.2f} g/s")
        # the actual real-time cause behind MAP droop/backpressure: how
        # much the intake/exhaust ports are being asked to pass right
        # now against their own real flow_capacity_kg_s ceiling
        # (drivetrain_graph.py) -- >100% means genuinely choked THIS
        # tick, not just a pressure symptom after the fact
        flow_fill = int(clamp(st.intake_flow_demand_frac, 0, 1) * map_bar_w)
        flow_bar = "#" * flow_fill + "." * (map_bar_w - flow_fill)
        intake_choke_note = "  CHOKED" if st.intake_flow_demand_frac > 1.0 else ""
        exhaust_choke_note = "  CHOKED" if st.exhaust_flow_demand_frac > 1.0 else ""
        lines.append(f"  INTAKE  [{flow_bar}] flow {st.intake_flow_demand_frac * 100:5.1f}% of port capacity"
                     f"{intake_choke_note}   {sim._intake_demand_kg_s * 1000:5.2f} g/s")
        lines.append(f"  EXHAUST FLOW  {st.exhaust_flow_demand_frac * 100:5.1f}% of port capacity{exhaust_choke_note}"
                     f"   {sim._exhaust_demand_kg_s * 1000:5.2f} g/s")
        batt_fill = int(clamp(st.battery_soc_frac, 0, 1) * map_bar_w)
        batt_bar = "#" * batt_fill + "." * (map_bar_w - batt_fill)
        net = sim.electrical
        bus_note = "regulating" if st.bus_regulating else ("charging" if net.reading.battery_current_a > 0 else "DISCHARGING")
        lines.append(f"  BUS     [{batt_bar}] {st.battery_voltage:5.2f} V  SOC {st.battery_soc_frac * 100:3.0f}%  "
                     f"{net.battery.capacity_ah:4.0f} Ah   alternator {st.alternator_current_a:5.1f} A "
                     f"/ {net.alternator.rated_current_a:4.0f} A  {bus_note}")
        lines.append(f"  loads {st.electrical_load_w:6.0f} W   spark energy {st.spark_energy_frac * 100:4.0f}%"
                     f"   ECU {'powered' if sim.ecu.powered else 'BROWN-OUT'}"
                     f"   load resistor: {sim.electrical_load_frac * 100:5.0f}% of alternator rating  (Z/X to adjust)")
        # real coolant/oil circuit state -- air-cooled engines and
        # two-stroke total-loss lubrication genuinely have no such
        # circuit at all, shown honestly rather than a fake ambient
        # reading for a system that doesn't exist on this engine
        if eng.accessories.water_pump:
            coolant_fill = int(clamp((st.coolant_temp_k - 293.15) / 90.0, 0, 1) * map_bar_w)
            coolant_bar = "#" * coolant_fill + "." * (map_bar_w - coolant_fill)
            coolant_text = f"COOLANT [{coolant_bar}] {st.coolant_temp_k - 273.15:5.0f} C  {st.coolant_flow_lpm:5.1f} L/min"
        else:
            coolant_text = "COOLANT: n/a (air-cooled)"
        if not eng.architecture.two_stroke:
            oil_text = f"OIL {st.oil_pressure_pa / 1000:5.0f} kPa, {st.oil_temp_k - 273.15:4.0f} C, {st.oil_flow_lpm:4.1f} L/min"
        else:
            oil_text = "OIL: n/a (two-stroke total-loss)"
        lines.append(f"  {coolant_text}   {oil_text}")
        flags = []
        flags.append("KNOCK!" if st.knock_flag else "knock: clear")
        flags.append("MISFIRE" if st.misfire_flag else "misfire: clear")
        if st.rev_limiter_active:
            flags.append("LIMITER")
        flags.append("limiter:ON" if sim.rev_limiter_enabled else "limiter:OFF -- unchecked over-rev")
        flags.append("brake clutch " + ("locked" if st.brake_clutch_locked else "SLIPPING"))
        lines.append(f"  {'  '.join(flags)}  (N limiter)")
        spring = eng.lifter_spring
        spring_cap = spring.max_safe_rpm()
        spring_drag = spring.drag_torque_nm(
            eng.architecture.cylinders, valves_per_cylinder=cylinder_ports.valves_per_cylinder(eng))
        float_note = f"  VALVE FLOAT (risk {st.valve_float_risk * 100:.0f}%)" if st.valve_float_flag else ""
        # a spring nowhere near float is not news; near it, or floating,
        # it is the most important line on the screen
        if st.valve_float_flag or sim.rpm > spring_cap * 0.85:
            lines.append(f"  lifter spring: {spring.spring_rate_n_per_mm:.0f} N/mm, safe to ~{spring_cap:.0f} rpm"
                         f"  (drag {spring_drag:.1f} Nm){float_note}")
        fi = eng.forced_induction
        if fi.kind != "none":
            if fi.kind == "turbo":
                boost_fill = int(clamp(st.boost_frac / max(fi.max_boost_frac, 1e-6), 0, 1) * map_bar_w)
                boost_bar = "#" * boost_fill + "." * (map_bar_w - boost_fill)
                lines.append(f"  TURBO   [{boost_bar}] spool {st.turbo_spool_frac * 100:4.0f}%   "
                             f"boost {st.boost_frac * 100:4.0f}%"
                             + ("  WASTEGATE" if st.wastegate_flutter else "")
                             + ("  SURGE" if st.surge_flag else ""))
                if fi.anti_lag_capable:
                    lines.append(f"  anti-lag: {'ON' if sim.anti_lag_enabled else 'off'}  (A to toggle)")
            else:
                # blower PR-1 is what the belt is delivering (rotor speed
                # x displacement ratio); the GAUGE reads what a real boost
                # gauge reads -- manifold pressure above atmospheric, which
                # the upstream throttle (draw-through) gates
                gauge_boost = max(0.0, st.manifold_pressure_frac - 1.0)
                boost_fill = int(clamp(gauge_boost / max(fi.max_boost_frac, 1e-6), 0, 1) * map_bar_w)
                boost_bar = "#" * boost_fill + "." * (map_bar_w - boost_fill)
                lines.append(f"  BLOWER  [{boost_bar}] gauge {gauge_boost * 100:4.0f}%   "
                             f"rotor {st.turbo_spool_frac * 100:4.0f}% of belt   PR {1.0 + st.boost_frac:.2f}")
            if st.backfire_flag:
                lines.append(f"  BACKFIRE! ({st.backfire_kind})")
        # TANK/PUMP (general, above) already covers this engine's own
        # fuel supply -- tank_lines() covers every OTHER real reservoir
        # (nitrous, WMI, pneumatic, coolant, ...), genuinely piston-
        # accessory-specific
        lines.extend(tanks_and_controls_lines(sim))
        lines.extend(leak_lines(sim))
        carb = eng.carburetor
        if carb.is_carbureted:
            ref_mm = engines.reference_jet_diameter_mm(eng, current_fuel(sim))
            metering = engines.derive_jet_metering_frac(eng, current_fuel(sim))
            mix_note = "lean" if metering < 0.95 else ("rich" if metering > 1.05 else "correct")
            lines.append(f"  CARB    main jet {carb.main_jet_diameter_mm:.2f}mm "
                         f"(ref {ref_mm:.2f}mm for this fuel)  mixture: {mix_note} ({metering * 100:5.0f}%)")
        else:
            lines.append(f"  INJECTOR  electronic fuel rail  metering: correct (100%)")
    if panel is not None:
        panel_lines = panel.lines()
        if panel_lines:
            lines.extend(panel_lines)
    st = sim.state
    cat = sim.catalytic_converter
    if cat is not None:
        lines.append(f"  CAT     {cat.kind} {cat.substrate_volume_l:.1f} L  brick {st.catalyst_brick_temp_k - 273.15:5.0f} C  "
                     f"CO conv {st.catalyst_co_efficiency * 100:3.0f}%  NOx conv {st.catalyst_nox_efficiency * 100:3.0f}%  "
                     f"PGM {cat.pgm_g:.1f} g (Pt {cat.pt_g:.1f} Pd {cat.pd_g:.1f} Rh {cat.rh_g:.2f})  scrap ~${cat.scrap_value_usd:,.0f}")
    elif eng.exhaust_system.layout_has_catalyst:
        lines.append("  CAT     PULLED -- no conversion; backpressure down, everything engine-out goes straight out the pipe")
    lines.append(f"  EXHAUST phi {st.mixture_phi:4.2f}  CO {st.co_tailpipe_g_s * 3.6:6.2f} kg/h (engine-out {st.co_engine_out_g_s * 3.6:5.2f})  "
                 f"HC {st.hc_tailpipe_g_s * 3600:6.0f} g/h  NOx {st.nox_tailpipe_g_s * 3600:6.0f} g/h")
    room = sim.garage_mode
    extractor = " + extractor duct" if sim.exhaust_extractor_fitted and room != "outdoors" else ""
    lines.append(f"  AIR     bay {st.bay_air_temp_k - 273.15:4.0f} C  CO {st.bay_co_ppm:6.0f} ppm  O2 {st.bay_o2_frac * 100:4.1f}%   "
                 f"intake from: {st.intake_source} (O2 factor {st.intake_o2_factor:4.2f})   room: {room}{extractor}")
    if room != "outdoors":
        occ = sim.occupant
        ttl = occ.minutes_to_lethal(st.room_co_ppm) if occ else None
        ttl_txt = f"  time to lethal COHb at this ppm: {ttl:5.0f} min" if ttl is not None else ""
        lines.append(f"  ROOM    CO {st.room_co_ppm:6.0f} ppm  O2 {st.room_o2_frac * 100:4.1f}%  {st.room_temp_k - 273.15:3.0f} C   "
                     f"you: COHb {st.occupant_cohb_pct:4.1f}% -- {occ.condition if occ else ''}{ttl_txt}")
    lines.append(f"  listening: {listen.mode}  (L to cycle: stereo header/bay -> header solo -> bay solo)")
    if last_export:
        lines.append(f"  last export: {last_export}")
    if help_text:
        lines.append("-" * 64)
        lines.extend(help_text.strip().splitlines())
        # TRIM TO WHAT THE SCREEN ACTUALLY HAS. A caller that knows how
    # many lines it can draw says so, and the tail is what goes: the
    # readings at the bottom (catalyst, emissions, bay air, what the
    # microphone is listening to) are reference rather than things you
    # watch while something is going wrong, and the gauges above them
    # are not. Silently drawing past the bottom of the window -- which
    # is what used to happen -- hides whichever section happens to be
    # last, which on a damaged engine was the tank gauges.
    if max_lines is not None and len(lines) > max_lines:
        hidden = len(lines) - (max_lines - 1)
        lines = lines[:max_lines - 1] + [f"  ... {hidden} more lines below the window"]
    return lines

def bake_snapshot(sim: EngineCycleSim) -> str:
    rpm = max(sim.rpm, sim.engine.idle_rpm)
    throttle = sim.throttle
    load_frac = min(1.0, sim.brake_load_nm / max(sim.engine.peak_torque_nm, 1.0))

    out_dir = os.path.join(os.path.dirname(__file__), "bake_out")
    os.makedirs(out_dir, exist_ok=True)

    # the engine as a mesh: vertices, triangles, materials (OBJ + MTL)
    try:
        from drivetrain_graph import build_drivetrain_graph
        from engine_mesh import build_engine_mesh, export_obj_mtl
        static, moving = build_engine_mesh(build_drivetrain_graph(sim.engine), crank_angle_deg=sim.state.crank_angle_deg, covers_off=True)
        export_obj_mtl(static, moving, os.path.join(out_dir, f"{sim.engine.identity}_engine.obj"))
    except Exception as exc:  # the bake must not die on a mesh export
        print("mesh export skipped:", exc)
    acoustic, acoustic_loop_s = engine_baker.bake_acoustic(sim.engine, rpm, throttle, load_frac, cycles=4)
    for name, samples in acoustic.items():
        path = os.path.join(out_dir, f"{sim.engine.identity}_acoustic_{name}.wav")
        engine_baker.write_wav(path, samples, engine_baker.AUDIO_SAMPLE_RATE)

    vibration, vibe_loop_s = engine_baker.bake_vibration(sim.engine, rpm, throttle, load_frac, cycles=4)
    for name, samples in vibration.items():
        path = os.path.join(out_dir, f"{sim.engine.identity}_mount_{name}.wav")
        engine_baker.write_wav(path, samples, engine_baker.VIBE_SAMPLE_RATE)

    physics, physics_loop_s = engine_baker.bake_physics_rate(sim.engine, rpm, throttle, load_frac, cycles=1)
    npz_path = os.path.join(out_dir, f"{sim.engine.identity}_physics_rate.npz")
    np.savez(npz_path, sample_rate=engine_baker.PHYSICS_SAMPLE_RATE,
             loop_s=physics_loop_s, **physics)

    return (f"{len(acoustic)} acoustic pt(s) @ {acoustic_loop_s * 1000:.0f}ms, "
            f"{len(vibration)} mount(s) @ {vibe_loop_s * 1000:.0f}ms, "
            f"physics trace @ {engine_baker.PHYSICS_SAMPLE_RATE}Hz -> {out_dir}")


def _prompt(text: str, default: str) -> str:
    raw = input(f"{text} [{default}]: ").strip()
    return raw if raw else default


def run_engine_builder_form(custom_engines: list):
    """A sequence of text-field prompts (blank = default) that builds an
    Engine via engine_builder.build_custom_engine and hands it back --
    console input/output either way (the pygame frontend just pauses its
    loop and drops to the same console the process was launched from)."""
    print("\n" + "=" * 64)
    print("CUSTOM ENGINE BUILDER  (blank accepts the default, Ctrl+C cancels)")
    print("=" * 64)
    try:
        label = _prompt("Name", f"Custom build #{len(custom_engines) + 1}")
        kind = _prompt("Kind (combustion/electric)", "combustion")
        cylinders = int(_prompt("Cylinders (0 for electric/servo)", "8" if kind == "combustion" else "0"))
        if cylinders > 0:
            banks = int(_prompt("Banks (1=inline, 2=V/flat, 4=W)", "2" if cylinders > 4 else "1"))
            bank_angle = float(_prompt("Bank angle, degrees (0=inline, 180=flat, else V-angle)",
                                        "90" if banks == 2 else ("45" if banks == 4 else "0")))
        else:
            banks, bank_angle = 1, 0.0
        displacement = float(_prompt("Displacement (liters)", "5.0"))
        redline = float(_prompt("Redline (rpm)", "7000"))
        performance = _prompt(f"Performance level ({'/'.join(PERFORMANCE_PRESETS)})", "sport")
        if performance not in PERFORMANCE_PRESETS:
            performance = "sport"
        fi_kind = _prompt("Forced induction (none/turbo/supercharger)", "none")
        forced_induction = ForcedInduction(kind=fi_kind) if fi_kind in ("turbo", "supercharger") else ForcedInduction()

        spring_key = _prompt(f"Lifter spring ({'/'.join(LIFTER_SPRING_PRESETS)})", "stock")
        if spring_key not in LIFTER_SPRING_PRESETS:
            spring_key = "stock"
        lifter_spring = LIFTER_SPRING_PRESETS[spring_key]

        slug = "".join(c if c.isalnum() else "-" for c in label.lower())[:24].strip("-") or "engine"
        identity = f"custom-{len(custom_engines) + 1}-{slug}"
        engine = build_custom_engine(
            identity, label, cylinders=max(0, cylinders), banks=max(1, banks), bank_angle_degrees=bank_angle,
            displacement_l=max(0.05, displacement), redline_rpm=max(500.0, redline),
            kind=kind if kind in ("combustion", "electric") else "combustion",
            performance=performance, forced_induction=forced_induction, lifter_spring=lifter_spring,
        )
        print(f"Built: {engine.label} -- {engine.architecture.layout}")
        if engine.architecture.firing_order:
            print(f"Firing order: {engine.architecture.firing_order}  (crank-throw evaluation score {engine.architecture.firing_score:.3f}, lower is better)")
        print(f"Peak torque {engine.peak_torque_nm:.0f} Nm, redline {engine.redline_rpm:.0f} rpm")
        if cylinders > 0:
            cap = lifter_spring.max_safe_rpm()
            note = "OK" if cap >= engine.redline_rpm else f"WARNING: spring only safe to ~{cap:.0f} rpm, redline will float"
            print(f"'{spring_key}' spring safe to ~{cap:.0f} rpm -- {note}")
        print("=" * 64)
        return engine
    except (ValueError, KeyboardInterrupt) as exc:
        print(f"Cancelled ({exc.__class__.__name__}).")
        print("=" * 64)
        return None
