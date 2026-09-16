"""Real, per-component wear and durability accounting.

Three real, distinct wear MECHANISMS cover essentially all engine wear
in practice -- this module doesn't invent a fourth:

  - "time": degrades whether the engine runs or not (rubber/elastomer
    hardening, oil chemical breakdown, corrosion). Real basis: simple
    first-order chemical/environmental decay -- accumulates with
    wall-clock seconds regardless of rpm or load.

  - "sliding_distance" (rpm x time): abrasive wear between two moving
    surfaces in contact (ring-against-bore, valve-stem-against-guide,
    bearing-against-journal). Real basis: Archard's wear equation, the
    established real form -- wear volume is proportional to sliding
    distance x normal load, not to elapsed time or revolution COUNT
    alone. This module uses cumulative piston sliding distance
    (Engine.mean_piston_speed_m_s(rpm) * dt, a real quantity Engine
    already exposes) as the real distance term.

  - "per_event": damage that happens in one discrete pulse -- spark
    erosion happens once per discharge, fatigue crack growth happens
    once per load cycle. Real basis: Miner's-rule cumulative-damage
    accounting, the standard real method for cyclic fatigue design. A
    knock/detonation event isn't just "one more cycle": real peak
    cylinder pressure during knock runs roughly 1.5-2.5x a clean burn
    (shockwave superimposed on the normal pressure trace), and real
    high-cycle steel fatigue life falls off with stress raised to a
    real Basquin exponent (b ~= 6-10 for typical steels) -- so a single
    bad detonation event can do the fatigue damage of dozens to
    hundreds of clean cycles. KNOCK_STRESS_RATIO/FATIGUE_EXPONENT below
    are disclosed, order-of-magnitude choices in that real cited range,
    not a separate hand-tuned "knock damage" number.

Every component in COMPONENTS declares which real mechanism governs it
and a real, disclosed life constant in that mechanism's own units. Damage
accumulates 0.0 (new) -> 1.0 (worn out) per component, per running
engine instance -- this module only does the accounting; what a
component's damage fraction should actually DO to the engine (weaker
spark, more blow-by, a real failure) is deliberately left to the
caller, the same "declare the real quantity, wire feedback later"
sequencing machining.py's Deviation used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TIME = "time"
SLIDING_DISTANCE = "sliding_distance"
PER_EVENT = "per_event"

# Real disclosed knock-fatigue multiplier: stress ratio during a
# detonation event to the exponent of a real Basquin-law high-cycle
# steel S-N slope. 2.0 ** 6 = 64 -- a single knock event does roughly
# the same PER_EVENT fatigue components' cumulative damage as 64 clean
# combustion cycles. See module docstring for the real citation range.
KNOCK_STRESS_RATIO = 2.0
FATIGUE_EXPONENT = 6
KNOCK_FATIGUE_MULTIPLIER = KNOCK_STRESS_RATIO ** FATIGUE_EXPONENT

# Real service life is the calibration, and pacing is a separate knob.
# 1.0 is real time: a heavy diesel reaches its rings' overhaul life in
# about fifteen thousand running hours, which is what one really does.
# Raise it to watch an engine age inside a session -- it multiplies the
# accumulation, not the constants, so nothing calibrated against a real
# life (oil analysis, service intervals, corrosion) stops being true.
WEAR_TIME_ACCELERATION = 1.0


@dataclass(frozen=True)
class WearComponent:
    """One real wearing part. `life_constant` is in the mechanism's own
    real unit: seconds of running time for "time", meters of cumulative
    piston sliding distance for "sliding_distance", or a real count of
    normal (non-knock) events to reach damage=1.0 for "per_event".
    `knock_multiplier` only applies to "per_event" components, and only
    on ticks a real knock was flagged."""
    name: str
    mechanism: str
    life_constant: float
    knock_multiplier: float = 1.0
    applies_to_diesel: bool = True   # False = combustion-ignition only (spark-specific parts)
    note: str = ""


COMPONENTS: tuple[WearComponent, ...] = (
    WearComponent(
        "spark_plug", PER_EVENT, life_constant=5.0e8,
        knock_multiplier=25.0, applies_to_diesel=False,
        note="real electrode arc-erosion per discharge; ~300k firing "
             "events is the right order of magnitude for a stock "
             "plug's rated life; knock's shockwave-scrubbed arc erodes "
             "the electrode far faster than a clean discharge"),
    WearComponent(
        "fuel_injector", PER_EVENT, life_constant=2.0e9,
        knock_multiplier=3.0,
        note="real nozzle/needle erosion per injection event; "
             "compression-ignition's own high-pressure injector, not "
             "the spark plug -- knock still matters here (cylinder "
             "pressure spikes stress the injector tip too) but far "
             "less than it does to an electrode arc"),
    WearComponent(
        "apex_seals", SLIDING_DISTANCE, life_constant=8.0e7,
        knock_multiplier=1.0,
        note="Wankel-specific: the real, famously short-lived wear "
             "item on a rotary -- the seal at each rotor apex scrapes "
             "the epitrochoid housing directly (no bore/ring pair at "
             "all), and real-world rotary rebuild intervals are well "
             "known to run far shorter than a piston engine's ring "
             "life, hence the much lower life constant here"),
    WearComponent(
        "piston_rings", SLIDING_DISTANCE, life_constant=3.0e8,
        note="real Archard abrasive wear between ring face and bore; "
             "~6,000 km of cumulative piston travel (not vehicle "
             "distance) is the right order of magnitude before ring "
             "seal degrades enough to matter"),
    WearComponent(
        "oil_control_ring", SLIDING_DISTANCE, life_constant=2.5e8,
        note="a real, distinct ring in the same pack -- scrapes the "
             "cylinder wall on every stroke same as the compression "
             "rings above, but sees less combustion-pressure loading "
             "so it genuinely outlasts them"),
    WearComponent(
        "valve_guides", SLIDING_DISTANCE, life_constant=4.0e8,
        note="real stem-in-guide sliding wear, same Archard mechanism "
             "as the rings, off the same piston-sliding-distance proxy "
             "(valve actuation count tracks piston stroke count 1:1 "
             "in any 4-stroke poppet-valve engine)"),
    WearComponent(
        "valve_seats", PER_EVENT, life_constant=2.0e9,
        knock_multiplier=8.0,
        note="real seat recession per closing impact -- a genuinely "
             "different real failure mode from guide wear above "
             "(impact peening, not sliding abrasion), so it gets its "
             "own per-event accounting instead of sharing the guides'"),
    WearComponent(
        "connecting_rod_bearings", PER_EVENT, life_constant=3.0e9,
        knock_multiplier=KNOCK_FATIGUE_MULTIPLIER,
        note="real Miner's-rule cyclic fatigue off combustion-pressure "
             "loading transmitted through the rod -- the classic real "
             "reason a detonating engine spins a bearing"),
    WearComponent(
        "piston_crown", PER_EVENT, life_constant=4.0e9,
        knock_multiplier=KNOCK_FATIGUE_MULTIPLIER,
        note="real cyclic thermal/pressure fatigue of the crown itself "
             "-- longer real life than the rod bearings (the crown "
             "sees the same pressure pulse but no bending-stress "
             "concentration), same knock multiplier since it's the "
             "same real overpressure event"),
    WearComponent(
        "crankshaft", PER_EVENT, life_constant=1.0e10,
        knock_multiplier=KNOCK_FATIGUE_MULTIPLIER,
        note="real Miner's-rule fatigue of the crank itself -- the "
             "most over-built rotating part by real design convention, "
             "so its own disclosed life constant is the longest here"),
    WearComponent(
        "engine_oil", TIME, life_constant=180_000.0,
        note="real chemical/thermal breakdown of the oil's own "
             "additive package -- ~180,000 s (50 running hours) is the "
             "right order of magnitude for a real conventional-oil "
             "change interval; independent of rpm or knock, matching "
             "why oil is changed on a time/hour basis, not a cycle one"),
)


_RECIPROCATING_ONLY = frozenset({
    "piston_rings", "oil_control_ring", "valve_guides", "valve_seats",
    "piston_crown",
})
_ROTARY_ONLY = frozenset({"apex_seals"})


def _components_for(engine) -> tuple[WearComponent, ...]:
    """Real applicability filter -- an electric motor has none of these
    real wearing parts (no combustion, no sliding contact at all), a
    compression-ignition engine has no spark plug (a real injector
    wears in its place instead), and a rotary has no poppet valves or
    piston rings to wear in the first place -- its own real wear item
    (apex_seals) takes their place rather than being silently applied
    where it doesn't belong."""
    if not engine.architecture.cylinders or engine.kind not in ("combustion",):
        return ()
    arch = engine.architecture
    out = []
    for c in COMPONENTS:
        if c.name == "spark_plug" and engine.compression_ignition:
            continue
        if c.name == "fuel_injector" and not engine.compression_ignition:
            continue
        if c.name in _RECIPROCATING_ONLY and arch.rotary:
            continue
        if c.name in _ROTARY_ONLY and not arch.rotary:
            continue
        if c.name in ("valve_guides", "valve_seats") and not arch.has_poppet_valves:
            continue
        out.append(c)
    return tuple(out)


def new_wear_state(engine) -> dict[str, float]:
    """A fresh, real damage=0.0 ledger for every component this engine
    actually has -- called once per engine selection, the same point
    machining.py's spec swap happens at."""
    return {c.name: 0.0 for c in _components_for(engine)}


def firing_events(engine, rpm: float, dt: float) -> float:
    """How many real combustion events this engine had in `dt`.

    A fourteen-cylinder engine fires fourteen times per cycle and a
    single fires once, so fatigue accumulates fourteen times faster on
    the one -- which is exactly what a per-event mechanism is for. This
    used to be a BOOL per physics substep: every cylinder that fired in
    the same substep counted as one event, so the Wartsila aged like a
    lawnmower, and the rate depended on the substep size rather than on
    the engine. Counting real events instead makes per-event wear
    independent of how finely the sim happens to be stepping."""
    arch = engine.architecture
    cylinders = max(1, int(getattr(arch, "cylinders", 1) or 1))
    cycle_deg = float(getattr(arch, "cycle_degrees", 720.0) or 720.0)
    revs = max(0.0, float(rpm)) / 60.0 * max(0.0, float(dt))
    return revs * cylinders * (360.0 / max(1.0, cycle_deg))




# ---------------------------------------------------------------------
# WHAT THE FUEL DOES TO THE HARDWARE
# ---------------------------------------------------------------------
#
# Until now every life constant here described a part running on the
# fuel it was designed for. That is the ordinary case and it was the
# only case the model had. But a converted engine is not running on the
# fuel it was designed for, and for one component that difference is not
# a percentage -- it is two orders of magnitude.
#
# fuel_network.hardware_notes() already knew this and said so in a
# string: "hardened valve seats: a dry gaseous fuel gives no seat
# lubrication". Its own docstring admitted the notes were "advice for
# the build, not physics the sim enforces". This is the physics.
#
# RECESSION IS A LOAD PHENOMENON, which is the part that makes it feel
# real rather than punitive. The micro-welding that tears the seat needs
# seat temperature and seat pressure, so a converted engine pottering
# around town is genuinely fine and the same engine towing up a long
# grade eats its head in an afternoon. Scaling on load squared gives
# exactly that: a well-known conversion failure that owners consistently
# describe as arriving from nowhere.

#: Worst-case multiplier on seat recession: dry gaseous fuel, plain cast
#: iron seats, sustained full load. Takes a five-thousand-hour seat to
#: something under fifty, which is the documented experience of gas
#: conversions done to unprepared heads.
UNLUBRICATED_SEAT_MULTIPLIER = 150.0
#: What a top-lube doser buys back. It does not eliminate the problem --
#: it replaces the lead film with a potassium/phosphorus one and gets
#: most of the way there.
TOP_LUBE_PROTECTION = 0.92


def seat_recession_multiplier(engine, load_frac: float = 1.0,
                              top_lube: bool = False) -> float:
    """How much faster the seats go on THIS fuel in THIS head.

    1.0 means the seats are seeing what they were designed for: either
    the fuel lubricates them, or the head was built not to need it."""
    fuel_key = getattr(engine, "preferred_fuel_profile", None)
    if not fuel_key:
        return 1.0
    try:
        import working_fluids as wfl
        fluid = wfl.WORKING_FLUIDS.get(fuel_key)
    except Exception:
        return 1.0
    if fluid is None or not getattr(fluid, "hardened_valve_seats_required", False):
        return 1.0
    arch = getattr(engine, "architecture", None)
    if arch is not None and getattr(arch, "hardened_valve_seats", True):
        return 1.0
    if arch is not None and not getattr(arch, "has_poppet_valves", True):
        return 1.0
    load = max(0.0, min(1.0, float(load_frac)))
    mult = 1.0 + (UNLUBRICATED_SEAT_MULTIPLIER - 1.0) * load * load
    if top_lube:
        mult = 1.0 + (mult - 1.0) * (1.0 - TOP_LUBE_PROTECTION)
    return mult


def environment_multipliers(engine, load_frac: float = 1.0,
                            top_lube: bool = False) -> dict:
    """Per-component rate factors that come from how the engine is being
    RUN rather than from how hard it is working.

    A dict rather than a special case inside step_wear, so the next one
    of these -- a fuel that washes the bores, a coolant that pits the
    liners -- has somewhere obvious to go."""
    out = {}
    m = seat_recession_multiplier(engine, load_frac, top_lube)
    if m != 1.0:
        out["valve_seats"] = m
    return out


def step_wear(wear_state: dict[str, float], engine, dt: float, rpm: float,
              fired: bool, knock: bool, load_frac: float = 1.0,
              top_lube: bool = False) -> None:
    """Advance every applicable component's real damage fraction by one
    tick, in place. `fired`/`knock` are this tick's real combustion
    outcome (see EngineCycleSim._record_ignition/state.knock_flag) --
    "time" components ignore them and just accrue dt; "sliding_distance"
    components convert this tick's real piston travel (Engine.mean_
    piston_speed_m_s(rpm) * dt) into abrasive wear; "per_event"
    components advance by the REAL NUMBER of firing events in this tick
    (see `firing_events`), and jump harder on a knocking one.

    WEAR_TIME_ACCELERATION multiplies everything. It exists because the
    life constants below are real service lives -- fifteen thousand
    hours to an overhaul -- and nobody plays for fifteen thousand hours.
    Keeping the physics honest and the pacing as a separate, declared
    knob is the only way to have both; burying a x50 factor in the life
    constants themselves would make every one of them a lie and would
    corrupt anything calibrated against them, starting with the oil
    analysis in wear_debris.py."""
    if not wear_state:
        return
    accel = max(0.0, WEAR_TIME_ACCELERATION)
    if accel <= 0.0:
        return
    sliding_m = engine.mean_piston_speed_m_s(rpm) * dt * accel
    events = (firing_events(engine, rpm, dt) * accel) if fired else 0.0
    env = environment_multipliers(engine, load_frac, top_lube)
    for c in _components_for(engine):
        if c.name not in wear_state:
            continue
        damage = wear_state[c.name]
        if c.mechanism == TIME:
            damage += dt * accel / c.life_constant
        elif c.mechanism == SLIDING_DISTANCE:
            damage += sliding_m / c.life_constant
        elif c.mechanism == PER_EVENT:
            if events > 0.0:
                increment = events / c.life_constant
                if knock:
                    increment *= c.knock_multiplier
                # the fuel's own contribution, which for a converted
                # engine's seats is the dominant term by a long way
                increment *= env.get(c.name, 1.0)
                damage += increment
        wear_state[c.name] = min(1.0, damage)


def worst_component(wear_state: dict[str, float]) -> tuple[str, float] | None:
    """The single most-worn declared component right now, or None if
    this engine has no wearing parts at all (electric) or hasn't run
    yet."""
    if not wear_state:
        return None
    name, damage = max(wear_state.items(), key=lambda kv: kv[1])
    return name, damage
