"""Bake a contamination and wear profile for every piston engine.

The sim now advances wear, debris, soot and filter loading from real
duty, which means it can be ASKED what an engine does over its life
rather than told. This runs each engine through its own duty and records
the answer once, so anything that needs to know "what does this engine's
oil look like at four hundred hours" reads a baked row instead of
running a simulation.

WHY BAKE IT AT ALL. Two reasons, and the second is the important one.
The first is cost: nobody wants to simulate fifteen thousand hours to
draw a service gauge. The second is CALIBRATION. One engine calibrated
against published figures tells you almost nothing about the rest --
it cannot separate "this class needs a different coefficient" from "the
model is missing a term". A profile across the whole catalogue, next to
whatever industrial specs each engine really has, is what turns one
anchor into a family: the engines with published service data check the
model, and the engines without get a defensible default plus an offset
from the class they belong to.

WHAT A ROW IS. Everything a used-oil report and a service schedule
actually contain, at a stated interval:

    oil analysis      ASTM D5185 wear metals, ppm by mass
    soot              percentage by mass, how a diesel report states it
    filter loading    grams held, against the element's own capacity
    wear              damage fraction of every tracked component
    service due       what has crossed a threshold and wants attention

THE SPEC COLUMN IS THE POINT. `industrial_spec` carries what is really
published for that engine -- overhaul life, drain interval, sump volume
-- and it is deliberately EMPTY for engines nobody has figures for.
Comparing the baked row against a filled spec is a calibration check;
the empty ones are what the calibration is eventually FOR.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import crankcase_state
import wear as wear_module
import wear_debris


@dataclass(frozen=True)
class IndustrialSpec:
    """What the manufacturer actually publishes, where anyone does.

    Empty fields are honest: most engines in this catalogue are types
    rather than part numbers, and inventing a spec for them would turn
    the calibration into a circle."""
    overhaul_hours: float = 0.0
    drain_interval_hours: float = 0.0
    sump_litres: float = 0.0
    rated_kw: float = 0.0
    bsfc_kg_per_kwh: float = 0.0
    source: str = ""


#: Only engines with figures that can actually be checked. Each of these
#: is a real published service interval for the type, and each is a
#: calibration point rather than a decoration.
INDUSTRIAL_SPECS: dict[str, IndustrialSpec] = {
    "cat-c18-industrial-diesel": IndustrialSpec(
        overhaul_hours=15_000.0, drain_interval_hours=500.0, sump_litres=57.0,
        rated_kw=450.0, bsfc_kg_per_kwh=0.21,
        source="heavy industrial diesel: published overhaul and S.O.S interval"),
    "ldt465-multifuel-deuce": IndustrialSpec(
        overhaul_hours=6_000.0, drain_interval_hours=150.0, sump_litres=22.0,
        rated_kw=104.0, bsfc_kg_per_kwh=0.26,
        source="military multifuel: sustained-load overhaul life"),
    "mazda-b6ze-miata-1990": IndustrialSpec(
        overhaul_hours=5_000.0, drain_interval_hours=120.0, sump_litres=3.8,
        rated_kw=85.0, bsfc_kg_per_kwh=0.30,
        source="light petrol: 7,500 km drain at an average road speed"),
}

#: What an oil is condemned at, regardless of engine. These are the
#: thresholds a lab actually flags on.
CONDEMN_SOOT_FRAC = 0.05          # 5% by mass: a diesel is finished well before this
CONDEMN_FILTER_BLOCKED = 0.80     # past this the bypass is open and nothing is filtered


@dataclass
class ProfileRow:
    identity: str
    hours: float
    ppm: dict = field(default_factory=dict)
    soot_frac: float = 0.0
    captured_g: float = 0.0
    wear: dict = field(default_factory=dict)
    sump_kg: float = 0.0
    fuel_burned_kg: float = 0.0
    spec: IndustrialSpec = field(default_factory=IndustrialSpec)

    @property
    def worst_component(self) -> tuple:
        real = {k: v for k, v in self.wear.items() if k != "engine_oil"}
        if not real:
            return ("", 0.0)
        return max(real.items(), key=lambda t: t[1])

    @property
    def oil_condemned(self) -> bool:
        return self.soot_frac >= CONDEMN_SOOT_FRAC


def bake(engine, hours: float, rpm: float | None = None,
         load_frac: float = 0.75, steps: int = 400) -> ProfileRow:
    """Run one engine's duty forward and record what it did to itself.

    Deliberately NOT a live sim run: this steps the wear and debris
    ledgers directly over coarse blocks, because the question is what
    an engine looks like after hundreds of hours and the per-tick
    combustion detail has already been integrated into the duty figures
    (`rpm`, `load_frac`) it is being asked about."""
    spec = INDUSTRIAL_SPECS.get(getattr(engine, "identity", ""), IndustrialSpec())
    rpm = float(rpm if rpm is not None else getattr(engine, "torque_peak_rpm", 2000.0))
    state = wear_module.new_wear_state(engine)
    before = wear_debris.damage_vector(state)
    led = wear_debris.DebrisLedger()
    cc = crankcase_state.CrankcaseState.from_engine(engine)
    sump_kg = (spec.sump_litres or crankcase_state.derive_oil_capacity_l(engine)) \
        * crankcase_state.OIL_DENSITY_KG_L

    # fuel actually burnt over the interval, from the engine's own rating
    # THE ENGINE ALREADY KNOWS HOW BIG IT IS. A kW-per-litre fallback
    # cannot: specific output runs from about 3 kW/L on a large slow-
    # speed marine diesel to over 100 on a race engine, and a single
    # figure rated the Wartsila at 633 MW against a real 80, which put
    # nearly a tonne of soot in its sump. `peak_power_kw` is derived
    # from its own declared torque and rated speed and lands within a
    # couple of percent of the real engine.
    rated_kw = (spec.rated_kw
                or float(getattr(engine, "rated_power_kw", 0.0) or 0.0)
                or max(1.0, float(getattr(engine, "peak_power_kw", 0.0) or 0.0)))
    bsfc = spec.bsfc_kg_per_kwh or 0.25
    fuel_kg = rated_kw * load_frac * bsfc * hours

    import combustion_kernel
    soot_family = combustion_kernel.family_for(engine)

    dt = hours * 3600.0 / max(1, steps)
    for _ in range(max(1, steps)):
        wear_module.step_wear(state, engine, dt, rpm, True, False)
        now = wear_debris.damage_vector(state)
        led.debit(before, now, float(getattr(engine, "displacement_l", 4.0) or 4.0))
        before = now
        cc.apply_ring_wear(float(state.get("piston_rings", 0.0) or 0.0))
        led.debit_soot(wear_debris.soot_into_oil_g(
            fuel_kg / max(1, steps), soot_family, float(cc.ring_leak.mean())))
        led.through_filter(0.95, 200.0)
        led.settle(dt * 2.0e-6)

    return ProfileRow(
        identity=getattr(engine, "identity", ""), hours=hours,
        ppm=led.ppm_by_metal(sump_kg), soot_frac=led.soot_frac_of_oil(sump_kg),
        captured_g=led.captured_g, wear=dict(state), sump_kg=sump_kg,
        fuel_burned_kg=fuel_kg, spec=spec)


def bake_catalogue(engines_iter, hours: float | None = None) -> list:
    """One row per piston engine, each at its OWN drain interval where a
    spec gives one -- comparing a 500 hour diesel with a 120 hour petrol
    at the same hour count compares nothing."""
    out = []
    for e in engines_iter:
        arch = getattr(e, "architecture", None)
        if arch is None or not getattr(arch, "cylinders", 0):
            continue
        if getattr(e, "kind", "") != "combustion":
            continue
        spec = INDUSTRIAL_SPECS.get(e.identity, IndustrialSpec())
        h = hours if hours is not None else (spec.drain_interval_hours or 250.0)
        out.append(bake(e, h))
    return out


def calibration_report(rows) -> list:
    """Which baked rows can be checked, and what they say.

    Only the engines with a real published spec are a check on anything;
    the rest are the reason the check matters."""
    lines = []
    for r in rows:
        if not r.spec.source:
            continue
        fe = r.ppm.get("Fe", 0.0)
        worst, frac = r.worst_component
        lines.append(
            f"{r.identity}: {r.hours:.0f} h  Fe {fe:5.1f} ppm  soot {r.soot_frac * 100:4.2f}%  "
            f"worst {worst} {frac * 100:5.2f}%  (overhaul at {r.spec.overhaul_hours:.0f} h)")
    return lines
