"""Builds a custom Engine from a handful of user-chosen parameters.

Every downstream system (engine_cycle_sim, engine_baker, engine_sound,
mech_parts, engine_geometry) already operates generically on the
Engine/EngineArchitecture/Accessories/ForcedInduction/LifterSpring
dataclasses -- none of it was written against the 13 hand-built
catalogue engines specifically. So a custom engine mostly comes down
to two real choices this module has to make on the user's behalf:
firing order (crank_phase.generate_firing_order's evaluate-and-evolve
search over throw phasing) and everything derivable from displacement/
performance level/redline (BMEP, inertia, mass -- so the user isn't
typing fifteen numbers for a first pass).
"""
from __future__ import annotations

from dataclasses import dataclass

from engines import (
    Engine, EngineArchitecture, Accessories, ForcedInduction, IntakeSystem, ExhaustSystem,
    LifterSpring, LIFTER_SPRING_PRESETS, RPM_TO_RAD_S, derive_gross_bmep_pa,
)
import crank_phase


def layout_label(cylinders: int, banks: int, bank_angle_degrees: float) -> str:
    if cylinders == 0:
        return "custom-electric"
    per_bank = -(-cylinders // banks)
    words = {1: "single", 2: "twin", 3: "triple", 4: "four", 5: "five", 6: "six",
             8: "eight", 10: "ten", 12: "twelve", 16: "sixteen"}
    cyl_word = words.get(cylinders, str(cylinders))
    if banks <= 1:
        return f"inline-{cyl_word}"
    if abs(bank_angle_degrees - 180.0) < 1e-6:
        return f"flat-{cyl_word}"
    return f"v{bank_angle_degrees:.0f}-{cyl_word}"


# "performance" is now only a convenience default for the REAL parts a
# builder can also set explicitly (compression ratio, valvespring, boost)
# -- not a lookup that hands back a power number directly. Two builds at
# "sport" with different intake/exhaust/fuel/compression choices now make
# genuinely different torque; two builds with identical parts but
# different performance labels make the same torque, because the label
# no longer touches bmep at all.
PERFORMANCE_PART_DEFAULTS: dict[str, dict] = {
    "mild": dict(compression_ratio=9.0, lifter_spring="soft", forced_induction=None),
    "sport": dict(compression_ratio=10.5, lifter_spring="stock", forced_induction=None),
    "race": dict(compression_ratio=12.0, lifter_spring="race", forced_induction=None),
}

# Real, disclosed typical ratio of braking/pumping MEP to drive BMEP for a
# throttled Otto-cycle engine (matches roughly the ~0.13-0.16 ratio already
# implicit in the hand-built catalogue's own bmep/braking_bmep pairs).
BRAKING_BMEP_FRAC = 0.14


def build_custom_engine(
    identity: str, label: str, *,
    cylinders: int, banks: int, bank_angle_degrees: float,
    displacement_l: float, redline_rpm: float,
    kind: str = "combustion",
    performance: str = "sport",
    compression_ratio: float | None = None,
    intake_system: IntakeSystem | None = None,
    exhaust_system: ExhaustSystem | None = None,
    combustion_efficiency: float = 0.90,
    compression_ignition: bool = False,
    wobble_amt: float | None = None,
    accessories: Accessories | None = None,
    forced_induction: ForcedInduction | None = None,
    lifter_spring: LifterSpring | None = None,
    load_resistor_frac: float | None = None,
    preferred_fuel_profile: str = "pump-gasoline-93",
    fuel_compatibility: dict | None = None,
    firing_order_generations: int = 400,
    firing_order_seed: int = 0,
) -> Engine:
    """The minimal real choices (cylinders/banks/bank-angle/displacement/
    redline) plus every part that actually determines how much torque the
    engine makes -- compression ratio, intake, exhaust, fuel, forced
    induction, valvespring -- each pulling its real weight through
    engines.derive_gross_bmep_pa and the valvespring's own natural-
    frequency ceiling, instead of a "performance" tier handing back a
    prescribed BMEP number. `performance` still exists as a one-word
    convenience: it only fills in compression_ratio/lifter_spring
    defaults when you don't set them yourself -- it no longer touches
    the resulting power at all."""
    part_defaults = PERFORMANCE_PART_DEFAULTS.get(performance, PERFORMANCE_PART_DEFAULTS["sport"])
    cr = part_defaults["compression_ratio"] if compression_ratio is None else compression_ratio
    spring = lifter_spring or LIFTER_SPRING_PRESETS[part_defaults["lifter_spring"]]
    intake = intake_system or IntakeSystem()
    exhaust = exhaust_system or ExhaustSystem()
    induction = forced_induction or ForcedInduction()

    # rev-range shape, each control point traced to a real mechanism
    # instead of a hand-picked fraction:
    #  - power_peak_rpm can't exceed what the chosen valvespring can
    #    physically follow (LifterSpring.max_safe_rpm(), the same real
    #    spring-vs-valvetrain-mass natural frequency the catalogue
    #    engines are already held to) -- a soft spring on a high-redline
    #    build genuinely peaks power lower than its redline, a race
    #    spring can use the redline you asked for.
    #  - torque_peak_rpm sits earlier in the band, and higher compression
    #    (faster, more complete early-flame burn) shifts it up a little,
    #    same real direction real high-CR engines tune toward.
    #  - idle_frac: fewer cylinders need a higher relative idle speed for
    #    flywheel smoothing (the same reason the catalogue's own single-
    #    cylinder curved-dash idles much higher, relative to its redline,
    #    than a straight-six).
    valvetrain_ceiling_rpm = min(redline_rpm, spring.max_safe_rpm())
    power_peak_rpm = valvetrain_ceiling_rpm * 0.93
    torque_peak_frac = max(0.45, min(0.85, 0.62 + 0.015 * (cr - 8.0)))
    torque_peak_rpm = power_peak_rpm * torque_peak_frac
    idle_frac = max(0.07, min(0.22, 0.22 - 0.015 * cylinders))
    idle_rpm = max(400.0, redline_rpm * idle_frac)

    firing_order, firing_score = crank_phase.generate_firing_order(
        cylinders, banks, generations=firing_order_generations, seed=firing_order_seed)
    layout = layout_label(cylinders, banks, bank_angle_degrees)
    arch = EngineArchitecture(
        layout=layout, cylinders=cylinders, banks=banks, bank_angle_degrees=bank_angle_degrees,
        firing_order=firing_order, firing_score=firing_score,
        wobble_amt=(0.35 if banks >= 2 and 60.0 <= bank_angle_degrees <= 100.0 else
                    (0.12 if banks >= 2 else 0.0)) if wobble_amt is None else wobble_amt,
        compression_ratio=cr,
    )

    gross_bmep_pa = derive_gross_bmep_pa(
        compression_ratio=cr, fuel_profile=preferred_fuel_profile,
        intake_system=intake, exhaust_system=exhaust, forced_induction=induction,
        combustion_efficiency=combustion_efficiency, compression_ignition=compression_ignition,
    )
    braking_bmep_pa = gross_bmep_pa * BRAKING_BMEP_FRAC

    # scale-derived defaults so displacement alone gives sane inertia/mass
    inertia_kg_m2 = max(0.02, displacement_l * 0.055 * (cylinders / 6.0 + 0.5))
    mass_kg = max(30.0, displacement_l * 42.0 + cylinders * 6.0)
    clutch_torque_nm = displacement_l * 1000.0 * gross_bmep_pa / (4 * 3.141592653589793) * 1.4

    return Engine(
        identity=identity, label=label, kind=kind,
        displacement_l=displacement_l, bmep_pa=gross_bmep_pa, braking_bmep_pa=braking_bmep_pa,
        idle_rpm=idle_rpm, torque_peak_rpm=torque_peak_rpm, power_peak_rpm=power_peak_rpm,
        redline_rpm=redline_rpm, inertia_kg_m2=inertia_kg_m2, mass_kg=mass_kg,
        clutch_torque_nm=clutch_torque_nm, combustion_efficiency=combustion_efficiency,
        coupling_efficiency=0.94,
        architecture=arch, accessories=accessories or Accessories(),
        preferred_fuel_profile=preferred_fuel_profile,
        fuel_compatibility=fuel_compatibility or {preferred_fuel_profile: 1.0, "nitromethane-race": 1.0},
        forced_induction=induction,
        lifter_spring=spring,
        intake_system=intake,
        exhaust_system=exhaust,
        compression_ignition=compression_ignition,
    )
