"""What conditions do to an engine, as one chain instead of six flags.

The project already carries the hard parts and never joins them up.
crankcase_state tracks a real oil film in MILLIGRAMS on every bore,
deposited by splash, scraped by the rings, burned past them. valve_state
carries a real per-valve float speed from its own spring rate and the
deposit mass riding on it. centrifuges and hydraulics each already solve
oil viscosity against temperature. What none of them do is ask the
question that turns all of it into harm:

    IS THERE ENOUGH OIL, AT THIS TEMPERATURE, AT THIS SPEED, UNDER THIS
    LOAD, TO KEEP THE METAL APART?

That is one question with a standard answer, and everything an engine
dies of in service falls out of it.

THE CHAIN, WHICH IS A FEEDBACK LOOP AND THAT IS THE WHOLE POINT

  1. FILM THICKNESS. crankcase_state.film_kg is a mass on a known bore
     area, so h = m / (rho . A). Real thickness, from a real mass the
     sim is already integrating.

  2. REGIME. The specific film thickness lambda = h / sigma, where sigma
     is the combined surface roughness, is the standard measure of
     whether surfaces touch. Above ~3 the oil carries everything and the
     metal never meets (hydrodynamic). Below ~1 the asperities carry the
     load (boundary). Between, both (mixed). This is textbook tribology,
     not a curve anyone drew.

  3. FRICTION. Hydrodynamic sliding runs around mu = 0.002 -- oil
     shearing on oil. Boundary contact runs around mu = 0.12, sixty
     times worse, because it is metal dragging on metal. In the mixed
     regime the load is shared, so mu follows the share.

  4. HEAT, at the contact: P = mu . F . v. Real watts into a real small
     piece of metal.

  5. EXPANSION, and here is why engines seize rather than merely wear.
     A piston is aluminium (alpha ~23e-6/K) inside a cast-iron bore
     (alpha ~11e-6/K). The piston grows TWICE AS FAST as the hole it
     runs in. Heat the piston relative to the bore and the running
     clearance closes; close it to zero and the piston is an
     interference fit in a bore, moving at ten metres a second.

  6. WHICH FEEDS 1. A hotter piston means hotter oil, thinner oil, a
     thinner film, more contact, more heat. The loop closes with
     positive gain, which is why a seizure takes seconds once it starts
     and why nursing a dry engine does not work.

WHY THIS IS NOT A DAMAGE SCORE. Nothing here rolls a die or accumulates
an abstract "damage" number. Every step is a real quantity with a unit,
and the failure is reached when a real clearance reaches zero or a real
stress exceeds a real strength. An engine that is not harmed by a
condition is not harmed BECAUSE THE NUMBERS SAY SO, not because it
failed a check.

OVER-REVVING, SPECIFICALLY, because it is the condition most often
modelled as magic. Spinning an engine too fast does not "cause damage".
It does two real things:

  - RECIPROCATING INERTIA. The piston and rod must be stopped and
    reversed twice per revolution, and the force to do it goes with the
    SQUARE of speed: F = m . omega^2 . r . (cos t + (r/L) cos 2t), which
    is largest at TDC where it becomes m.omega^2.r.(1 + r/L). At TDC on
    the exhaust stroke there is no gas pressure to oppose it, so the rod
    is in pure TENSION and the rod bolts carry all of it. That is what
    "throwing a rod" is, and it is quadratic in rpm -- which is why the
    last few hundred rpm cost so much more than the first few thousand.

  - VALVE FLOAT, which valve_state already derives per valve from its
    own spring. A floating valve is simply late closing, and in an
    interference engine the piston arrives while it is still down.
    The energy in that collision is the piston's own kinetic energy,
    which is again quadratic in speed.

Neither is a threshold. Both are forces and energies that are either
enough to break something or not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# --- material and surface constants, all real properties --------------
#
# Thermal expansion coefficients are measured properties of the metals,
# not tuning. The RATIO between them is what drives seizure, and it is
# roughly two to one for every aluminium-piston-in-iron-bore engine ever
# built -- which is exactly why piston clearance exists at all.
ALUMINIUM_EXPANSION_PER_K = 23.1e-6
CAST_IRON_EXPANSION_PER_K = 11.0e-6
OIL_DENSITY_KG_M3 = 870.0

#: Combined RMS roughness of a honed bore and a piston skirt, metres.
#: A production hone is around 0.5 um Ra on the bore; the combined
#: sigma of two surfaces is the root sum of squares of each.
SURFACE_ROUGHNESS_COMBINED_M = 0.7e-6

#: The standard regime boundaries in specific film thickness. These are
#: the textbook values for where asperity contact begins and where it
#: takes over, not choices.
LAMBDA_FULL_FILM = 3.0
LAMBDA_BOUNDARY = 1.0

#: Sliding friction coefficients at the two ends of the Stribeck curve.
#: Real measured values for lubricated steel/iron pairs.
MU_HYDRODYNAMIC = 0.002
MU_BOUNDARY = 0.12

#: The clearance a piston is meant to RUN with, hot, as a fraction of
#: bore. This is the number that must stay positive; it is not the
#: number a machinist sets.
#:
#: WHY THIS AND NOT A COLD CLEARANCE. A first version of this module
#: took cold clearance as 0.001 of bore and let both parts expand from
#: ambient. That model says a perfectly healthy engine is seized, and
#: the module's own cross-check caught it: a piston at 520 K in a 380 K
#: bore came out at MINUS 335 um. Real crowns run near 570 K against a
#: 390 K bore and do not pick up.
#:
#: The premise was wrong. A machinist does not set a clearance and hope;
#: they set it SO THAT the running clearance is correct once everything
#: is hot. Cold clearance is therefore an OUTPUT of the design
#: temperatures, not an input -- see cold_clearance_m below. Seizure is
#: then what happens when the engine exceeds the differential it was
#: built for, which is the real failure, rather than something every
#: engine does on warm-up.
RUNNING_CLEARANCE_FRAC_OF_BORE = 0.0004

#: The differential the piston was designed around. The SKIRT is what
#: touches the bore, and it runs far cooler than the crown -- near 150 C
#: against a water-jacketed bore near 120 C. Using crown temperature
#: here would be asking whether the wrong part of the piston fits.
DESIGN_PISTON_TEMP_K = 423.15
DESIGN_BORE_TEMP_K = 393.15

#: How much of FREE aluminium expansion a real skirt actually delivers
#: across its diameter.
#:
#: A piston is not a solid cylinder of aluminium and must not be modelled
#: as one. It is CAM GROUND -- deliberately oval, with the thrust faces
#: cut down -- so that expansion takes up the ovality instead of closing
#: the clearance, and a cast piston is strutted with steel besides. The
#: skirt is also thin enough to deflect rather than jam. Together these
#: are the entire reason a piston can be aluminium in an iron bore at
#: all.
#:
#: The real machining range: shop practice cuts an aluminium piston
#: 0.001 to 0.0015 inch per inch of bore. This is the published anchor,
#: and it is the INPUT below rather than a target to check against
#: afterwards.
REAL_COLD_CLEARANCE_FRAC_RANGE = (0.0010, 0.0015)


def _solve_skirt_restraint() -> float:
    """SOLVED FROM THE SPEC, not chosen.

    Nobody publishes "effective skirt expansion as a fraction of free
    aluminium" -- it depends on the cam-grind profile and the struts,
    which are the manufacturer's business. What IS published is the cold
    clearance a piston gets cut to. That, the two expansion
    coefficients, and the design temperatures leave exactly one unknown,
    so it is solved rather than picked:

        cold/bore = running + restraint.a_al.dT_piston - a_fe.dT_bore

    Two hand attempts at this constant were wrong -- the first ignored
    the mechanism entirely and made every healthy engine seize, the
    second dropped the bore's own growth from the balance and landed at
    0.00065 in/in, outside the real range. Solving it removes the
    arithmetic from the loop and makes the provenance the spec itself."""
    target = sum(REAL_COLD_CLEARANCE_FRAC_RANGE) / 2.0
    d_piston = DESIGN_PISTON_TEMP_K - 293.15
    d_bore = DESIGN_BORE_TEMP_K - 293.15
    numerator = (target - RUNNING_CLEARANCE_FRAC_OF_BORE
                 + CAST_IRON_EXPANSION_PER_K * d_bore)
    return numerator / (ALUMINIUM_EXPANSION_PER_K * max(d_piston, 1e-9))


SKIRT_EXPANSION_RESTRAINT = _solve_skirt_restraint()

#: Rod bolts are torqued to a real preload and the rod cap parts when
#: tension exceeds it. A rod bolt set is typically sized so the assembly
#: carries around twice the inertia load at the engine's own redline --
#: the real design margin, stated rather than hidden.
ROD_ASSEMBLY_DESIGN_MARGIN_AT_REDLINE = 2.0


def oil_viscosity_pa_s(temp_k: float) -> float:
    """Dynamic viscosity of the oil at this temperature.

    Delegates to centrifuges.oil_viscosity_pa_s, which already solves a
    real SAE 40 against temperature. Deliberately NOT a third
    implementation -- hydraulics has one too, on the Walther equation,
    and a project with three viscosity curves has none it can trust."""
    import centrifuges
    return centrifuges.oil_viscosity_pa_s(float(temp_k))


@dataclass
class Lubrication:
    """The state of one sliding contact: how thick the film is, what
    regime that puts it in, and what that costs in friction and heat."""
    film_thickness_m: float
    specific_film: float          # lambda = h / sigma
    regime: str                   # "hydrodynamic" | "mixed" | "boundary"
    contact_frac: float           # 0 = oil carries all of it, 1 = metal does
    friction_coefficient: float
    friction_heat_w: float

    @property
    def metal_to_metal(self) -> bool:
        return self.contact_frac > 0.0


def lubrication(film_kg: float, bore_m: float, stroke_m: float,
                normal_force_n: float, sliding_speed_m_s: float,
                roughness_m: float = SURFACE_ROUGHNESS_COMBINED_M) -> Lubrication:
    """The Stribeck question, answered for one bore.

    film_kg comes straight from crankcase_state, which is already
    integrating it against splash, ring scrape and burn-off. This turns
    that mass into the thing that decides whether the engine survives:
    a thickness, compared against how rough the surfaces are."""
    area = math.pi * max(bore_m, 1e-6) * max(stroke_m, 1e-6)
    h = max(0.0, float(film_kg)) / (OIL_DENSITY_KG_M3 * area)
    sigma = max(roughness_m, 1e-12)
    lam = h / sigma
    if lam >= LAMBDA_FULL_FILM:
        regime, contact = "hydrodynamic", 0.0
    elif lam <= LAMBDA_BOUNDARY:
        regime, contact = "boundary", 1.0
    else:
        regime = "mixed"
        # the load shares between film and asperities across the
        # transition, linear in lambda between the two standard bounds
        contact = (LAMBDA_FULL_FILM - lam) / (LAMBDA_FULL_FILM - LAMBDA_BOUNDARY)
    mu = MU_HYDRODYNAMIC + (MU_BOUNDARY - MU_HYDRODYNAMIC) * contact
    heat = mu * max(0.0, normal_force_n) * max(0.0, sliding_speed_m_s)
    return Lubrication(film_thickness_m=h, specific_film=lam, regime=regime,
                       contact_frac=contact, friction_coefficient=mu,
                       friction_heat_w=heat)


@dataclass
class Fit:
    """What is left of the running clearance once things are hot."""
    cold_clearance_m: float
    hot_clearance_m: float
    piston_growth_m: float
    bore_growth_m: float
    seized: bool

    @property
    def closed_frac(self) -> float:
        if self.cold_clearance_m <= 0.0:
            return 1.0
        return max(0.0, 1.0 - self.hot_clearance_m / self.cold_clearance_m)


def cold_clearance_m(bore_m: float,
                     design_piston_k: float = DESIGN_PISTON_TEMP_K,
                     design_bore_k: float = DESIGN_BORE_TEMP_K,
                     assembly_temp_k: float = 293.15,
                     running_frac: float = RUNNING_CLEARANCE_FRAC_OF_BORE) -> float:
    """What the piston is cut to, cold, so that it runs right hot.

    The design rule, stated directly: cold clearance is the wanted
    running clearance PLUS however much the piston will out-grow the
    bore on the way to operating temperature. A bigger design
    differential means a looser cold fit, which is exactly why a forged
    piston (which expands more) is clearanced looser than a cast one and
    why a forged engine rattles until it warms up."""
    differential = ((ALUMINIUM_EXPANSION_PER_K * SKIRT_EXPANSION_RESTRAINT
                     * (design_piston_k - assembly_temp_k))
                    - (CAST_IRON_EXPANSION_PER_K * (design_bore_k - assembly_temp_k)))
    return bore_m * (running_frac + max(0.0, differential))


def running_fit(bore_m: float, piston_temp_k: float, bore_temp_k: float,
                assembly_temp_k: float = 293.15,
                clearance_frac: float | None = None) -> Fit:
    """Does the piston still fit the hole?

    Both parts grow; they just grow at different rates, and the
    DIFFERENCE is the entire mechanism. An engine running hot but evenly
    is fine -- piston and bore rise together and the clearance barely
    moves. An engine whose piston is hot while the bore is not is the
    one that seizes, which is exactly what a lost oil film does: it
    heats the piston locally and leaves the water-jacketed bore alone."""
    cold = (bore_m * clearance_frac if clearance_frac is not None
            else cold_clearance_m(bore_m, assembly_temp_k=assembly_temp_k))
    piston_growth = (ALUMINIUM_EXPANSION_PER_K * SKIRT_EXPANSION_RESTRAINT * bore_m
                     * (float(piston_temp_k) - assembly_temp_k))
    bore_growth = (CAST_IRON_EXPANSION_PER_K * bore_m
                   * (float(bore_temp_k) - assembly_temp_k))
    hot = cold + bore_growth - piston_growth
    return Fit(cold_clearance_m=cold, hot_clearance_m=hot,
               piston_growth_m=piston_growth, bore_growth_m=bore_growth,
               seized=hot <= 0.0)


# ---------------------------------------------------------------------
# over-revving, as force and energy
# ---------------------------------------------------------------------

def reciprocating_mass_kg(bore_m: float, stroke_m: float) -> float:
    """Piston, rings, pin and the small end of the rod, per cylinder.

    A piston is a short aluminium cup: its mass is its own wall and
    crown volume times the density of the alloy, and both scale with
    bore. Taking a crown thickness and skirt length as real fractions of
    bore -- which is how pistons are actually proportioned, because
    crown thickness is set by the gas load per unit area and that is
    bore-independent -- gives a mass that goes with bore^3, the real
    scaling. The rod's small end adds about a third again, the standard
    split between a rod's reciprocating and rotating halves."""
    d = max(bore_m, 1e-6)
    crown_t = 0.08 * d          # real proportion: crown thickness to bore
    skirt_len = 0.60 * d        # a modern slipper skirt is around 0.6 of bore
    wall_t = 0.045 * d
    crown_vol = math.pi * d * d / 4.0 * crown_t
    skirt_vol = math.pi * d * wall_t * skirt_len
    aluminium_density = 2700.0
    piston = (crown_vol + skirt_vol) * aluminium_density
    return piston * 1.33        # + the rod's reciprocating third


def inertia_force_at_tdc_n(rpm: float, bore_m: float, stroke_m: float,
                           rod_length_m: float = 0.0) -> float:
    """Peak reciprocating inertia force, at top dead centre.

    The exact slider-crank inertia term is
        F = m . omega^2 . r . (cos t + (r/L) . cos 2t)
    which at t = 0 is m.omega^2.r.(1 + r/L). On the exhaust stroke there
    is no gas pressure pushing back, so the rod carries this in pure
    tension and the rod bolts hold the cap on against all of it.

    Quadratic in rpm. That is the whole story of over-revving: going
    from 5000 to 6000 rpm is not 20% more load, it is 44% more."""
    r = max(stroke_m, 1e-9) / 2.0
    L = rod_length_m if rod_length_m > 0.0 else 1.75 * stroke_m  # real rod/stroke ratios sit near 1.75
    omega = max(0.0, float(rpm)) * 2.0 * math.pi / 60.0
    m = reciprocating_mass_kg(bore_m, stroke_m)
    return m * omega * omega * r * (1.0 + r / L)


@dataclass
class OverRev:
    """What spinning this fast actually does. No thresholds crossed,
    just forces compared against what holds them."""
    rpm: float
    redline_rpm: float
    inertia_force_n: float
    rod_capacity_n: float
    rod_margin: float             # capacity / load; below 1.0 it comes apart
    rod_failed: bool
    float_rpm: float
    floating: bool
    valve_contact_energy_j: float
    note: str = ""


def over_rev(engine, rpm: float, float_rpm: float | None = None) -> OverRev:
    """The two real mechanisms, evaluated at this speed.

    float_rpm should come from valve_state's own per-valve derivation
    (the worst valve's, which is the one that decides) when a live
    engine is available -- it accounts for spring sag and the deposit
    mass riding on the valve, so a tired engine floats earlier than a
    fresh one. Omitted, this falls back to the engine's declared redline
    as the float speed, which is what a healthy build is designed for."""
    arch = engine.architecture
    bore, stroke = arch.bore_m, arch.stroke_m
    rod_l = float(getattr(arch, "rod_length_m", 0.0) or 0.0)
    redline = float(getattr(engine, "redline_rpm", 0.0) or 0.0)

    load = inertia_force_at_tdc_n(rpm, bore, stroke, rod_l)
    # the rod assembly is built to hold the redline load with the real
    # design margin -- so its capacity is a property of the ENGINE's own
    # declared redline, not a number typed in here
    capacity = (inertia_force_at_tdc_n(redline, bore, stroke, rod_l)
                * ROD_ASSEMBLY_DESIGN_MARGIN_AT_REDLINE)
    margin = capacity / max(load, 1e-9)

    f_rpm = float(float_rpm) if float_rpm else redline
    floating = rpm > f_rpm > 0.0
    contact_j = 0.0
    if floating:
        # a floating valve is late closing; the piston arrives carrying
        # its own kinetic energy at mean piston speed, and that is what
        # lands on the valve head
        v = 2.0 * stroke * rpm / 60.0
        m = reciprocating_mass_kg(bore, stroke)
        # only the overspeed fraction of the stroke actually closes the
        # gap -- how far past float it is decides how late the valve is
        late = (rpm - f_rpm) / max(f_rpm, 1.0)
        contact_j = 0.5 * m * v * v * min(1.0, late)

    notes = []
    if load > capacity:
        notes.append(f"ROD: {load/1000.0:.1f} kN of inertia at TDC against a "
                     f"{capacity/1000.0:.1f} kN assembly -- the cap comes off")
    if floating:
        notes.append(f"FLOAT: valves float past {f_rpm:.0f} rpm; {contact_j:.0f} J "
                     "of piston energy arriving on a valve that has not closed")
    if not notes:
        notes.append(f"{load/1000.0:.1f} kN at TDC, {margin:.2f}x margin, valves "
                     "still following the cam")
    return OverRev(rpm=float(rpm), redline_rpm=redline, inertia_force_n=load,
                   rod_capacity_n=capacity, rod_margin=margin,
                   rod_failed=load > capacity, float_rpm=f_rpm, floating=floating,
                   valve_contact_energy_j=contact_j, note="; ".join(notes))


def report(engine, rpms=None) -> str:
    redline = float(getattr(engine, "redline_rpm", 4000.0) or 4000.0)
    if rpms is None:
        rpms = [redline * f for f in (0.5, 0.8, 1.0, 1.15, 1.3, 1.5)]
    lines = [f"over-rev accounting for {getattr(engine, 'identity', '?')} "
             f"(redline {redline:.0f})",
             f"  {'rpm':>6s} {'% redline':>10s} {'TDC load kN':>12s} {'rod margin':>11s}   what happens"]
    for r in rpms:
        o = over_rev(engine, r)
        lines.append(f"  {r:6.0f} {100.0 * r / redline:9.0f}% "
                     f"{o.inertia_force_n / 1000.0:12.1f} {o.rod_margin:11.2f}   {o.note}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import engines
    print(report(engines.get(sys.argv[1] if len(sys.argv) > 1 else "amc-258-jeep-i6")))
