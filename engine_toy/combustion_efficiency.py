"""How much of the fuel actually burns, computed instead of declared.

Engine.combustion_efficiency is a bare number -- 0.840 on the catalogue's
AMC 258 -- and nothing in the project derives it. The sim's own flame
model is not that quantity: _burned_frac tracks how far through a burn
the flame has got, ramping 0 to 1 every event, and it reaches exactly
1.0 on two thirds of its samples. That is burn RATE. Every burn
completes, because nothing in it can fail to.

Combustion efficiency is a different question: of the fuel that went in,
how much released its chemical energy at all? The answer is never 1.0,
and it is not one number per engine either -- it moves with mixture and
with the chamber's own geometry. Three real mechanisms, and nothing
else is needed:

  1. OXYGEN, WHEN RICH. Past stoichiometric there is simply not enough
     air to oxidise all the fuel, and at most 1/phi of it can burn. This
     is exact stoichiometry, not a correlation, and it dominates
     everything else the moment an engine enriches: at phi 1.2 a fifth
     of the extra fuel has nothing to react with. It is also why a rich
     engine makes more power (it burns all the AIR, which is what is
     actually limited) while getting worse specific consumption.

  2. CREVICES. At peak pressure the charge is squeezed into the gap
     between piston crown and top ring, which is far too narrow for a
     flame to propagate into. That fuel comes back out during expansion
     unburnt. The crevice is a fixed real volume, so what matters is its
     size relative to the CLEARANCE volume -- and clearance volume is
     Vd/(CR-1), which means a high-compression engine has a small
     clearance volume and therefore loses a LARGER fraction to the same
     physical crevice. A real and slightly counterintuitive result.

  3. WALL QUENCHING. A flame cannot survive closer to a cold wall than
     its quenching distance, because the wall pulls heat out faster than
     the reaction makes it. The standard form is a Peclet number:

         delta_q  =  Pe . alpha / S_L

     with alpha the charge's thermal diffusivity and S_L the laminar
     flame speed -- the same constant the sim's own flame model already
     declares. The quenched mass is the chamber's surface area times
     that distance, over its volume, so a chamber with a lot of wall for
     its size loses more. That is why a small bore has worse unburnt
     hydrocarbons than a large one at the same compression ratio, and
     why a pancake chamber is worse than a compact one.

WHAT THIS DELIBERATELY DOES NOT DO. It does not replace
Engine.combustion_efficiency by fiat. The declared value is a real
figure for a real engine and may encode things this does not model.
`compare` puts the two side by side so a disagreement is visible, which
is the same treatment torque_curve.audit_anchor gives the declared BMEP:
where a derivation and a declaration differ, one of them is wrong and
both answers are worth having.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Peclet number for flame quenching at a cold wall. A real, standard
#: value for hydrocarbon-air two-plate quenching; it is the ratio the
#: quenching distance is DEFINED by, not a fitted constant.
QUENCH_PECLET = 50.0

#: Thermal diffusivity of the burnt-gas-side charge at end of
#: compression, m2/s. From k/(rho.cp) at roughly 700 K and 20 bar --
#: real transport properties at real in-cylinder conditions, not
#: ambient ones, because the quenching distance at ambient is nearly an
#: order of magnitude larger and would be the wrong number entirely.
CHARGE_THERMAL_DIFFUSIVITY_M2_S = 4.6e-6

#: Laminar flame speed at those conditions. Higher than the 0.4 m/s the
#: sim declares for its own torque kernel, because that figure is quoted
#: near atmospheric and flame speed rises steeply with temperature.
FLAME_SPEED_AT_COMPRESSION_M_S = 1.0

#: Top-land crevice volume as a fraction of DISPLACED volume per
#: cylinder. A real piston's top land is a few tenths of a millimetre
#: wide over the bore circumference; expressed against displacement this
#: is the standard order quoted for production pistons.
CREVICE_FRAC_OF_DISPLACEMENT = 0.0025


def quench_distance_m(flame_speed_m_s: float | None = None,
                      diffusivity_m2_s: float | None = None,
                      peclet: float | None = None) -> float:
    """How close to a wall a flame can get before it dies.

    Pe = delta_q . S_L / alpha, rearranged. Lands near a quarter of a
    millimetre at in-cylinder conditions, which is the right order for
    a running engine -- the millimetre-scale figure often quoted is the
    ATMOSPHERIC one and does not apply inside a compressed charge."""
    # read at call time: a module constant used as a default argument is
    # frozen at import and cannot be swept or fitted. See the same note
    # in engine_harm.cold_clearance_m.
    flame_speed_m_s = (FLAME_SPEED_AT_COMPRESSION_M_S if flame_speed_m_s is None
                       else flame_speed_m_s)
    diffusivity_m2_s = (CHARGE_THERMAL_DIFFUSIVITY_M2_S if diffusivity_m2_s is None
                        else diffusivity_m2_s)
    peclet = QUENCH_PECLET if peclet is None else peclet
    return peclet * diffusivity_m2_s / max(flame_speed_m_s, 1e-6)


@dataclass
class Completeness:
    """What fraction of the fuel burned, and what stopped the rest."""
    efficiency: float
    oxygen_limit: float          # the 1/phi ceiling, 1.0 when not rich
    crevice_loss: float
    quench_loss: float
    quench_distance_m: float
    note: str = ""


def completeness(bore_m: float, stroke_m: float, compression_ratio: float,
                 phi: float = 1.0,
                 crevice_frac: float | None = None) -> Completeness:
    """Combustion efficiency from geometry and mixture, nothing fitted.

    Every input is something the engine already declares. The three
    mechanisms are independent and multiply, because a molecule trapped
    in a crevice is not also available to be quenched at a wall."""
    cr = max(float(compression_ratio), 1.0001)
    disp = math.pi * bore_m * bore_m / 4.0 * stroke_m
    clearance = disp / (cr - 1.0)

    # --- 1. oxygen, when rich ------------------------------------------
    p = max(float(phi), 1e-6)
    oxygen = min(1.0, 1.0 / p)

    # --- 2. crevices ---------------------------------------------------
    # a fixed physical volume, measured against the clearance volume it
    # sits beside -- so a high-CR engine loses a bigger FRACTION
    crevice_frac = (CREVICE_FRAC_OF_DISPLACEMENT if crevice_frac is None
                    else crevice_frac)
    crevice_vol = crevice_frac * disp
    crevice_loss = min(0.15, crevice_vol / max(clearance, 1e-12))

    # --- 3. wall quenching ---------------------------------------------
    # the chamber at TDC as a disc of bore diameter: two faces and a rim
    h = clearance / (math.pi * bore_m * bore_m / 4.0)
    wall_area = 2.0 * (math.pi * bore_m * bore_m / 4.0) + math.pi * bore_m * h
    dq = quench_distance_m()
    quench_loss = min(0.15, wall_area * dq / max(clearance, 1e-12))

    eff = oxygen * (1.0 - crevice_loss) * (1.0 - quench_loss)
    bits = []
    if oxygen < 1.0:
        bits.append(f"rich at phi {p:.2f}: only {oxygen * 100:.0f}% of the fuel "
                    "can find oxygen")
    bits.append(f"crevice {crevice_loss * 100:.1f}%")
    bits.append(f"quench {quench_loss * 100:.1f}% at {dq * 1e6:.0f} um")
    return Completeness(efficiency=max(0.0, min(1.0, eff)), oxygen_limit=oxygen,
                        crevice_loss=crevice_loss, quench_loss=quench_loss,
                        quench_distance_m=dq, note="; ".join(bits))


def for_engine(engine, phi: float = 1.0) -> Completeness:
    arch = engine.architecture
    cr = float(getattr(engine, "compression_ratio", 0.0)
               or getattr(arch, "compression_ratio", 9.0) or 9.0)
    return completeness(arch.bore_m, arch.stroke_m, cr, phi)


def compare(engine, phi: float = 1.0) -> dict:
    """Derived against declared, as a disagreement rather than a score.

    Same treatment torque_curve.audit_anchor gives the declared BMEP: if
    the two differ badly one of them is wrong, and knowing which way is
    more useful than quietly preferring either."""
    c = for_engine(engine, phi)
    declared = float(getattr(engine, "combustion_efficiency", 0.0) or 0.0)
    return {"identity": getattr(engine, "identity", "?"),
            "derived": c.efficiency, "declared": declared,
            "disagreement": (c.efficiency / declared - 1.0) if declared > 0 else float("nan"),
            "why": c.note}


def audit_catalogue(phi: float = 1.0) -> str:
    import engines as _eng
    rows = []
    for ident, e in _eng.BY_IDENTITY.items():
        if getattr(e, "kind", "") != "combustion":
            continue
        if not getattr(e.architecture, "bore_m", 0.0):
            continue
        try:
            rows.append(compare(e, phi))
        except Exception:
            continue
    rows.sort(key=lambda r: -abs(r.get("disagreement", 0.0)))
    out = [f"combustion efficiency, derived vs declared (phi = {phi:.2f})",
           f"  {'engine':28s} {'derived':>8s} {'declared':>9s} {'gap':>8s}   why"]
    for r in rows:
        out.append(f"  {r['identity'][:28]:28s} {r['derived']:8.3f} {r['declared']:9.3f} "
                   f"{r['disagreement'] * 100:7.1f}%   {r['why']}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(audit_catalogue(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0))
