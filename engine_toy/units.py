"""Customary units, for reading. SI stays the only thing computed in.

Every number this project calculates is SI and stays SI. This module
exists solely so a figure can be READ by someone who thinks in
horsepower and pound-feet, which is most people talking about engines,
and specifically the people who quote the catalogue figures the anchor
comes from.

WHY IT IS A MODULE AND NOT A FACTOR TYPED WHERE IT IS NEEDED. There is
no conversion factor anywhere in this project right now -- checked --
and this is the moment one would be introduced. Introduced at a call
site it gets copied to the next call site, and then 745.7 and 746 and
0.746 are all in the repo meaning the same thing, and one of them is
the electrical horsepower and one is the metric one and nobody can tell
which by looking. Defined once, with the definition next to it, that
cannot happen.

WHICH HORSEPOWER. The mechanical one, which is what "bhp" means on an
engine spec sheet, defined EXACTLY as 550 foot-pounds-force per second.
Not the metric horsepower (PS/CV, 735.49875 W) which is what European
sheets often quote, and not the electrical one (746 W exactly). They
differ by around 1.4%, which is small enough to look like rounding and
large enough to matter when comparing against a published figure -- so
the distinction is named here rather than discovered later. PS is
provided too, for exactly that reason: a figure quoted for a German or
Japanese engine is very often PS, and silently reading it as bhp
overstates the engine by 1.4%.

Every constant below is a definition or a product of definitions, not a
measurement, so they are exact:
    inch            = 0.0254 m                      (defined, 1959)
    pound-force     = 0.45359237 kg x 9.80665 m/s2  (both defined)
    foot            = 12 inches
    horsepower      = 550 ft.lbf/s
"""
from __future__ import annotations

# --- the four definitions everything else is built from ---------------
METRE_PER_INCH = 0.0254                 # exact, international yard and pound, 1959
KG_PER_POUND = 0.45359237               # exact, same agreement
STANDARD_GRAVITY_M_S2 = 9.80665         # exact, by definition of kgf/lbf
METRE_PER_FOOT = METRE_PER_INCH * 12.0  # exact: 0.3048

# --- and the products of them -----------------------------------------
NEWTON_PER_POUND_FORCE = KG_PER_POUND * STANDARD_GRAVITY_M_S2      # 4.4482216152605
NM_PER_POUND_FOOT = NEWTON_PER_POUND_FORCE * METRE_PER_FOOT        # 1.3558179483314004
WATT_PER_HORSEPOWER = 550.0 * NM_PER_POUND_FOOT                    # 745.6998715822702
WATT_PER_METRIC_HORSEPOWER = 75.0 * STANDARD_GRAVITY_M_S2          # 735.49875, exact -- PS/CV

#: 1 slug = 1 lbf / (1 ft/s^2). Included because it is the coherent
#: imperial mass unit and someone will eventually ask; nothing in this
#: project computes in it, and nothing should.
KG_PER_SLUG = NEWTON_PER_POUND_FORCE / METRE_PER_FOOT              # 14.593902937206364


def hp(watts: float) -> float:
    """Mechanical horsepower (bhp) from watts."""
    return float(watts) / WATT_PER_HORSEPOWER


def kw_to_hp(kw: float) -> float:
    return hp(float(kw) * 1000.0)


def ps(watts: float) -> float:
    """Metric horsepower (PS/CV) from watts -- 1.4% larger a count than
    bhp for the same engine, which is why it is named separately."""
    return float(watts) / WATT_PER_METRIC_HORSEPOWER


def lb_ft(newton_metres: float) -> float:
    return float(newton_metres) / NM_PER_POUND_FOOT


def power(kw: float) -> str:
    """'123.4 kW (165.5 bhp)' -- SI first, because SI is what was
    computed; the customary figure is the parenthetical gloss."""
    return f"{float(kw):.1f} kW ({kw_to_hp(kw):.1f} bhp)"


def torque(nm: float) -> str:
    """'280.9 Nm (207.2 lb-ft)'."""
    return f"{float(nm):.1f} Nm ({lb_ft(nm):.1f} lb-ft)"


if __name__ == "__main__":
    print("exact definitions, derived:")
    print(f"  1 lbf   = {NEWTON_PER_POUND_FORCE!r} N")
    print(f"  1 lb-ft = {NM_PER_POUND_FOOT!r} Nm")
    print(f"  1 bhp   = {WATT_PER_HORSEPOWER!r} W   (550 ft.lbf/s)")
    print(f"  1 PS    = {WATT_PER_METRIC_HORSEPOWER!r} W   (75 kgf.m/s)")
    print(f"  1 slug  = {KG_PER_SLUG!r} kg")
    print()
    print("  100 kW reads as", power(100.0), "or", f"{ps(100_000.0):.1f} PS")
    print("  and 1 bhp is", f"{WATT_PER_HORSEPOWER / WATT_PER_METRIC_HORSEPOWER:.4f}",
          "PS -- the 1.4% that looks like rounding")
