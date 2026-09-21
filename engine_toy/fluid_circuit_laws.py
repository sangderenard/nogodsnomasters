"""Symbolic transport laws used by the graph-discovered fluid circuits.

The circuit graph decides *where* fluid can move.  These equations decide
how much crosses one declared edge.  They are SymPy expressions so the same
law can be reduced and compiled with the rest of the repository rather than
being re-authored in each runtime.

All pressure arguments are absolute.  The signed wrappers orient positive
flow from endpoint ``a`` to endpoint ``b``; the magnitude laws take an
upstream state and therefore never receive a negative pressure drop.
"""
from __future__ import annotations

import sympy as sp


R_UNIVERSAL_J_MOL_K = sp.Float("8.31446261815324")


def ideal_gas_pressure_pa(amount_mol, temperature_k, volume_m3):
    """Ideal-mixture closure ``P V = n R T`` for one rigid volume."""

    return amount_mol * R_UNIVERSAL_J_MOL_K * temperature_k / volume_m3


def compressible_orifice_mass_flow_kg_s(
    upstream_pressure_pa,
    downstream_pressure_pa,
    upstream_temperature_k,
    area_m2,
    specific_gas_constant_j_kg_k,
    heat_capacity_ratio,
    discharge_coefficient,
):
    """Saint-Venant/Wanzel isentropic nozzle flow, including choking.

    This is the general form of ``drivetrain_graph``'s existing air-only
    choked-orifice law.  Below the critical pressure ratio it uses the choked
    branch; above it the subcritical branch retains downstream backpressure.
    """

    p0 = sp.sympify(upstream_pressure_pa)
    p1 = sp.sympify(downstream_pressure_pa)
    t0 = sp.sympify(upstream_temperature_k)
    area = sp.sympify(area_m2)
    gas_r = sp.sympify(specific_gas_constant_j_kg_k)
    gamma = sp.sympify(heat_capacity_ratio)
    cd = sp.sympify(discharge_coefficient)
    ratio = p1 / p0
    critical = (sp.Integer(2) / (gamma + 1)) ** (gamma / (gamma - 1))
    choked = (
        cd * area * p0 * sp.sqrt(gamma / (gas_r * t0))
        * (sp.Integer(2) / (gamma + 1))
        ** ((gamma + 1) / (2 * (gamma - 1)))
    )
    subcritical = cd * area * p0 * sp.sqrt(
        2 * gamma / (gas_r * t0 * (gamma - 1))
        * (ratio ** (sp.Rational(2) / gamma)
           - ratio ** ((gamma + 1) / gamma))
    )
    return sp.Piecewise(
        (sp.Integer(0), sp.Or(p0 <= 0, t0 <= 0, area <= 0, p1 >= p0)),
        (choked, ratio <= critical),
        (subcritical, True),
    )


def signed_compressible_orifice_mass_flow_kg_s(
    pressure_a_pa,
    pressure_b_pa,
    temperature_a_k,
    temperature_b_k,
    area_m2,
    specific_gas_constant_j_kg_k,
    heat_capacity_ratio,
    discharge_coefficient,
):
    """Bidirectional compressible flow; positive means ``a -> b``."""

    forward = compressible_orifice_mass_flow_kg_s(
        pressure_a_pa, pressure_b_pa, temperature_a_k, area_m2,
        specific_gas_constant_j_kg_k, heat_capacity_ratio,
        discharge_coefficient,
    )
    reverse = compressible_orifice_mass_flow_kg_s(
        pressure_b_pa, pressure_a_pa, temperature_b_k, area_m2,
        specific_gas_constant_j_kg_k, heat_capacity_ratio,
        discharge_coefficient,
    )
    return sp.Piecewise(
        (forward, pressure_a_pa > pressure_b_pa),
        (-reverse, pressure_b_pa > pressure_a_pa),
        (sp.Integer(0), True),
    )


def hagen_poiseuille_mass_flow_kg_s(
    pressure_a_pa,
    pressure_b_pa,
    radius_m,
    length_m,
    dynamic_viscosity_pa_s,
    density_kg_m3,
):
    """Signed laminar round-pipe flow from Hagen-Poiseuille."""

    return (
        sp.pi * radius_m ** 4 * (pressure_a_pa - pressure_b_pa)
        / (8 * dynamic_viscosity_pa_s * length_m)
        * density_kg_m3
    )


def convected_enthalpy_w(mass_flow_kg_s, specific_enthalpy_j_kg):
    """Energy carried by a material stream, ``Hdot = mdot h``."""

    return mass_flow_kg_s * specific_enthalpy_j_kg


__all__ = [
    "R_UNIVERSAL_J_MOL_K",
    "ideal_gas_pressure_pa",
    "compressible_orifice_mass_flow_kg_s",
    "signed_compressible_orifice_mass_flow_kg_s",
    "hagen_poiseuille_mass_flow_kg_s",
    "convected_enthalpy_w",
]
