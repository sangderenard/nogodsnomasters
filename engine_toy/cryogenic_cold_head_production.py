"""Parametric production graph for a cryogenic cold head.

The thermodynamic cycle already lives in :mod:`cryogenics`.  This module
gives that cycle the hardware and boundary needed by another simulation:

* a high-pressure working-gas inlet and a low-pressure cold return;
* separate hot and cold passages through one counterflow recuperator;
* a turboexpander with a real brake;
* a chamber-facing cold tip;
* an inner cold box, outer jacket and vacuum annulus with a service port.

This module declares the physical parts and connections only. Their mechanics,
fluid flow and heat transfer belong to engine-toy's existing simulations.
"""
from __future__ import annotations

import math

from compressors import DiaphragmCompressorSet
from turret_production import ProductionGraph


def build(identity: str = "cryo.cold_head", *,
          mass_flow_kg_s: float = 0.05,
          inlet_pressure_pa: float = 4.0e6,
          return_pressure_pa: float = 1.5e5,
          recuperator_effectiveness: float = 0.97,
          brake: str = "generator") -> ProductionGraph:
    """Author the parametric cold-head parts and working-gas paths."""
    g = ProductionGraph(identity=f"{identity}/production")
    g.assembly = identity

    # The jacket is the warm load-bearing body.  The inner box is carried
    # through low-conductance supports across a declared evacuated volume.
    g.node(f"{identity}.outer_jacket", (0.0, 0.18, 0.0), "vacuum-jacket",
           half_extent_m=(0.15, 0.20, 0.15), shape="drum",
           drum_axis=(0.0, 1.0, 0.0), drum_radius_m=0.15,
           drum_length_m=0.40, material="stainless-steel",
           thermal_domain="shell",
           thermal_shell_thickness_m=0.003,
           thermal_group=f"{identity}.warm_jacket")
    g.node(f"{identity}.inner_cold_box", (0.0, 0.15, 0.0), "cold-box",
           half_extent_m=(0.11, 0.15, 0.11), shape="drum",
           drum_axis=(0.0, 1.0, 0.0), drum_radius_m=0.11,
           drum_length_m=0.30, material="aluminium-casting",
           thermal_domain="shell",
           thermal_shell_thickness_m=0.002,
           thermal_group=f"{identity}.cold_end")
    g.node(f"{identity}.vacuum_annulus", (0.0, 0.18, 0.0), "vacuum-volume",
           half_extent_m=(0.14, 0.19, 0.14), mass_kg=0.0,
           render_primitive=False, structural_participation=False,
           thermal_domain="none",
           fluid="gas",
           fluid_volume_l=math.pi * (
               0.14 ** 2 * 0.38 - 0.11 ** 2 * 0.30) * 1000.0,
           vacuum_intact=True, pressure_pa=1.0,
           separates=(f"{identity}.outer_jacket", f"{identity}.inner_cold_box"))

    # Two passages through one recuperator.  Their identities remain
    # separate because they carry opposite-flowing streams at different
    # pressures and temperatures.
    g.node(f"{identity}.recuperator_high", (-0.045, 0.18, 0.0),
           "counterflow-recuperator-passage", half_extent_m=(0.035, 0.12, 0.035),
           shape="drum", drum_axis=(0.0, 1.0, 0.0), drum_radius_m=0.035,
           drum_length_m=0.24, material="aluminium-casting",
           fluid="compressed-air", pressure_pa=inlet_pressure_pa,
           fluid_volume_l=math.pi * 0.035 ** 2 * 0.24 * 1000.0,
           flow_direction="warm-to-cold",
           exchanges_with=f"{identity}.recuperator_return",
           effectiveness=recuperator_effectiveness,
           thermal_domain="shell",
           thermal_shell_thickness_m=0.0015,
           thermal_group=f"{identity}.cold_end")
    g.node(f"{identity}.recuperator_return", (0.045, 0.18, 0.0),
           "counterflow-recuperator-passage", half_extent_m=(0.035, 0.12, 0.035),
           shape="drum", drum_axis=(0.0, -1.0, 0.0), drum_radius_m=0.035,
           drum_length_m=0.24, material="aluminium-casting",
           fluid="cold-air", pressure_pa=return_pressure_pa,
           fluid_volume_l=math.pi * 0.035 ** 2 * 0.24 * 1000.0,
           flow_direction="cold-to-warm",
           exchanges_with=f"{identity}.recuperator_high",
           effectiveness=recuperator_effectiveness,
           thermal_domain="shell",
           thermal_shell_thickness_m=0.0015,
           thermal_group=f"{identity}.cold_end")
    g.node(f"{identity}.expander", (0.0, 0.01, 0.0), "turboexpander",
           half_extent_m=(0.075, 0.055, 0.075), shape="drum",
           drum_axis=(1.0, 0.0, 0.0), drum_radius_m=0.075,
           drum_length_m=0.11, material="aluminium-casting",
           inlet_pressure_pa=inlet_pressure_pa,
           outlet_pressure_pa=return_pressure_pa,
           mass_flow_kg_s=mass_flow_kg_s,
           isentropic_efficiency=0.82,
           heat_capacity_ratio=1.40,
           specific_heat_j_kg_k=1005.0,
           thermal_domain="volume",
           thermal_group=f"{identity}.cold_end")
    g.node(f"{identity}.brake", (0.12, 0.01, 0.0), "generator-brake",
           half_extent_m=(0.045, 0.055, 0.055), material="copper",
           thermal_domain="volume",
           brake_kind=brake)
    g.edge(f"{identity}.expander_brake_shaft", f"{identity}.expander",
           f"{identity}.brake", "shaft-service-drive", radius=0.012,
           medium_rate_state="expander-shaft-power-w")

    # The only process-facing body.  Placement may put any fraction of this
    # copper finger through a chamber boundary; the world coupling computes
    # the occupied and exposed fractions from this geometry.
    g.node(f"{identity}.cold_tip", (0.0, -0.105, 0.0), "cold-tip-heat-exchanger",
           half_extent_m=(0.045, 0.085, 0.045), shape="drum",
           drum_axis=(0.0, 1.0, 0.0), drum_radius_m=0.045,
           drum_length_m=0.17, material="copper",
           thermal_domain="volume",
           thermal_group=f"{identity}.cold_end",
           heat_exchange_ua_w_per_k=120.0,
           interface_role="process-heat-exchange-surface")
    g.edge(f"{identity}.tip_mount", f"{identity}.inner_cold_box",
           f"{identity}.cold_tip", "bolted-flange-mount", radius=0.020,
           medium_rate_state="process-heat-flow-w")

    # External supply and return are distinct boundary ports.  The high side
    # passes through the recuperator and expander; the cold low side returns
    # through the other recuperator passage before it leaves the machine.
    g.node(f"{identity}.high_pressure_inlet", (-0.055, 0.41, 0.0),
           "gas-service-port", half_extent_m=(0.015, 0.015, 0.015),
           thermal_domain="none",
           wrench_point=True, supplied_elsewhere=True,
           port_role="high-pressure-working-gas-inlet",
           pressure_pa=inlet_pressure_pa, fluid="compressed-air")
    g.node(f"{identity}.low_pressure_return", (0.055, 0.41, 0.0),
           "gas-service-port", half_extent_m=(0.020, 0.020, 0.020),
           thermal_domain="none",
           wrench_point=True, supplied_elsewhere=True,
           port_role="low-pressure-cold-gas-return",
           pressure_pa=return_pressure_pa, fluid="cold-air")
    g.edge(f"{identity}.supply_line", f"{identity}.high_pressure_inlet",
           f"{identity}.recuperator_high", "pressure-rated-air-line",
           radius=0.009, circuit_identity=f"{identity}.high-pressure-gas",
           part_role="through-port", medium_rate_state="mass-flow-pressure-temperature")
    g.edge(f"{identity}.high_to_expander", f"{identity}.recuperator_high",
           f"{identity}.expander", "pressure-rated-air-line", radius=0.008,
           circuit_identity=f"{identity}.high-pressure-gas",
           medium_rate_state="mass-flow-pressure-temperature")
    g.edge(f"{identity}.expander_to_cold_tip", f"{identity}.expander",
           f"{identity}.cold_tip", "air-line", radius=0.012,
           circuit_identity=f"{identity}.cold-return-gas",
           medium_rate_state="mass-flow-pressure-temperature")
    g.edge(f"{identity}.cold_tip_to_return", f"{identity}.cold_tip",
           f"{identity}.recuperator_return", "air-line", radius=0.012,
           circuit_identity=f"{identity}.cold-return-gas",
           medium_rate_state="mass-flow-pressure-temperature")
    g.edge(f"{identity}.return_line", f"{identity}.recuperator_return",
           f"{identity}.low_pressure_return", "air-line", radius=0.014,
           circuit_identity=f"{identity}.cold-return-gas",
           part_role="through-port", medium_rate_state="mass-flow-pressure-temperature")

    # Vacuum is serviceable hardware.  Losing this boundary changes the
    # annulus pressure and therefore the insulation state used by the thermal system.
    g.node(f"{identity}.vacuum_service", (0.15, 0.27, 0.0),
           "vacuum-service-port", half_extent_m=(0.015, 0.015, 0.015),
           thermal_domain="none",
           wrench_point=True, supplied_elsewhere=True,
           port_role="vacuum-service", fluid="vacuum")
    g.edge(f"{identity}.vacuum_service_line", f"{identity}.vacuum_service",
           f"{identity}.vacuum_annulus", "vacuum-line", radius=0.005,
           circuit_identity=f"{identity}.vacuum-service",
           part_role="through-port", medium_rate_state="annulus-pressure-pa")

    for i, x in enumerate((-0.10, 0.10), 1):
        mount = f"{identity}.mount_{i}"
        g.node(mount, (x, 0.36, 0.0), "structural-mount",
               half_extent_m=(0.025, 0.015, 0.025),
               thermal_domain="volume",
               material="stainless-steel",
               port_role="structural-mount", direction=(0.0, 1.0, 0.0),
               joint="bolted-flange")
        g.edge(f"{mount}.joint", mount, f"{identity}.outer_jacket",
               "bolted-flange-mount", radius=0.012)
    return g
