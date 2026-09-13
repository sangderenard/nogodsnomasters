"""The auxiliary plant as real hardware in the graph.

Everything the treatment chain, the refrigerant loop, the hydraulic
tank and their controls actually ARE, laid out as nodes and edges so
they render, can be shot, leak through hole_emitters, burst, and show
up in node_effects like every other part.

    air:     compressor -> aftercooler (electric fan) -> chiller ->
             water separator -> coalescing filter -> particulate
             filter -> reheater -> wet tank -> pneumatic manifold ->
             isolated reserve set
    cold:    AC compressor -> condenser -> receiver-drier -> the two
             chiller evaporators (air, hydraulic) and the cabin
    oil:     hydraulic tank (with its chiller coil) -> hydronic manifold
    control: the chiller/tank control panel, the pressure switches, the
             drain solenoids
    power:   an accessory battery bank behind a charge isolator, so the
             plant's own electric fans, heater and solenoids cannot
             flatten the engine's starting battery

The positions are all derived from the engine's own real envelope, the
same way every other bolt-on here is placed -- this is a skid of
equipment alongside the engine, not a floating decoration.
"""
from __future__ import annotations

import math

import numpy as np

CIRCUIT_AIR = "pneumatic-reserve"
CIRCUIT_REFRIGERANT = "refrigerant"
CIRCUIT_HYDRAULIC = "hydraulic"


def _drum(node, identity, centre, axis, radius, length, **kw):
    node(identity, [float(v) for v in centre], kw.pop("kind", "engine-block-component"),
         drum_axis=[float(v) for v in axis], drum_radius_m=float(radius), drum_length_m=float(length),
         body_half_extent_m=[float(length) / 2.0, float(radius), float(radius)], **kw)


def emit_auxiliary_plant(engine, nodes, edges, node, edge) -> None:
    """Called after the pneumatic hardware exists, so the wet tank can
    be plumbed into the reserve set that is already there."""
    plant = getattr(engine, "auxiliary_plant", None)
    if plant is None or not plant.fitted:
        return
    node_by_id = {n["identity"]: n for n in nodes}
    # anchor on whatever the pneumatic block actually built
    src = node_by_id.get("pneumatic_compressor") or node_by_id.get("electrical.pneumatic_compressor")
    if src is None:
        return
    base = np.asarray(src["reference_position"], dtype=float)
    # the treatment skid runs outboard of the compressor, along -z
    x0, y0, z0 = float(base[0]), float(base[1]) - 0.05, float(base[2]) - 0.30
    ax = [1.0, 0.0, 0.0]
    step = 0.26

    def at(i, dz=0.0, dy=0.0):
        return (x0 - 0.10 - i * 0.02, y0 + dy, z0 - i * step + dz)

    # ---------------- the air treatment train ----------------
    prev = "pneumatic_compressor" if "pneumatic_compressor" in node_by_id else "electrical.pneumatic_compressor"
    chain = []

    _drum(node, "plant.aftercooler", at(0), ax, 0.085, 0.30, mass_kg=6.0, mass_in_total=True,
          material="aluminium-casting", exchanger_kind="air-to-air-aftercooler", fluid="compressed-air",
          fluid_volume_l=1.2, stage="aftercooler")
    node("plant.aftercooler_fan", [x0 - 0.10, y0 + 0.16, z0], "electric-fan", mass_kg=1.8, mass_in_total=True,
         fan_disk_radius_m=0.10, rated_w=90.0, drive="electric-motor")
    edge("plant.aftercooler_fan_mount", "plant.aftercooler_fan", "plant.aftercooler", "rigid-bolted-joint", radius=0.005)
    chain.append(("plant.aftercooler", 0.012))

    _drum(node, "plant.air_chiller", at(1), ax, 0.080, 0.28, mass_kg=7.5, mass_in_total=True,
          material="aluminium-casting", exchanger_kind="refrigerant-to-air-chiller", fluid="compressed-air",
          fluid_volume_l=1.0, stage="chiller")
    chain.append(("plant.air_chiller", 0.012))

    _drum(node, "plant.water_separator", at(2), [0.0, 1.0, 0.0], 0.055, 0.26, mass_kg=3.2, mass_in_total=True,
          material="aluminium-casting", separator_kind="centrifugal", fluid="condensate", fluid_volume_l=0.4,
          stage="separator")
    node("plant.separator_drain", [at(2)[0], at(2)[1] - 0.15, at(2)[2]], "drain-solenoid", mass_kg=0.4,
         mass_in_total=True, body_half_extent_m=[0.03, 0.03, 0.03], drive="electric-solenoid")
    edge("plant.separator_to_drain", "plant.water_separator", "plant.separator_drain", "condensate-line",
         radius=0.004, circuit_identity="condensate", material="nylon-airline")
    chain.append(("plant.water_separator", 0.010))

    _drum(node, "plant.coalescing_filter", at(3), [0.0, 1.0, 0.0], 0.048, 0.30, mass_kg=2.4, mass_in_total=True,
          material="aluminium-casting", filter_kind="coalescing", fluid="compressed-air", fluid_volume_l=0.5,
          stage="coalescing")
    chain.append(("plant.coalescing_filter", 0.010))

    _drum(node, "plant.particulate_filter", at(4), [0.0, 1.0, 0.0], 0.048, 0.28, mass_kg=2.2, mass_in_total=True,
          material="aluminium-casting", filter_kind="particulate", fluid="compressed-air", fluid_volume_l=0.45,
          stage="particulate")
    chain.append(("plant.particulate_filter", 0.010))

    _drum(node, "plant.reheater", at(5), ax, 0.060, 0.24, mass_kg=3.0, mass_in_total=True,
          material="steel-pipe", exchanger_kind="electric-reheater", rated_w=2500.0, fluid="compressed-air",
          fluid_volume_l=0.6, stage="reheater")
    chain.append(("plant.reheater", 0.010))

    # the wet tank: the last place liquid can fall out before the set
    wet_l = max(6.0, plant.wet_tank_capacity_l)
    wv = wet_l / 1000.0
    wr = (wv / (3.0 * math.pi)) ** (1.0 / 3.0)
    wl = 6.0 * wr
    wet_pos = (x0 - 0.25, y0 - 0.12, z0 - 6 * step)
    _drum(node, "plant.wet_tank", wet_pos, ax, wr, wl, kind="high-pressure-canister",
          mass_kg=max(4.0, wet_l * 0.9), mass_in_total=True, material="pressed-steel",
          working_pressure_pa=engine.pneumatics.tank_pressure_pa,
          capacity_kg=wv * 1.2 * engine.pneumatics.tank_pressure_pa / 101_325.0,
          tank_kind="wet-tank", fluid="compressed-air", fluid_volume_l=wet_l, stage="tank")
    node("plant.wet_tank_drain", [wet_pos[0], wet_pos[1] - wr - 0.04, wet_pos[2]], "drain-solenoid",
         mass_kg=0.4, mass_in_total=True, body_half_extent_m=[0.03, 0.03, 0.03], drive="electric-solenoid")
    edge("plant.wet_tank_to_drain", "plant.wet_tank", "plant.wet_tank_drain", "condensate-line",
         radius=0.004, circuit_identity="condensate", material="nylon-airline")
    chain.append(("plant.wet_tank", 0.012))

    # one real pipe run down the chain
    for (ident, r) in chain:
        edge(f"{ident}.inlet", prev, ident, "compressed-air-line", radius=r, circuit_identity=CIRCUIT_AIR,
             material="steel-pipe", medium_rate_state="compressed-air-flow-and-temperature")
        prev = ident

    # ---------------- the ultra-dry adsorption stage ----------------
    # Twin towers after the coalescing filter (and the filter is before
    # them for a reason: oil poisons desiccant permanently). This is
    # what takes the air past the refrigerated floor of about +3 C.
    if getattr(plant, "desiccant_dryer_fitted", False):
        for i, name in enumerate(("plant.desiccant_tower_a", "plant.desiccant_tower_b")):
            tc = (x0 - 0.32 - i * 0.16, y0 + 0.10, z0 - 5.6 * step)
            _drum(node, name, tc, [0.0, 1.0, 0.0], 0.070, 0.42, mass_kg=plant.desiccant_bed_kg + 6.0,
                  mass_in_total=True, material="aluminium-casting", filter_kind="desiccant-tower",
                  desiccant=plant.desiccant_kind, bed_kg=plant.desiccant_bed_kg,
                  fluid="compressed-air", fluid_volume_l=1.4, stage="desiccant")
        node("plant.dryer_changeover_valve", [x0 - 0.40, y0 - 0.02, z0 - 5.6 * step], "changeover-valve",
             mass_kg=1.1, mass_in_total=True, body_half_extent_m=[0.05, 0.04, 0.09], drive="electric-solenoid")
        for name in ("plant.desiccant_tower_a", "plant.desiccant_tower_b"):
            edge(f"{name}.inlet", "plant.dryer_changeover_valve", name, "compressed-air-line", radius=0.010,
                 circuit_identity=CIRCUIT_AIR, material="steel-pipe")
        node("plant.purge_muffler", [x0 - 0.46, y0 - 0.16, z0 - 5.6 * step], "exhaust-muffler", mass_kg=0.8,
             mass_in_total=True, body_half_extent_m=[0.05, 0.05, 0.05])
        edge("plant.dryer_purge", "plant.dryer_changeover_valve", "plant.purge_muffler", "compressed-air-line",
             radius=0.006, circuit_identity=CIRCUIT_AIR, material="steel-pipe", role="regeneration-purge")

    # ---------------- the manifolds ----------------
    man_pos = (x0 - 0.45, y0 - 0.02, z0 - 6 * step)
    node("plant.pneumatic_manifold", list(man_pos), "fluid-manifold", mass_kg=2.8, mass_in_total=True,
         material="aluminium-casting", ports=int(plant.pneumatic_manifold_ports),
         body_half_extent_m=[0.05, 0.045, 0.16], fluid="compressed-air", fluid_volume_l=0.5)
    edge("plant.wet_tank_to_manifold", "plant.wet_tank", "plant.pneumatic_manifold", "compressed-air-line",
         radius=0.012, circuit_identity=CIRCUIT_AIR, material="steel-pipe")
    # the manifold feeds the isolated reserve set the pneumatic block built
    for target in ("pneumatic_protection_valve", "pneumatic_reserve_tank"):
        if target in node_by_id:
            edge(f"plant.manifold_to_{target}", "plant.pneumatic_manifold", target, "compressed-air-line",
                 radius=0.010, circuit_identity=CIRCUIT_AIR, material="steel-pipe")
            break

    # ---------------- the hydraulic side ----------------
    if plant.hydraulic_fitted:
        hv = max(10.0, plant.hydraulic_tank_capacity_l) / 1000.0
        hr = (hv / (3.0 * math.pi)) ** (1.0 / 3.0)
        hl = 6.0 * hr
        h_pos = (x0 - 0.25, y0 + 0.30, z0 - 3.0 * step)
        _drum(node, "plant.hydraulic_tank", h_pos, ax, hr, hl, mass_kg=max(8.0, hv * 900.0 * 0.12),
              mass_in_total=True, material="pressed-steel", fluid="hydraulic-oil",
              fluid_volume_l=plant.hydraulic_tank_capacity_l, tank_kind="hydraulic-reservoir")
        _drum(node, "plant.hydraulic_chiller", (h_pos[0], h_pos[1] - hr - 0.06, h_pos[2]), ax, 0.05, hl * 0.7,
              mass_kg=3.5, mass_in_total=True, material="copper", exchanger_kind="refrigerant-to-oil-chiller",
              fluid="refrigerant", fluid_volume_l=0.3)
        edge("plant.hydraulic_chiller_coil", "plant.hydraulic_chiller", "plant.hydraulic_tank",
             "heat-exchange-path", radius=0.006, heat_share_frac=0.0, medium_rate_state="rejected-heat-flow-w")
        node("plant.hydronic_manifold", [h_pos[0] - 0.30, h_pos[1], h_pos[2]], "fluid-manifold", mass_kg=3.2,
             mass_in_total=True, material="steel", ports=int(plant.hydronic_manifold_ports),
             body_half_extent_m=[0.05, 0.05, 0.16], fluid="hydraulic-oil", fluid_volume_l=0.8)
        edge("plant.hydraulic_tank_to_manifold", "plant.hydraulic_tank", "plant.hydronic_manifold",
             "hydraulic-line", radius=0.010, circuit_identity=CIRCUIT_HYDRAULIC, material="steel-pipe",
             medium_rate_state="hydraulic-flow-and-temperature-and-pressure")

        # --- the reservoir's gas inlet: a three-way selector, its
        # sources, and the emergency vacuum break behind them ---
        gas_pos = (h_pos[0] + hl * 0.5 + 0.10, h_pos[1] + hr + 0.05, h_pos[2])
        node("plant.blanket_manifold", list(gas_pos), "three-way-selector", mass_kg=1.4, mass_in_total=True,
             material="brass", body_half_extent_m=[0.05, 0.04, 0.05],
             sources=["plant-air", "nitrogen", "vacuum-break"])
        edge("plant.blanket_to_tank", "plant.blanket_manifold", "plant.hydraulic_tank", "low-pressure-air-line",
             radius=0.005, circuit_identity="blanket", material="nylon-airline")
        edge("plant.blanket_from_air", "plant.pneumatic_manifold", "plant.blanket_manifold",
             "low-pressure-air-line", radius=0.005, circuit_identity="blanket", material="nylon-airline")
        node("plant.blanket_regulator", [gas_pos[0] + 0.10, gas_pos[1], gas_pos[2]], "pressure-regulator",
             mass_kg=0.5, mass_in_total=True, body_half_extent_m=[0.03, 0.04, 0.03],
             setpoint_pa=151_325.0)
        edge("plant.blanket_regulator_feed", "plant.blanket_regulator", "plant.blanket_manifold",
             "low-pressure-air-line", radius=0.004, circuit_identity="blanket", material="nylon-airline")
        node("plant.blanket_relief", [gas_pos[0], gas_pos[1] + 0.08, gas_pos[2]], "relief-valve", mass_kg=0.3,
             mass_in_total=True, body_half_extent_m=[0.025, 0.03, 0.025], relief_pressure_pa=201_325.0)
        edge("plant.blanket_relief_tap", "plant.hydraulic_tank", "plant.blanket_relief", "low-pressure-air-line",
             radius=0.004, circuit_identity="blanket", material="nylon-airline")
        if plant.hydraulic_nitrogen_backup:
            n2_pos = (gas_pos[0] + 0.26, gas_pos[1] - 0.10, gas_pos[2] + 0.14)
            _drum(node, "plant.nitrogen_bottle", n2_pos, [0.0, 1.0, 0.0], 0.075, 0.62,
                  kind="high-pressure-canister", mass_kg=14.0, mass_in_total=True, material="steel",
                  working_pressure_pa=20_000_000.0, capacity_kg=1.2, tank_kind="nitrogen-bottle",
                  fluid="nitrogen", fluid_volume_l=10.0)
            node("plant.nitrogen_regulator", [n2_pos[0], n2_pos[1] + 0.34, n2_pos[2]], "pressure-regulator",
                 mass_kg=0.6, mass_in_total=True, body_half_extent_m=[0.035, 0.04, 0.035], setpoint_pa=151_325.0)
            edge("plant.nitrogen_to_regulator", "plant.nitrogen_bottle", "plant.nitrogen_regulator",
                 "high-pressure-gas-line", radius=0.003, circuit_identity="nitrogen", material="steel-pipe")
            edge("plant.nitrogen_to_manifold", "plant.nitrogen_regulator", "plant.blanket_manifold",
                 "low-pressure-air-line", radius=0.004, circuit_identity="blanket", material="nylon-airline")
        node("plant.vacuum_break", [gas_pos[0] - 0.10, gas_pos[1] + 0.04, gas_pos[2]], "vacuum-break-valve",
             mass_kg=0.35, mass_in_total=True, body_half_extent_m=[0.03, 0.05, 0.03],
             crack_pressure_pa=98_325.0, desiccant="silica-gel", cartridge_kg=0.4)
        edge("plant.vacuum_break_to_manifold", "plant.vacuum_break", "plant.blanket_manifold",
             "low-pressure-air-line", radius=0.005, circuit_identity="blanket", material="nylon-airline")

        # --- warming the oil: the gentle way and the crude way ---
        if plant.hydraulic_tank_heater:
            node("plant.tank_heater", [h_pos[0] - hl * 0.3, h_pos[1] - hr * 0.7, h_pos[2]],
                 "immersion-heater", mass_kg=2.0, mass_in_total=True,
                 body_half_extent_m=[hl * 0.3, 0.03, 0.03], rated_w=1500.0,
                 element_area_cm2=1200.0, drive="electric-element")
            edge("plant.tank_heater_power", "plant.control_panel", "plant.tank_heater",
                 "insulated-copper-wire", radius=0.003, palette="active", circuit="accessory-main",
                 maximum_current_a=70.0, routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")
        _drum(node, "plant.oil_coolant_exchanger", (h_pos[0] - hl * 0.6, h_pos[1] - hr - 0.14, h_pos[2]),
              ax, 0.045, hl * 0.5, mass_kg=3.0, mass_in_total=True, material="copper",
              exchanger_kind="oil-to-coolant", fluid="engine-oil", fluid_volume_l=0.6)
        edge("plant.oil_exchanger_oil_side", "plant.hydraulic_tank", "plant.oil_coolant_exchanger",
             "hydraulic-line", radius=0.008, circuit_identity=CIRCUIT_HYDRAULIC, material="steel-pipe")
        if "powertrain.coolant_thermostat" in node_by_id:
            edge("plant.oil_exchanger_coolant_side", "powertrain.coolant_thermostat",
                 "plant.oil_coolant_exchanger", "coolant-line", radius=0.008, circuit_identity="coolant",
                 material="rubber-hose", medium_rate_state="coolant-flow-and-temperature")
        _drum(node, "plant.oil_cooler", (h_pos[0], h_pos[1] + hr + 0.18, h_pos[2]), ax, 0.02, hl,
              mass_kg=5.0, mass_in_total=True, material="aluminium-casting",
              exchanger_kind="oil-to-air-cooler", fluid="engine-oil", fluid_volume_l=1.5)
        edge("plant.oil_cooler_feed", "plant.hydronic_manifold", "plant.oil_cooler", "hydraulic-line",
             radius=0.010, circuit_identity=CIRCUIT_HYDRAULIC, material="steel-pipe")

        # --- the actuator in the bay: a hydraulic fan drive ---
        if plant.hydraulic_fan_drive:
            fan = node_by_id.get("powertrain.cooling_fan")
            fpos = (np.asarray(fan["reference_position"], dtype=float) if fan is not None
                    else np.array([h_pos[0] - 0.5, h_pos[1], h_pos[2]]))
            node("plant.fan_drive_motor", [float(fpos[0]) - 0.10, float(fpos[1]), float(fpos[2])],
                 "hydraulic-motor", mass_kg=6.5, mass_in_total=True, material="cast-iron-casting",
                 displacement_cc_rev=30.0, reversible=True, body_half_extent_m=[0.06, 0.07, 0.07])
            edge("plant.fan_drive_supply", "plant.hydronic_manifold", "plant.fan_drive_motor",
                 "hydraulic-line", radius=0.008, circuit_identity=CIRCUIT_HYDRAULIC, material="steel-pipe",
                 medium_rate_state="hydraulic-flow-and-temperature-and-pressure")
            edge("plant.fan_drive_return", "plant.fan_drive_motor", "plant.hydraulic_tank",
                 "hydraulic-line", radius=0.010, circuit_identity=CIRCUIT_HYDRAULIC, material="steel-pipe")
            if fan is not None:
                edge("plant.fan_drive_to_fan", "plant.fan_drive_motor", "powertrain.cooling_fan",
                     "rigid-keyed-hub", radius=0.010)

    # ---------------- the refrigerant loop ----------------
    ac = node_by_id.get("ac_compressor")
    cond = node_by_id.get("powertrain.condenser")
    if ac is not None:
        ac_pos = np.asarray(ac["reference_position"], dtype=float)
        drier_pos = (float(ac_pos[0]) - 0.12, float(ac_pos[1]) + 0.10, float(ac_pos[2]) - 0.10)
        _drum(node, "plant.receiver_drier", drier_pos, [0.0, 1.0, 0.0], 0.045, 0.22, mass_kg=1.6,
              mass_in_total=True, material="aluminium-casting", fluid="refrigerant", fluid_volume_l=0.5,
              vessel_kind="receiver-drier")
        if cond is not None:
            edge("plant.compressor_to_condenser", "ac_compressor", "powertrain.condenser", "refrigerant-line",
                 radius=0.007, circuit_identity=CIRCUIT_REFRIGERANT, material="aluminium-tube",
                 medium_rate_state="refrigerant-flow-and-pressure", side="discharge")
            edge("plant.condenser_to_drier", "powertrain.condenser", "plant.receiver_drier", "refrigerant-line",
                 radius=0.006, circuit_identity=CIRCUIT_REFRIGERANT, material="aluminium-tube", side="liquid")
        # one expansion valve and one suction return per evaporator
        evaporators = [("plant.air_chiller", "air-chiller")]
        if plant.hydraulic_fitted:
            evaporators.append(("plant.hydraulic_chiller", "oil-chiller"))
        if engine.accessories.air_conditioning:
            cabin = "plant.cabin_evaporator"
            node(cabin, [float(ac_pos[0]) - 0.35, float(ac_pos[1]) + 0.22, float(ac_pos[2])], "engine-block-component",
                 mass_kg=2.2, mass_in_total=True, chassis_side=True, material="aluminium-casting",
                 exchanger_kind="cabin-evaporator", fluid="refrigerant", fluid_volume_l=0.4)
            evaporators.append((cabin, "cabin"))
        for ident, role in evaporators:
            if ident not in node_by_id and ident not in {n["identity"] for n in nodes}:
                continue
            txv = f"{ident}.expansion_valve"
            pos = np.asarray(next(n["reference_position"] for n in nodes if n["identity"] == ident), dtype=float)
            node(txv, [float(pos[0]) + 0.06, float(pos[1]) + 0.10, float(pos[2])], "expansion-valve",
                 mass_kg=0.35, mass_in_total=True, body_half_extent_m=[0.028, 0.035, 0.028], evaporator_role=role)
            edge(f"plant.drier_to_{role}_txv", "plant.receiver_drier", txv, "refrigerant-line", radius=0.004,
                 circuit_identity=CIRCUIT_REFRIGERANT, material="aluminium-tube", side="liquid")
            edge(f"plant.{role}_txv_to_evaporator", txv, ident, "refrigerant-line", radius=0.005,
                 circuit_identity=CIRCUIT_REFRIGERANT, material="aluminium-tube", side="evaporator")
            edge(f"plant.{role}_suction_return", ident, "ac_compressor", "refrigerant-line", radius=0.009,
                 circuit_identity=CIRCUIT_REFRIGERANT, material="aluminium-tube", side="suction")
        # the switches that actually decide the clutch
        node("plant.lp_switch", [drier_pos[0] - 0.05, drier_pos[1] + 0.12, drier_pos[2]], "pressure-switch",
             mass_kg=0.12, mass_in_total=True, body_half_extent_m=[0.02, 0.02, 0.02], switch_side="low")
        node("plant.hp_switch", [drier_pos[0] - 0.05, drier_pos[1] + 0.12, drier_pos[2] - 0.06], "pressure-switch",
             mass_kg=0.12, mass_in_total=True, body_half_extent_m=[0.02, 0.02, 0.02], switch_side="high")
        for sw in ("plant.lp_switch", "plant.hp_switch"):
            edge(f"{sw}_tap", "plant.receiver_drier", sw, "refrigerant-line", radius=0.002,
                 circuit_identity=CIRCUIT_REFRIGERANT, material="aluminium-tube")

    # ---------------- controls and power ----------------
    panel_pos = (x0 - 0.55, y0 + 0.24, z0 - 3.0 * step)
    node("plant.control_panel", list(panel_pos), "control-panel", mass_kg=4.0, mass_in_total=True,
         material="steel", body_half_extent_m=[0.06, 0.16, 0.20],
         controls=["main-chiller-switch", "air-dryer-enable", "reheater-enable", "separator-drain",
                   "wet-tank-drain", "hydraulic-chiller-enable", "reserve-isolation"])
    for target in ("plant.aftercooler_fan", "plant.reheater", "plant.separator_drain", "plant.wet_tank_drain",
                   "plant.lp_switch", "plant.hp_switch", "plant.hydraulic_chiller"):
        if any(n["identity"] == target for n in nodes):
            edge(f"plant.control_to_{target.split('.')[-1]}", "plant.control_panel", target,
                 "insulated-copper-wire", radius=0.002, palette="active", circuit="plant-control",
                 maximum_current_a=15.0, routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")

    if plant.accessory_bank_fitted:
        # a second battery bank, behind an isolator, so the plant's fans,
        # heater and solenoids cannot flatten what has to crank the engine
        bank_pos = (x0 - 0.55, y0 - 0.22, z0 - 5.0 * step)
        node("electrical.accessory_battery", list(bank_pos), "battery-bank", mass_kg=plant.accessory_bank_ah * 0.35,
             mass_in_total=True, material="pressed-steel", capacity_ah=plant.accessory_bank_ah,
             nominal_voltage_v=plant.accessory_bank_voltage_v, chemistry=plant.accessory_bank_chemistry,
             body_half_extent_m=[0.13, 0.11, 0.17])
        node("electrical.charge_isolator", [bank_pos[0] + 0.18, bank_pos[1] + 0.10, bank_pos[2]],
             "charge-isolator", mass_kg=0.6, mass_in_total=True, isolator_kind=plant.isolator_kind,
             body_half_extent_m=[0.05, 0.04, 0.05])
        edge("electrical.wire.isolator_from_start_bank", "electrical.battery", "electrical.charge_isolator",
             "insulated-copper-wire", radius=0.006, palette="active", circuit="battery-main",
             maximum_current_a=120.0, routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")
        edge("electrical.wire.isolator_to_accessory_bank", "electrical.charge_isolator", "electrical.accessory_battery",
             "insulated-copper-wire", radius=0.006, palette="active", circuit="accessory-main",
             maximum_current_a=120.0, routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")
        edge("electrical.wire.accessory_bank_to_panel", "electrical.accessory_battery", "plant.control_panel",
             "insulated-copper-wire", radius=0.004, palette="active", circuit="accessory-main",
             maximum_current_a=60.0, routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom")
