"""The engine as ONE mesh: vertices, triangles, normals, and a material
per triangle -- the dressed, colour-coded design lifted into the form
a renderer, an exporter and a game all consume. Static parts (castings,
covers, headers, dressing) are built once per engine; MOVING parts
(pistons, rods, crank, valves, springs, followers, rotors, crossheads)
are rebuilt from the live crank angle.

Materials mirror the spectral analyzer's Phong record (ambient,
spec_strength, shininess, inner colour) over a PBR base (albedo,
roughness, opacity): the same fields base_material.frag.glsl reads
from its SSBOs, so this mesh's material table can be uploaded to
that renderer unchanged, and the software Phong in mesh_visualizer
evaluates the same lighting on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

import numpy as np

from vehicle_mesh import (build_agt1500_moving_parts,
                          build_drivetrain_solid_parts,
                          build_throttle_parts)


@dataclass(frozen=True)
class Material:
    name: str
    label: str
    albedo: tuple            # linear-ish sRGB 0..1
    opacity: float = 1.0
    roughness: float = 0.6
    ambient: float = 0.22    # phong[0]
    spec_strength: float = 0.25   # phong[1]
    shininess: float = 24.0  # phong[2]
    metallic: float = 0.0

    def pbr_record(self) -> list[float]:
        """PBRBaseRecord, 16 floats (albedo, roughness, metallic,
        transmission, ior, opacity, emission, reserved)."""
        return [*self.albedo, self.roughness, self.metallic, 0.0, 1.5, self.opacity, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0]

    def phong_record(self) -> list[float]:
        """PhongRecord, 8 floats (ambient, spec_strength, shininess,
        reserved, inner colour rgb, pad)."""
        return [self.ambient, self.spec_strength, self.shininess, 0.0, *self.albedo, 0.0]


def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# (name-substrings, material) -- first match wins; the colour system of the design renders
MATERIAL_RULES = [
    (("_valve_head", "_valve_stem"), Material("valve", "valves", _rgb("#ece4d0"), 1.0, 0.25, 0.2, 0.7, 60.0, 0.8)),
    (("_spring",), Material("spring", "valve springs", _rgb("#e0b030"), 1.0, 0.35, 0.2, 0.5, 40.0, 0.6)),
    (("_retainer", "_bucket", "_rocker", "_pushrod"), Material("follower", "followers / rockers / pushrods", _rgb("#c08040"), 1.0, 0.4, 0.2, 0.4, 30.0, 0.5)),
    (("camshaft", "cam_lobe", "cam_in_block"), Material("cam", "camshaft + lobes", _rgb("#d09040"), 1.0, 0.3, 0.2, 0.6, 50.0, 0.7)),
    (("_piston",), Material("piston", "pistons", _rgb("#c9d3da"), 1.0, 0.3, 0.22, 0.6, 50.0, 0.7)),
    # combustion_kernel.py: the burn itself and its residue. Both are
    # pure live-emission/opacity rows (engine_gl_view's material layer
    # overwrites them per cylinder per tick); the base albedo only
    # matters for whatever ambient reaches an unlit puff of smoke
    (("_flame_kernel",), Material("flame", "combustion (live flame kernel)", _rgb("#ffb060"), 0.85, 0.9, 0.0, 0.0, 1.0)),
    (("_smoke_puff",), Material("smoke", "exhaust residue (soot / oil smoke / ash)", _rgb("#8a8a8a"), 0.35, 1.0, 0.3, 0.0, 1.0)),
    (("_con_rod", "_piston_rod", "_crosshead"), Material("rod", "con rods", _rgb("#b8892a"), 1.0, 0.35, 0.2, 0.5, 40.0, 0.6)),
    (("crank_", "crankpin", "flywheel", "_pinion", "_freewheel", "_eccentric"), Material("crank", "crankshaft + flywheel", _rgb("#23262b"), 1.0, 0.45, 0.2, 0.45, 32.0, 0.5)),
    (("_rotor",), Material("rotor", "rotor", _rgb("#e0c060"), 1.0, 0.35, 0.2, 0.5, 40.0, 0.6)),
    (("_intake_port", "intake_port_", "transfer_port", "scavenge", "_air_port", "_gas_port"), Material("intake_port", "intake ports", _rgb("#4c8dff"), 0.95, 0.5, 0.25, 0.3, 20.0)),
    (("_exhaust_port", "exhaust_port_", "_head_exhaust", "_crank_exhaust"), Material("exhaust_port", "exhaust ports", _rgb("#ff5a3c"), 0.95, 0.5, 0.25, 0.3, 20.0)),
    (("spark_plug", "glow_plug", "_flame_port", "leading_plug", "trailing_plug"), Material("igniter", "plugs / igniters", _rgb("#ffd23c"), 1.0, 0.4, 0.25, 0.5, 30.0)),
    (("_injector",), Material("injector", "injector bosses", _rgb("#3ccf6a"), 1.0, 0.4, 0.25, 0.4, 30.0)),
    (("_admission", "_drain_cock", "_lubricator", "_rod_gland"), Material("expander_port", "expander passages / cocks", _rgb("#b070ff"), 0.95, 0.5, 0.25, 0.3, 20.0)),
    (("edge_powertrain_cylinder", "edge_powertrain_exhaust", "node_powertrain_exhaust_collector"), Material("exhaust", "exhaust primaries / collector", _rgb("#e06040"), 0.6, 0.5, 0.25, 0.35, 24.0)),
    # Air filter, throttle body, and plenum/runners each get their own
    # distinct color -- these used to share one "intake" material, which
    # made the real, genuinely different boxes in a real distributor
    # chain (air_filter -> throttle_body(s) -> plenum chamber(s) ->
    # runners -> ports) hard to tell apart by eye even though the real
    # graph topology connecting them was always correct.
    (("node_powertrain_air_filter", "snorkel_inlet"), Material("air_filter", "air filter / cleaner / remote box / snorkel inlet", _rgb("#c8102e"), 0.85, 0.7, 0.28, 0.25, 18.0)),   # the red cotton-gauze cleaner look
    (("throttle_plate", "throttle_linkage"), Material("throttle_plate", "throttle plate + linkage", _rgb("#e8e8ec"), 1.0, 0.3, 0.2, 0.6, 45.0, 0.6)),
    (("node_powertrain_throttle_body", "throttle_body"), Material("throttle_body", "throttle body / barrel(s)", _rgb("#d4a017"), 0.85, 0.5, 0.25, 0.35, 26.0)),
    (("runner", "node_powertrain_intake_plenum", "stack"), Material("intake", "intake runners / plenum", _rgb("#8fb0d8"), 0.5, 0.5, 0.25, 0.3, 24.0)),
    (("fuel_rail", "rail_feed", "fuel_filter", "edge_fuel"), Material("fuel", "fuel rail + filter", _rgb("#3ccf6a"), 0.75, 0.5, 0.25, 0.3, 24.0)),
    (("lead", "distributor", "ignition_coil", "coil_pack", "leading_coil", "trailing_coil"), Material("ignition", "distributor / coils / plug leads", _rgb("#d21f26"), 0.95, 0.45, 0.28, 0.35, 26.0)),   # the red aftermarket-ignition look
    (("magneto", "glow_plug_bus"), Material("magneto", "magneto / glow-plug bus", _rgb("#202020"), 0.9, 0.7, 0.3, 0.15, 12.0)),
    (("oil_filter", "oil_reserve", "scavenge", "oil_bath", "oil_pump", "edge_powertrain_oil", "edge_powertrain_pan", "edge_powertrain_trough", "splash"), Material("lube", "lubrication", _rgb("#8a6a30"), 0.75, 0.6, 0.25, 0.25, 16.0)),
    (("compression_brake",), Material("compression_brake", "compression-release brake (housing + slave pistons)", _rgb("#3a4a3f"), 1.0, 0.45, 0.25, 0.4, 28.0, 0.4)),
    # THE COLD SIDE, and it should look cold. Everything on the
    # refrigerant loop and everything it chills -- the chillers, the
    # condenser, the receiver-drier, the expansion valves, the AC
    # compressor itself -- is one icy pale blue, so at a glance you can
    # see how far the cold actually reaches into the skid. This rule
    # sits ABOVE the pneumatic rule on purpose: `plant.air_chiller` is
    # both a refrigeration part and an air part, and the cold side is
    # the more informative thing to say about it.
    (("chiller", "condenser", "receiver_drier", "expansion_valve", "evaporator",
      "refrigerant", "ac_compressor", "aftercooler", "_lp_switch", "_hp_switch"),
     Material("refrigeration", "refrigeration / chilled coolers", _rgb("#9fe0f2"), 0.9, 0.25, 0.22, 0.6, 70.0, 0.35)),
    # THE HYDRAULIC SIDE, in the dark red its own fluid is dyed -- the
    # tank, its heater and breather, the pilot and blanket hardware, and
    # the fan-drive motor that runs off it. Deliberately in the same
    # family as the hydraulic and ATF leak colours below, because it is
    # the same fluid.
    (("hydraulic_", "hydronic_", "fan_drive_motor", "blanket_", "vacuum_break",
      "nitrogen_bottle", "nitrogen_regulator"),
     Material("hydraulic", "hydraulic system (tank, pilot, blanket, motors)", _rgb("#8c2f3f"),
              0.95, 0.5, 0.25, 0.4, 30.0, 0.35)),
    # THE AIR SIDE, in a deep saturated blue -- compressed air is the
    # other utility running through this machine and it should never be
    # mistaken for the cold side or for a coolant line. Every real part
    # of the train belongs to it: treatment vessels, driers, manifolds,
    # tanks, brake reservoirs and valves.
    (("pneumatic_", "wet_tank", "water_separator", "coalescing_filter", "particulate_filter",
      "desiccant_tower", "dryer_changeover", "purge_muffler", "pneumatic_manifold",
      "air_reservoir", "compressed_air"),
     Material("pneumatic", "compressed-air system (compressor, treatment, tanks, valves, lines)",
              _rgb("#1b3fb8"), 0.95, 0.4, 0.25, 0.45, 34.0, 0.35)),
    (("port_",), Material("casting_port", "casting ports (fill, galleries, coolant, breather)", _rgb("#404040"), 0.95, 0.7, 0.3, 0.15, 12.0)),
    # The four "look-through" shell materials (head/cover/cylinder/case)
    # are how covers_off shows the real internals through the real
    # casting -- see-through only reads as "glass", not "faded plastic",
    # if it actually behaves like glass: low roughness (a tight,
    # near-mirror highlight, not a soft plastic sheen) and a strong
    # spec_strength/shininess so the Schlick-Fresnel rim the shader
    # already computes from the material's ior (base_material.frag.glsl,
    # hardcoded 1.5 -- real glass) actually shows up at grazing angles,
    # instead of being swamped by a flat, low-gloss alpha blend.
    (("head_casting", "cam_box"), Material("head", "head castings", _rgb("#98a4b4"), 0.24, 0.08, 0.2, 0.8, 130.0)),
    (("valve_cover", "valley_cover", "case_top_cover"), Material("cover", "covers", _rgb("#606a78"), 0.2, 0.06, 0.2, 0.85, 140.0)),
    (("_bore", "_water_jacket", "_fin_", "_head", "chamber_roof", "_crank_cover", "_hopper", "_rotor_housing", "_side_plate", "_valve_chest"), Material("cylinder", "cylinders / jackets / fins", _rgb("#57626e"), 0.22, 0.08, 0.2, 0.8, 130.0)),
    (("crankcase", "sump", "front_cover", "rear_main", "bedplate", "main_pedestal", "frame_plate", "guide_bar", "rear_accessory"), Material("case", "crankcase / sump / frame", _rgb("#34363a"), 0.2, 0.1, 0.2, 0.75, 120.0)),
    (("node_mount_",), Material("mount", "engine mounts / isolators", _rgb("#8a3324"), 0.3, 0.6, 0.3, 0.3, 18.0)),
    # hole_emitters.py: what is passing through the damage holes, drawn
    # as droplet streaks along each emitter's real jet (engine_gl_view.
    # set_emitter_particles) -- one row per fluid so oil reads amber,
    # coolant green, fuel pale, a gas plume white and thin
    (("leak_engine-oil",), Material("leak_oil", "leaking engine oil", _rgb("#d19a2a"), 0.85, 0.15, 0.2, 0.7, 90.0)),
    # Hydraulic oil and ATF are both dyed red in the real world, and
    # telling them apart by colour under a machine is genuinely hard --
    # that conflation is real and worth keeping. They are still two
    # different fluids with different viscosities (fluids.py), so they
    # get two neighbouring reds rather than one shared row: a leak you
    # have to look twice at, not a leak the simulation has decided for
    # you.
    (("leak_hydraulic-oil",), Material("leak_hydraulic", "leaking hydraulic oil", _rgb("#d84f66"), 0.85, 0.15, 0.2, 0.7, 90.0)),
    (("leak_transmission-fluid",), Material("leak_atf", "leaking transmission fluid", _rgb("#f2607f"), 0.85, 0.15, 0.2, 0.7, 90.0)),
    # gear oil is the dark, almost brown one -- an EP package in a heavy
    # base, and it looks it
    (("leak_gear-oil",), Material("leak_gear_oil", "leaking gear oil", _rgb("#6b4a1e"), 0.9, 0.2, 0.2, 0.6, 70.0)),
    (("leak_coolant",), Material("leak_coolant", "leaking coolant", _rgb("#3fd9a0"), 0.8, 0.15, 0.2, 0.7, 90.0)),
    (("leak_fuel", "leak_water"), Material("leak_fuel", "leaking fuel / water", _rgb("#d8e8ff"), 0.6, 0.1, 0.2, 0.8, 120.0)),
    (("leak_gas",), Material("leak_gas", "venting gas", _rgb("#e8e8e8"), 0.25, 0.9, 0.3, 0.0, 1.0)),
    (("leak_solid",), Material("burst_fragment", "burst fragments (casing shards)", _rgb("#9aa0a8"), 1.0, 0.35, 0.22, 0.6, 50.0, 0.6)),
    # ordnance.py: the detonation fireball, drawn as its own expanding
    # cloud of luminous products (emissive, like the combustion kernel)
    (("leak_flame",), Material("fireball", "detonation fireball", _rgb("#ffd08a"), 0.9, 1.0, 0.0, 0.0, 1.0)),
    (("leak_soot",), Material("fireball_soot", "detonation smoke", _rgb("#4a4038"), 0.45, 0.9, 0.25, 0.0, 1.0)),
    # universal bolt-ons (engine_parts.py)
    (("_pulley", "belt_tensioner", "_sprocket", "timing_chain_run"), Material("pulley", "pulleys / tensioner / sprockets", _rgb("#3a3d44"), 1.0, 0.4, 0.22, 0.5, 34.0, 0.7)),
    (("harmonic_balancer",), Material("damper", "harmonic damper", _rgb("#2c2f36"), 1.0, 0.45, 0.2, 0.45, 32.0, 0.5)),
    (("starter_motor", "recoil_starter", "crank_nose_fitting"), Material("starter", "starter hardware", _rgb("#4a4f58"), 1.0, 0.5, 0.25, 0.35, 24.0, 0.4)),
    (("expansion_bottle", "heater_core"), Material("coolant_gear", "coolant bottle / heater core", _rgb("#e8e6d8"), 0.85, 0.5, 0.3, 0.2, 16.0)),
    (("egr_valve",), Material("egr", "EGR valve", _rgb("#8a5a3a"), 1.0, 0.55, 0.25, 0.3, 20.0)),
    (("pcv_valve",), Material("pcv", "PCV valve", _rgb("#202020"), 1.0, 0.6, 0.3, 0.2, 14.0)),
    (("timing_cover", "bellhousing"), Material("housing", "timing cover / bellhousing", _rgb("#3f4349"), 0.3, 0.6, 0.25, 0.25, 16.0)),
    # forced induction (engine_parts.py)
    (("blower_case", "blower_manifold", "blower_snout"), Material("blower", "blower case / manifold / snout", _rgb("#b9bec6"), 1.0, 0.3, 0.22, 0.6, 48.0, 0.85)),
    (("blower_burst_panel",), Material("burst_panel", "burst panel / restraint", _rgb("#2a2a2e"), 1.0, 0.6, 0.25, 0.25, 16.0)),
    (("blower_pulley",), Material("blower_pulley", "blower drive pulley", _rgb("#3a3d44"), 1.0, 0.4, 0.22, 0.5, 34.0, 0.7)),
    (("compressor_housing", "blow_off_valve", "charge_cooler", "charge_pipe"), Material("charge_side", "compressor / charge cooler / BOV", _rgb("#c9ced6"), 0.95, 0.35, 0.24, 0.55, 40.0, 0.8)),
    (("turbine_housing", "wastegate", "downpipe", "up_pipe"), Material("hot_side", "turbine housing / wastegate / downpipe", _rgb("#8c5a3c"), 0.9, 0.55, 0.25, 0.3, 20.0)),
    (("barrel_valve",), Material("barrel_valve", "barrel valve", _rgb("#3ccf6a"), 1.0, 0.4, 0.25, 0.4, 30.0)),
    # aircraft / industrial / air-cooled families (engine_parts.py)
    (("prop_reduction_gearbox", "prop_shaft", "prop_hub"), Material("prop_drive", "propeller reduction / shaft / hub", _rgb("#5a6068"), 1.0, 0.35, 0.22, 0.5, 36.0, 0.75)),
    (("propeller_governor", "flyball_governor", "governor_latch"), Material("governor", "governors + linkage", _rgb("#8a7a2a"), 1.0, 0.4, 0.25, 0.45, 30.0, 0.5)),
    (("injection_pump",), Material("injection_pump", "diesel injection pump", _rgb("#2f6b3a"), 1.0, 0.5, 0.25, 0.35, 24.0)),
    (("oil_cooler",), Material("oil_cooler", "oil cooler", _rgb("#8a6a30"), 0.85, 0.55, 0.25, 0.3, 20.0)),
    (("cooling_fan_housing", "cooling_tin_", "_baffle", "cowl_flap_"), Material("cooling_tin", "fan housing / tins / baffles / cowl flaps", _rgb("#4b545e"), 0.55, 0.6, 0.25, 0.3, 18.0)),
    # non-piston kinds (engine_parts.py)
    (("compressor_impeller", "compressor_diffuser", "inlet_plenum", "turbine_ngv", "turbine_wheel", "power_turbine", "exhaust_duct", "bleed_valve"), Material("gas_path", "turbine gas path", _rgb("#c9cfd8"), 0.9, 0.3, 0.24, 0.6, 46.0, 0.85)),
    (("node_powertrain_combustor", "fuel_control_unit", "ignition_exciter", "accessory_gearbox", "starter_generator", "power_turbine_reduction"), Material("turbine_accessory", "combustor / FCU / accessory gearbox", _rgb("#6e5a3a"), 1.0, 0.5, 0.25, 0.35, 24.0)),
    (("motor_housing", "stator", "rotor_pack", "resolver", "planetary_gearhead", "output_flange", "reduction_gearset", "drive_unit_differential"), Material("drive_unit", "motor / gearset", _rgb("#5b6470"), 1.0, 0.4, 0.22, 0.5, 34.0, 0.7)),
    (("node_powertrain_inverter", "inverter_coolant_plate", "hv_contactor_box", "dc_dc_converter", "phase_busbar"), Material("power_electronics", "inverter / HV switchgear", _rgb("#e0641e"), 1.0, 0.45, 0.3, 0.3, 22.0)),
    (("boiler_barrel", "firebox", "steam_dome", "air_reservoir_", "reheater"), Material("boiler", "boiler / air reservoirs", _rgb("#3a3f46"), 1.0, 0.55, 0.22, 0.3, 18.0)),
    (("regulator_valve", "safety_valve", "water_gauge", "chimney", "blast_nozzle", "feedwater_injector", "water_tank", "charging_valve", "check_valve", "pressure_gauge", "reverser_lever", "valve_gear_eccentric"), Material("plant_fittings", "boiler / reservoir fittings + valve gear", _rgb("#b08a3c"), 1.0, 0.4, 0.25, 0.5, 36.0, 0.7)),
    (("_rack", "slide_valve", "valve_eccentric", "pilot_burner", "captive_ball_governor"), Material("otto_mechanism", "rack / slide valve / burner / governor", _rgb("#8a7a2a"), 1.0, 0.4, 0.25, 0.45, 30.0, 0.5)),
    (("small_engine_muffler", "primer_bulb", "water_hopper", "igniter_trip_lever"), Material("small_engine_kit", "small-engine kit", _rgb("#565c66"), 1.0, 0.55, 0.25, 0.3, 20.0)),
    # transverse driveline (drivetrain_graph.py)
    (("node_powertrain_transaxle", "final_drive", "node_powertrain_differential"), Material("transaxle", "transaxle / final drive / differential", _rgb("#6b7280"), 1.0, 0.45, 0.22, 0.45, 30.0, 0.6)),
    (("halfshaft",), Material("halfshaft", "halfshafts", _rgb("#2c2f36"), 1.0, 0.4, 0.2, 0.5, 34.0, 0.6)),
]
# Materials chosen by what a part SAYS it is made of, for graphs whose
# part names the keyword table above was never written around -- which
# is every machine (machines.py). These are opaque on purpose: a turret
# is not a see-through object, and falling through to the translucent
# default rendered one as a set of grey ghosts.
DECLARED_MATERIALS: dict[str, Material] = {
    # Armour is the machine's equivalent of the engine's valve cover:
    # you want to see the mechanism working inside it, so it is drawn
    # the way the see-through shells are (low alpha, tight highlight)
    # rather than as an opaque box that hides everything this view
    # exists to show.
    "armour-plate": Material("armour", "armour plate", _rgb("#8d9a8a"), 0.30, 0.10, 0.22, 0.75, 110.0),
    "gun-steel": Material("gun_steel", "gun steel", _rgb("#767d86"), 1.0, 0.38, 0.2, 0.5, 40.0, 0.6),
    "hardened-steel": Material("hardened", "hardened steel", _rgb("#b6bdc5"), 1.0, 0.3, 0.22, 0.6, 52.0, 0.75),
    "steel-plate": Material("plate", "steel plate", _rgb("#9aa2ab"), 1.0, 0.5, 0.24, 0.4, 30.0, 0.5),
    "pressed-steel": Material("pressed", "pressed steel", _rgb("#c2c8d0"), 1.0, 0.45, 0.24, 0.45, 34.0, 0.55),
    "cast-iron": Material("cast_iron", "cast iron", _rgb("#7d828a"), 1.0, 0.7, 0.24, 0.25, 16.0, 0.35),
    "steel-pipe": Material("pipe", "steel pipe", _rgb("#9aa1a9"), 1.0, 0.4, 0.24, 0.5, 38.0, 0.6),
    "steel-shaft": Material("shaft", "steel shaft", _rgb("#3a3d44"), 1.0, 0.35, 0.2, 0.55, 42.0, 0.7),
    "nylon-airline": Material("airline", "nylon airline", _rgb("#2b3a5c"), 1.0, 0.6, 0.28, 0.2, 14.0),
}
DEFAULT_MATERIAL = Material("other", "other graph parts", _rgb("#b0b0b8"), 0.35, 0.7, 0.25, 0.2, 14.0)
MATERIALS: list[Material] = ([m for _, m in MATERIAL_RULES]
                            + list(DECLARED_MATERIALS.values()) + [DEFAULT_MATERIAL])
MATERIAL_INDEX = {m.name: i for i, m in enumerate(MATERIALS)}

MOVING_KEYS = ("_piston", "_con_rod", "_piston_rod", "_crosshead", "crank_", "crankpin", "_valve_head", "_valve_stem",
               "_spring", "_retainer", "_bucket", "_rocker", "_pushrod", "cam_lobe", "_rotor", "_eccentric")
STATIC_OVERRIDES = ("_rotor_housing",)


def classify(name: str, declared_material: str | None = None) -> int:
    """Which material row this part draws with.

    The keyword table wins for engine hardware, because those names were
    chosen to carry that meaning (a part called `oil_filter` really is
    the lubrication row). A DECLARED material is the fallback for
    everything the table was not written for, which is the whole of
    machines.py."""
    for keys, mat in MATERIAL_RULES:
        if any(k in name for k in keys):
            return MATERIAL_INDEX[mat.name]
    if declared_material:
        mat = DECLARED_MATERIALS.get(str(declared_material))
        if mat is not None:
            return MATERIAL_INDEX[mat.name]
    return MATERIAL_INDEX[DEFAULT_MATERIAL.name]


def is_moving(name: str) -> bool:
    if any(k in name for k in STATIC_OVERRIDES):
        return False
    return any(k in name for k in MOVING_KEYS)


def wanted_in_view(name: str) -> bool:
    """Graph nodes/edges the engine view keeps: the dressing and the
    routed lines, not every harness wire and mount."""
    if "muffler" in name or "tailpipe" in name:
        # chassis plumbing (engine_parts.py marks these chassis_side):
        # real, in the graph for the exhaust circuit and the acoustic
        # model, but hung from the body, not the engine -- a metre of
        # tailpipe is not part of the engine's own view
        return False
    if name.startswith("node_"):
        return any(k in name for k in ("intake_plenum", "throttle_body", "air_filter", "fuel_rail", "distributor", "ignition_coil",
                                       "coil_pack", "leading_coil", "trailing_coil", "magneto", "oil_filter", "oil_reserve",
                                       "scavenge", "oil_bath", "glow_plug_bus", "exhaust_collector", "fuel_filter", "oil_pump",
                                       "mount_",
                                       # universal bolt-ons (engine_parts.py)
                                       "harmonic_balancer", "flywheel", "starter_motor", "recoil_starter", "crank_nose_fitting",
                                       "_pulley", "belt_tensioner", "timing_cover", "timing_drive", "expansion_bottle",
                                       "heater_core", "egr_valve", "pcv_valve", "bellhousing",
                                       # forced induction
                                       "blower_", "supercharger_rotor", "compressor_housing", "turbine_housing",
                                       "wastegate", "blow_off_valve", "charge_cooler", "downpipe", "barrel_valve",
                                       "fuel_pump",
                                       # the auxiliary plant (plant_parts.py): the air-treatment
                                       # train, the refrigerant loop's chillers and drier, the
                                       # hydraulic tank and its manifolds, the controls and the
                                       # accessory bank. Every one of these is a real body bolted
                                       # to the engine's own skid -- without them here the whole
                                       # plant was invisible to the view AND to the ray mesh, so
                                       # it could be neither seen nor shot.
                                       "plant.", "plant_",
                                       "accessory_battery", "charge_isolator",
                                       # aircraft / industrial / air-cooled
                                       "prop_reduction_gearbox", "prop_shaft", "prop_hub", "propeller_governor",
                                       "flyball_governor", "governor_latch", "injection_pump", "oil_cooler",
                                       "cooling_fan_housing", "cooling_tin_", "_baffle", "cowl_flap_", "magneto_2",
                                       # non-piston kinds
                                       "inlet_plenum", "compressor_impeller", "compressor_diffuser", "combustor", "turbine_ngv",
                                       "turbine_wheel", "power_turbine", "exhaust_duct", "fuel_control_unit", "ignition_exciter",
                                       "accessory_gearbox", "starter_generator", "bleed_valve",
                                       "motor_housing", "stator", "rotor_pack", "resolver", "inverter", "planetary_gearhead",
                                       "output_flange", "reduction_gearset", "drive_unit_differential", "hv_contactor_box", "dc_dc_converter",
                                       "boiler_barrel", "firebox", "steam_dome", "regulator_valve", "water_gauge", "chimney",
                                       "feedwater_injector", "air_reservoir_", "check_valve", "pressure_gauge", "reheater", "reverser_lever",
                                       "valve_gear_eccentric", "_rack", "slide_valve", "valve_eccentric", "pilot_burner",
                                       "captive_ball_governor", "small_engine_muffler", "primer_bulb", "water_hopper", "igniter_trip_lever",
                                       "node_fuel_bowl", "node_fuel_tank",
                                       "transaxle", "final_drive", "differential", "halfshaft",
                                       "air_box", "snorkel_inlet", "hose_end", "blower_case", "compression_brake",
                                       "pneumatic_"))
    if name.startswith("edge_"):
        return any(k in name for k in ("cylinder", "exhaust", "runner", "lead", "rail_feed", "oil", "pan", "trough", "scavenge",
                                       "air_filter", "stack", "coil", "throttle", "splash",
                                       # real visible lines from engine_parts.py: hoses, the EGR
                                       # tube, the timing run -- NOT the hub/bolted-joint edges,
                                       # which are joints, not pipes, and stay unlisted
                                       "heater_", "expansion_bottle", "egr_tube", "timing_chain",
                                       "blower_belt", "charge_pipe", "up_pipe", "turbine_to_downpipe",
                                       "blower_case_to_plenum_charge", "hat_nozzle", "barrel_valve",
                                       "cooling_tin_bank", "dry_sump_belt", "cooling_fan_belt", "oil_cooler",
                                       "injection_pump_to_rail",
                                       "to_compressor", "to_combustor", "to_turbine", "to_power_turbine", "to_exhaust_duct",
                                       "nozzle", "phase_busbar", "dome_to_regulator", "regulator_to_source", "tank_to_injector",
                                       "injector_to_barrel", "_to_source", "charging_to_check", "check_to_reservoir",
                                       "stage0_to_reheater", "eccentric_rod", "pilot_gas_line", "governor_belt",
                                       "collector_to_small_muffler", "primer_to_tank", "carb_pulse_line",
                                       "air_box", "snorkel", "hose_end", "blower_case_to_plenum_charge",
                                       "pneumatic_"))
    return True


@dataclass
class EngineMesh:
    vertices: np.ndarray                 # (V, 3)
    normals: np.ndarray                  # (V, 3)
    triangles: np.ndarray                # (T, 3) indices
    material_ids: np.ndarray             # (T,)
    part_names: list = field(default_factory=list)
    part_ranges: list = field(default_factory=list)   # (tri_start, tri_end) per part
    moving: bool = False
    # the real thermal-circuit membership each part already carries
    # (vehicle_mesh.SolidPart.thermal_group: "exhaust", "coolant", "oil",
    # "intake", "block_cyl_N", or None) -- kept per part so a renderer
    # can key a part's own blackbody self-emission off the REAL
    # temperature that group has this tick (engine_gl_view.py), instead
    # of every part of one material glowing together
    part_groups: list = field(default_factory=list)

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])

    def tri_vertices(self) -> np.ndarray:
        return self.vertices[self.triangles]          # (T, 3, 3)

    def tri_normals(self) -> np.ndarray:
        return self.normals[self.triangles]


def merge_engine_meshes(meshes: list[EngineMesh]) -> EngineMesh:
    """Join already-baked mesh objects without reinterpreting their parts.

    This is the scene-composition boundary for a complex object supplied by
    another subsystem: its triangles, material ids and thermal groups remain
    exactly the ones that subsystem baked.
    """
    live = [mesh for mesh in meshes if mesh is not None and mesh.n_triangles]
    if not live:
        return EngineMesh(np.zeros((0, 3)), np.zeros((0, 3)),
                          np.zeros((0, 3), dtype=np.int64),
                          np.zeros(0, dtype=np.int32), [], [], False, [])
    vertices, normals, triangles, materials = [], [], [], []
    names, ranges, groups = [], [], []
    vertex_offset = triangle_offset = 0
    for mesh in live:
        vertices.append(np.asarray(mesh.vertices))
        normals.append(np.asarray(mesh.normals))
        triangles.append(np.asarray(mesh.triangles) + vertex_offset)
        materials.append(np.asarray(mesh.material_ids))
        names.extend(mesh.part_names)
        ranges.extend((a + triangle_offset, b + triangle_offset)
                      for a, b in mesh.part_ranges)
        groups.extend(mesh.part_groups or [None] * len(mesh.part_names))
        vertex_offset += len(mesh.vertices)
        triangle_offset += mesh.n_triangles
    return EngineMesh(np.concatenate(vertices), np.concatenate(normals),
                      np.concatenate(triangles), np.concatenate(materials),
                      names, ranges, any(mesh.moving for mesh in live), groups)


def _from_parts(parts, moving: bool) -> EngineMesh:
    verts, norms, mats, names, ranges, groups = [], [], [], [], [], []
    t0 = 0
    for p in parts:
        n_tri = len(p.vertices) // 3
        if n_tri == 0:
            continue
        verts.append(np.asarray(p.vertices, dtype=np.float64))
        norms.append(np.asarray(p.normals, dtype=np.float64))
        mats.append(np.full(n_tri, classify(p.name, getattr(p, "declared_material", None)), dtype=np.int32))
        names.append(p.name); ranges.append((t0, t0 + n_tri)); t0 += n_tri
        groups.append(getattr(p, "thermal_group", None))
    if not verts:
        return EngineMesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int32), [], [], moving)
    v = np.concatenate(verts); n = np.concatenate(norms)
    tris = np.arange(len(v), dtype=np.int64).reshape(-1, 3)
    return EngineMesh(v, n, tris, np.concatenate(mats), names, ranges, moving, groups)


# ---------------------------------------------------------------------
# Blackbody self-emission: Planck's law, evaluated at three visible
# wavelengths, for any part hot enough to glow.
# ---------------------------------------------------------------------

# the spectral radiance L(lambda, T) = 2hc^2/lambda^5 / (exp(hc/(lambda k T)) - 1),
# sampled at the dominant wavelengths of the three display primaries. The
# COLOUR (the ratio between the three) and the RELATIVE brightness across
# temperature are real physics, unchanged. The one disclosed calibration
# is BLACKBODY_REFERENCE_K -> red-band radiance = 1.0 in the renderer's
# own light-rig units (those units are not physical -- the key/fill
# lights are scaled off the object's own size), so a single reference
# point is what maps real radiance onto that arbitrary scale: ~1000 K is
# where real steel is a plain, unmistakable cherry red.
_PLANCK_C2_UM_K = 14387.77           # h*c/k in micrometre-kelvin
_BLACKBODY_WAVELENGTHS_UM = (0.610, 0.550, 0.465)   # red, green, blue
BLACKBODY_REFERENCE_K = 1000.0
BLACKBODY_VISIBLE_FLOOR_K = 720.0    # below this the visible-band radiance is genuinely negligible
# the ONE scene-unit calibration: red-band radiance at the reference
# temperature, in the renderer's own (non-physical, size-relative) light
# units. 0.08 puts a 1000 K surface at the faint dull-red a real one shows
# in a dim bay, a 1150 K header at a plain orange glow (~1.8), and ~1300 K
# through yellow into white -- the real ladder, just pinned to this rig.
BLACKBODY_SCENE_GAIN = 0.08
BLACKBODY_EMISSION_CAP = 1000.0      # far past white-out already; keeps the SSBO floats sane


def _planck_relative(wavelength_um: float, temp_k: float) -> float:
    """Spectral radiance up to the common 2hc^2 constant (which cancels
    in every ratio taken here)."""
    if temp_k <= 0.0:
        return 0.0
    x = _PLANCK_C2_UM_K / (wavelength_um * temp_k)
    if x > 700.0:
        return 0.0
    return 1.0 / (wavelength_um ** 5 * math.expm1(x))


def blackbody_emission_rgb(temp_k: float) -> tuple[float, float, float]:
    """Linear RGB self-emission for a surface at temp_k, on the scale
    described above. (0, 0, 0) below the visible floor -- a 400 K oil
    pan is real, hot, and does not glow."""
    if temp_k < BLACKBODY_VISIBLE_FLOOR_K:
        return (0.0, 0.0, 0.0)
    ref = _planck_relative(_BLACKBODY_WAVELENGTHS_UM[0], BLACKBODY_REFERENCE_K)
    if ref <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(min(BLACKBODY_EMISSION_CAP, BLACKBODY_SCENE_GAIN * _planck_relative(w, temp_k) / ref)
                 for w in _BLACKBODY_WAVELENGTHS_UM)


def declared_in_view(graph: dict) -> set[str]:
    """Part names a graph has explicitly asked to be drawn.

    `wanted_in_view` below is a keyword allowlist written for engine
    hardware, and it was never going to know about a machine that is not
    an engine -- a scissor lift's cylinders and a turret's ring gear are
    not on it and never will be. Rather than growing that list forever
    (and matching parts by their names, which this project has already
    been bitten by), a node or edge can simply DECLARE `in_view`, and
    that declaration is honoured."""
    out: set[str] = set()
    for key, prefix in (("nodes", "node_"), ("edges", "edge_")):
        for item in graph.get(key, ()):
            if item.get("in_view"):
                part_name = prefix + str(item["identity"]).replace("/", "_").replace(".", "_")
                out.add(part_name)
                if item.get("moving_geometry"):
                    out.add(part_name + "_" + str(item["moving_geometry"]))
    return out


def build_engine_mesh(graph: dict, crank_angle_deg: float = 0.0, covers_off: bool = False) -> tuple[EngineMesh, EngineMesh]:
    """(static, moving) meshes for this graph at this crank angle."""
    declared = declared_in_view(graph)
    parts = [p for p in build_drivetrain_solid_parts(graph, crank_angle_deg=crank_angle_deg, covers_off=covers_off)
             if p.name in declared or wanted_in_view(p.name)]
    static = _from_parts([p for p in parts if not is_moving(p.name)], moving=False)
    moving = _from_parts([p for p in parts if is_moving(p.name)], moving=True)
    return static, moving


def build_moving_mesh(graph: dict, crank_angle_deg: float, covers_off: bool = False, spring_style: str = "helix") -> EngineMesh:
    """Only the moving parts, at this crank angle -- the per-frame call."""
    from cylinder_ports import deserialize_layout, build_parts_from_layout
    turbine_parts = build_agt1500_moving_parts(graph, crank_angle_deg)
    if turbine_parts:
        declared = declared_in_view(graph)
        return _from_parts([p for p in turbine_parts if p.name in declared], moving=True)
    layout_data = graph.get("cylinder_layout")
    if not layout_data:
        return _from_parts([], moving=True)
    parts = build_parts_from_layout(deserialize_layout(layout_data), crank_angle_deg=crank_angle_deg,
                                    covers_off=covers_off, moving_only=True, spring_style=spring_style)
    return _from_parts([p for p in parts if is_moving(p.name)], moving=True)


# ---------------------------------------------------------------------
# Throttle-plate animation: a SEPARATE small baked set, keyed by
# throttle position (0..1) instead of crank angle -- composited
# alongside whatever crank-angle frame is showing, not folded into it
# (a full crank-angle x throttle-position cross product would be N
# times the frame count for geometry that's a handful of triangles;
# indexing two small independent sets and drawing both is the same
# real "bake once, index during playback" contract as EngineAnimation,
# just on its own real driving parameter).
# ---------------------------------------------------------------------

def build_throttle_mesh(graph: dict, throttle_frac: float = 1.0) -> EngineMesh:
    parts = build_throttle_parts(graph, throttle_frac=throttle_frac)
    return _from_parts(parts, moving=True)


@dataclass
class ThrottleAnimation:
    fracs: np.ndarray
    frames: list                    # EngineMesh per fraction, baked upfront (cheap: a handful of triangles each)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def frame_index(self, throttle_frac: float) -> int:
        f = max(0.0, min(1.0, float(throttle_frac)))
        return int(np.argmin(np.abs(self.fracs - f)))

    def frame_for(self, throttle_frac: float):
        return self.frames[self.frame_index(throttle_frac)]


def build_throttle_animation(graph: dict, divisions: int = 9) -> ThrottleAnimation:
    """Baked once, upfront -- unlike EngineAnimation's incremental
    bake_next(), there's no per-frame cost worth spreading across
    ticks here (this is a plate and a lever arm, not a whole engine's
    moving parts)."""
    fracs = np.linspace(0.0, 1.0, max(2, divisions))
    frames = [build_throttle_mesh(graph, throttle_frac=float(f)) for f in fracs]
    return ThrottleAnimation(fracs=fracs, frames=frames)


def export_obj_mtl(static: EngineMesh, moving: EngineMesh, path_obj: str) -> tuple[str, str]:
    """Write the whole engine as OBJ + MTL with one material per rule
    (Kd = albedo, d = opacity, Ns = shininess, Ks = spec strength)."""
    path_mtl = os.path.splitext(path_obj)[0] + ".mtl"
    with open(path_mtl, "w") as f:
        for m in MATERIALS:
            f.write(f"newmtl {m.name}\nKd {m.albedo[0]:.4f} {m.albedo[1]:.4f} {m.albedo[2]:.4f}\n"
                    f"Ka {m.ambient:.3f} {m.ambient:.3f} {m.ambient:.3f}\nKs {m.spec_strength:.3f} {m.spec_strength:.3f} {m.spec_strength:.3f}\n"
                    f"Ns {m.shininess:.1f}\nd {m.opacity:.3f}\nillum 2\n\n")
    with open(path_obj, "w") as f:
        f.write(f"mtllib {os.path.basename(path_mtl)}\n")
        offset = 1
        for mesh in (static, moving):
            for k, name in enumerate(mesh.part_names):
                a, b = mesh.part_ranges[k]
                tri = mesh.triangles[a:b]
                verts = mesh.vertices[tri.reshape(-1)]
                norms = mesh.normals[tri.reshape(-1)]
                f.write(f"o {name}\nusemtl {MATERIALS[int(mesh.material_ids[a])].name}\n")
                for v in verts:
                    f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
                for n in norms:
                    f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
                for t in range(len(tri)):
                    i0 = offset + 3 * t
                    f.write(f"f {i0}//{i0} {i0 + 1}//{i0 + 1} {i0 + 2}//{i0 + 2}\n")
                offset += len(verts)
    return path_obj, path_mtl


def material_table() -> dict:
    """The material records in the spectral analyzer's chunk layout."""
    return {"names": [m.name for m in MATERIALS], "labels": [m.label for m in MATERIALS],
            "pbr": np.array([m.pbr_record() for m in MATERIALS], dtype=np.float32),
            "phong": np.array([m.phong_record() for m in MATERIALS], dtype=np.float32),
            "chunk_strides": {"pbr": 16, "phong": 8}}


# ---------------------------------------------------------------------
# Baked animation: moving parts at chosen crank divisions
# ---------------------------------------------------------------------

CYCLE_DEG = 720.0


@dataclass
class EngineAnimation:
    """The moving parts baked at a set of crank angles. A game plays
    these frames scaled by rpm (frame_for(angle)) and never re-derives
    geometry at run time; the divisions are the caller's choice -- an
    integer count evenly over one 720-degree cycle, or an explicit
    list of angles in degrees."""
    angles_deg: np.ndarray
    frames: list                    # EngineMesh per angle
    covers_off: bool

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    _graph: dict = None
    _detail: float = None

    @property
    def baked(self) -> int:
        return sum(1 for f in self.frames if f is not None)

    def bake_next(self) -> bool:
        """Bake the next missing frame (in an order that fills the cycle
        evenly: 0, half, quarters, ...). Returns False when complete."""
        missing = [i for i, f in enumerate(self.frames) if f is None]
        if not missing:
            return False
        n = len(self.frames)
        order = sorted(range(n), key=lambda i: (bin(i)[::-1].index("1") if i else -1, i)) if n > 1 else [0]
        # bit-reversed-ish fill: 0 first, then the middle, then quarters
        seq = [i for i in _even_fill_order(n) if self.frames[i] is None]
        i = seq[0]
        import mesh_primitives as mp
        prev = mp.DETAIL
        if self._detail is not None:
            mp.set_detail(self._detail)
        try:
            self.frames[i] = build_moving_mesh(self._graph, float(self.angles_deg[i]), covers_off=self.covers_off,
                                               spring_style=getattr(self, "_spring_style", "helix"))
        finally:
            mp.DETAIL = prev
        return any(f is None for f in self.frames)

    def frame_index(self, crank_angle_deg: float) -> int:
        a = float(crank_angle_deg) % CYCLE_DEG
        # nearest BAKED angle on the cycle (wrapping)
        d = np.abs((self.angles_deg - a + CYCLE_DEG / 2.0) % CYCLE_DEG - CYCLE_DEG / 2.0)
        d = np.where([f is None for f in self.frames], np.inf, d)
        return int(np.argmin(d))

    def frame_for(self, crank_angle_deg: float):
        return self.frames[self.frame_index(crank_angle_deg)]


def _even_fill_order(n: int) -> list:
    """0, n/2, n/4, 3n/4, ... -- so a partly baked animation already
    spans the cycle instead of clustering at its start."""
    if n <= 1:
        return [0]
    order, seen = [], set()
    step = n
    while step >= 1:
        for i in range(0, n, step):
            if i not in seen:
                order.append(i); seen.add(i)
        step //= 2
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order


DENSE_ANIMATION_STEP_DEG = 5.0
VALVE_OPEN_FRAC_OF_CYCLE = 0.32          # valvetrain_parts.py's own lift trace: open for this fraction of the cycle
DEFAULT_BURN_DURATION_DEG = 45.0


def significant_crank_angles(graph: dict, base_step_deg: float = DENSE_ANIMATION_STEP_DEG,
                             burn_duration_deg: float = DEFAULT_BURN_DURATION_DEG,
                             burn_phases: int = 8) -> list[float]:
    """The crank angles a baked animation must actually contain so that
    nothing the sim schedules falls between frames: a dense uniform
    grid, plus -- for every cylinder, off its own real throw phase --
    both dead centres, the intake and exhaust valve opening, peak-lift
    and closing angles (valvetrain_parts.py's own lift trace: opens at
    throw+180 / throw+540, open for 0.32 of the cycle), the firing TDC
    and each burn phase the combustion kernel bakes. Event angles
    already within 1 degree of a grid angle are not duplicated. Users
    render their own engine with this, so it is dense by default.

    A graph with NO cylinder layout -- a machine rather than an engine
    (machines.py) -- has no crank, so there is no such thing as a
    significant crank angle for it and every frame of a dense bake is an
    identical copy of the first. One frame is the whole animation, and
    baking a hundred and forty-four of them is what made a machine take
    minutes to appear on screen."""
    layout = graph.get("cylinder_layout") or []
    if not layout:
        return [0.0]
    # the grid step is made COMMENSURATE with the ignition schedule: the
    # firing interval (the cycle over the number of distinct firing
    # angles) divided into the fewest equal steps no coarser than
    # base_step_deg -- so every cylinder's dead centres, valve events and
    # burn phases land on the SAME relative frame phase (a V8's 90-degree
    # interval -> 5 degrees; a 14-cylinder's 51.43 -> 4.68), instead of
    # each cylinder's events scattering between grid lines differently
    fire_angles = set()
    for entry in layout:
        g = entry.get("geometry", entry) if isinstance(entry, dict) else (entry[0] if isinstance(entry, (list, tuple)) else entry)
        throw = float(g.get("throw_angle_deg", 0.0) if isinstance(g, dict) else getattr(g, "throw_angle_deg", 0.0))
        fire_angles.add(round((-throw) % CYCLE_DEG, 3))
    interval = CYCLE_DEG / max(1, len(fire_angles))
    step = interval / max(1, math.ceil(interval / base_step_deg - 1e-9))
    angles = set(float(a) for a in np.arange(0.0, CYCLE_DEG - 1e-9, step))
    for entry in layout:
        # the graph carries cylinder_ports' serialised form: {"geometry": {...}, "ports": [...]}
        g = entry.get("geometry", entry) if isinstance(entry, dict) else (entry[0] if isinstance(entry, (list, tuple)) else entry)
        throw = float(g.get("throw_angle_deg", 0.0) if isinstance(g, dict) else getattr(g, "throw_angle_deg", 0.0))
        events = []
        fire = (-throw) % CYCLE_DEG
        events += [fire, (fire + 180.0) % CYCLE_DEG, (fire + 360.0) % CYCLE_DEG, (fire + 540.0) % CYCLE_DEG]   # TDC/BDC x2
        open_len = VALVE_OPEN_FRAC_OF_CYCLE * CYCLE_DEG
        for base in (540.0, 180.0):   # intake opens at throw+180 -> crank 540-throw; exhaust at throw+540 -> 180-throw
            o = (base - throw) % CYCLE_DEG
            events += [o, (o + open_len * 0.5) % CYCLE_DEG, (o + open_len) % CYCLE_DEG]
        # the burn itself is covered by the grid (the combustion kernel
        # picks its own phase frame per tick, independent of these), so
        # only events more than half a step from any grid frame are
        # added -- on a commensurate grid that is normally none of them
        for a in events:
            if all(min(abs(a - b), CYCLE_DEG - abs(a - b)) > step * 0.5 for b in angles):
                angles.add(a)
    return sorted(angles)


def animation_angles(divisions) -> np.ndarray:
    if isinstance(divisions, int):
        return np.linspace(0.0, CYCLE_DEG, max(1, divisions), endpoint=False)
    return np.array(sorted(float(a) % CYCLE_DEG for a in divisions), dtype=np.float64)


def build_animation(graph: dict, divisions=24, covers_off: bool = False, detail: float | None = None,
                    spring_style: str = "helix") -> EngineAnimation:
    """Bake the moving parts once at every requested crank angle."""
    anim = start_animation(graph, divisions, covers_off, detail, spring_style)
    while anim.bake_next():
        pass
    return anim


def start_animation(graph: dict, divisions=24, covers_off: bool = False, detail: float | None = None,
                    spring_style: str = "helix") -> EngineAnimation:
    """An animation with only its first frame baked; bake_next() fills
    the rest one frame at a time (a live view calls it between renders
    so an engine switch never stalls the screen). Until a frame is
    baked, frame_for() returns the nearest baked one."""
    angles = animation_angles(divisions)
    anim = EngineAnimation(angles_deg=angles, frames=[None] * len(angles), covers_off=covers_off)
    anim._graph = graph
    anim._detail = detail
    anim._spring_style = spring_style
    anim.bake_next()
    return anim
